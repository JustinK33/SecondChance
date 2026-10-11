"""The two places the graph stops and asks the person a question.

``request_clarification`` runs before retrieval, when a field almost every rule
reads is missing. ``request_targeted_clarification`` runs after evaluation,
when the engine can prove that one answer would move a specific program off
``MORE_INFO_NEEDED``. The second is the better question and the first exists
only because, before retrieval, there is nothing yet to prove anything against.

Both call :func:`langgraph.types.interrupt`, which checkpoints the run and
returns from ``ainvoke`` with an ``__interrupt__`` key rather than raising. The
payload shape is the contract with the API layer and is documented on
:func:`clarification_payload`.

Bounds. Each pause spends one of :data:`MAX_CLARIFICATION_ROUNDS` rounds, and a
node that cannot think of an honest question spends the entire remaining budget
instead of looping: a workflow that can ask forever is a workflow that never
produces an answer. Nothing here can ask for an SSN, bank details or an
immigration status, because a question's ``field`` is validated against the
closed :data:`~app.schemas.profile.PROFILE_FIELD_NAMES` set and no such field
exists.

Skipping is durable. Every question is optional, so leaving one alone is a
choice, and ``request_targeted_clarification`` records it as the decline it is
rather than asking the same question again next round. See that function and
:func:`~app.workflows.questions.declines_for_unanswered`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from app.core.logging import get_logger
from app.schemas import ClarificationAnswer, ClarificationQuestion, MatchStatus
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import entering, profile_of
from app.workflows.questions import (
    declines_for_unanswered,
    intake_questions,
    not_already_declined,
)
from app.workflows.state import MAX_CLARIFICATION_ROUNDS, NavigatorState, StateUpdate

__all__ = [
    "FIRST_NODE",
    "TARGETED_NODE",
    "clarification_payload",
    "parse_answers",
    "request_clarification",
    "request_targeted_clarification",
]

FIRST_NODE = "request_clarification"
TARGETED_NODE = "request_targeted_clarification"

_log = get_logger(__name__)

_NOTHING_TO_ASK = (
    "We could not think of a question whose answer would change your results, "
    "so we went ahead with what we know."
)

#: Said instead of :data:`_NOTHING_TO_ASK` when there *was* something left to
#: ask and the only reason we are not asking is that the person already
#: declined it. Claiming we had no question would be untrue, and a caveat that
#: misstates why a result is incomplete is worse than no caveat at all.
_ALREADY_DECLINED = (
    "The optional questions still outstanding are ones you chose not to answer, "
    "so we did not ask again and some programs still need more information."
)


def clarification_payload(
    kind: str,
    *,
    run_id: uuid.UUID,
    round_number: int,
    questions: Sequence[ClarificationQuestion],
) -> dict[str, Any]:
    """The value handed to ``interrupt()``, and therefore to the API layer.

    ``WorkflowRunRepository.mark_awaiting_input`` stores this verbatim as the
    run's ``clarification_payload``, so a poll can render the questions without
    reading the checkpoint. It holds editorial text and field names only -
    questions, labels, why-it-matters copy - and never an answer, a narrative
    or anything else the user told us.
    """
    return {
        "kind": kind,
        "run_id": str(run_id),
        "round": round_number,
        "fields": [question.field for question in questions],
        "questions": [question.model_dump(mode="json") for question in questions],
    }


def parse_answers(payload: object) -> list[ClarificationAnswer]:
    """Validate a ``Command(resume=...)`` payload into answers.

    The canonical shape is ``{"answers": [<ClarificationAnswer>, ...]}``, which
    is what ``WorkflowResumeRequest`` serialises to. A bare list, a single
    answer and ``None`` are also accepted, because a resume is the one place a
    caller is most likely to send the obvious thing rather than the documented
    thing.

    An entry that fails validation is dropped rather than raising: the user
    answered the other three questions and is entitled to their results.
    """
    raw: object = payload
    if isinstance(raw, Mapping):
        raw = raw.get("answers", [])
    if isinstance(raw, ClarificationAnswer):
        return [raw]
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return []

    answers: list[ClarificationAnswer] = []
    dropped = 0
    for entry in raw:
        if isinstance(entry, ClarificationAnswer):
            answers.append(entry)
            continue
        try:
            answers.append(ClarificationAnswer.model_validate(entry))
        except ValidationError:
            dropped += 1
    if dropped:
        # A count, not the payload: an answer is personal content.
        _log.warning("workflow.clarification.answers_dropped", dropped=dropped)
    return answers


def _answered(
    node: str,
    *,
    questions: Sequence[ClarificationQuestion],
    answers: Sequence[ClarificationAnswer],
    round_number: int,
) -> StateUpdate:
    _log.info(
        "workflow.clarification.answered",
        node=node,
        asked=len(questions),
        answered=sum(1 for answer in answers if not answer.skipped),
        skipped=sum(1 for answer in answers if answer.skipped),
        round=round_number,
    )
    return {
        "clarification_questions": list(questions),
        "clarification_answers": list(answers),
        "clarification_rounds": round_number,
    }


async def request_clarification(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Pause for the material fields that are missing before retrieval."""
    update = entering(FIRST_NODE)
    run_id = state["run_id"]
    round_number = state.get("clarification_rounds", 0) + 1
    profile = state.get("profile")
    questions = (
        intake_questions(profile, state.get("profile_missing_fields", []), run_id=run_id)
        if profile is not None
        else []
    )

    if not questions:
        # Nothing honest to ask. Spend the whole budget so the routing
        # predicate cannot send us back here.
        update["clarification_questions"] = []
        update["clarification_answers"] = []
        update["clarification_rounds"] = MAX_CLARIFICATION_ROUNDS
        update["limitations"] = [_NOTHING_TO_ASK]
        return update

    _log.info(
        "workflow.clarification.asked",
        node=FIRST_NODE,
        count=len(questions),
        fields=[question.field for question in questions],
        round=round_number,
    )
    resume = interrupt(
        clarification_payload(
            "clarification", run_id=run_id, round_number=round_number, questions=questions
        )
    )
    answers = parse_answers(resume)
    update.update(
        _answered(FIRST_NODE, questions=questions, answers=answers, round_number=round_number)
    )
    return update


async def request_targeted_clarification(
    state: NavigatorState, *, deps: WorkflowDeps
) -> StateUpdate:
    """Pause for the fields that are actually blocking a ``POTENTIAL_MATCH``.

    The questions come from
    :meth:`app.eligibility.engine.RuleEngine.missing_field_questions`, derived
    from the conditions that really resolved ``UNKNOWN`` and ranked by how many
    programs an answer would unblock. Only evaluations sitting at
    ``MORE_INFO_NEEDED`` are considered, so a program already ruled out cannot
    generate a question - answering it would change nothing.

    This node also re-applies the answers to the profile, because the loop goes
    straight back to ``evaluate_eligibility`` and never through
    ``validate_profile`` again. The two writes to ``profile`` are strictly
    serial; see :mod:`app.workflows.state`.

    A question this gate already presented is never presented again. An answer
    that came back resolves its field; a question that came back untouched is
    recorded as declined (:func:`~app.workflows.questions.declines_for_unanswered`)
    and therefore lands in ``profile.unknown_fields``, which
    :func:`~app.workflows.questions.not_already_declined` reads on the next
    pass. So a second round surfaces only fields the person has not already
    seen and passed on, and the programs behind a declined field stay at
    ``MORE_INFO_NEEDED`` - the honest outcome of not knowing.
    """
    update = entering(TARGETED_NODE)
    run_id = state["run_id"]
    round_number = state.get("clarification_rounds", 0) + 1
    blocked = [
        evaluation
        for evaluation in state.get("evaluations", [])
        if evaluation.status is MatchStatus.MORE_INFO_NEEDED
    ]
    provable = deps.engine.missing_field_questions(blocked, run_id=run_id)
    questions = not_already_declined(provable, profile_of(state))

    if not questions or round_number > MAX_CLARIFICATION_ROUNDS:
        update["clarification_questions"] = []
        update["clarification_answers"] = []
        update["clarification_rounds"] = MAX_CLARIFICATION_ROUNDS
        if not questions:
            update["limitations"] = [_ALREADY_DECLINED if provable else _NOTHING_TO_ASK]
        return update

    _log.info(
        "workflow.clarification.asked",
        node=TARGETED_NODE,
        count=len(questions),
        fields=[question.field for question in questions],
        blocked_programs=len(blocked),
        round=round_number,
    )
    resume = interrupt(
        clarification_payload(
            "targeted_clarification",
            run_id=run_id,
            round_number=round_number,
            questions=questions,
        )
    )
    answers = parse_answers(resume)
    # Everything this gate showed and did not get back is a decline, and is
    # recorded as one. Without it the edge back to ``evaluate_eligibility``
    # returns here, re-derives the same unresolved fields and asks the identical
    # questions - same deterministic ids - in the next round, which tells the
    # person their skip did not count. Only this gate does it:
    # ``request_clarification`` cannot re-ask itself (``validate_profile`` opens
    # it only while no round has been spent) and its pre-retrieval questions are
    # guesses, so a field it raised still deserves the one evidence-backed
    # asking it gets here.
    implicit = declines_for_unanswered(questions, answers)
    if implicit:
        _log.info(
            "workflow.clarification.unanswered_declined",
            node=TARGETED_NODE,
            fields=[answer.field for answer in implicit],
            round=round_number,
        )
    answers.extend(implicit)
    update.update(
        _answered(TARGETED_NODE, questions=questions, answers=answers, round_number=round_number)
    )

    profile, changed = deps.assemble(
        state.get("extracted"),
        base=state.get("profile"),
        prefilled=state.get("prefilled"),
        answers=answers,
        min_confidence=deps.min_confidence,
    )
    update["profile"] = profile
    update["changed_fields"] = changed
    if changed:
        _log.info("workflow.clarification.profile_updated", changed=sorted(changed))
    return update
