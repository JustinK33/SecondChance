"""``evaluate_eligibility``: the only node that decides anything.

**No model call happens here, and none ever may.** The status a user reads is a
pure function of their profile and an approved rule set, which is what makes it
reproducible, explainable months later, and defensible. The model's entire role
in this product starts one node later, wording a verdict it cannot change.
``tests/integration/test_workflow_graph.py`` asserts the model client recorded
zero calls across this node.

Selective re-evaluation. On the first pass every candidate with an approved
rule set is evaluated. On a re-entry after targeted clarification only the
evaluations whose recorded evidence *read or missed* a changed field are
recomputed - :func:`app.eligibility.engine.reevaluate_affected` returns the
others untouched, as the same object rather than a re-derived copy. Editing
household size must not re-evaluate, re-explain and re-bill a program whose
rules never looked at household size.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from app.core.logging import get_logger
from app.schemas import EligibilityEvaluation, MatchStatus
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import entering, profile_of
from app.workflows.questions import not_already_declined
from app.workflows.state import MAX_CLARIFICATION_ROUNDS, NavigatorState, StateUpdate

__all__ = ["NODE", "evaluate_eligibility", "ranked_missing_fields"]

NODE = "evaluate_eligibility"

_log = get_logger(__name__)


def ranked_missing_fields(evaluations: Sequence[EligibilityEvaluation]) -> list[str]:
    """Fields to ask about, most-unblocking first.

    Ranked by how many programs an answer would unblock, then by how often the
    field appears, then alphabetically so the list is stable for a given set of
    evaluations. Only evaluations at ``MORE_INFO_NEEDED`` contribute: a field
    missing from a program that was ruled out on other grounds is not worth a
    question.
    """
    programs: Counter[str] = Counter()
    for evaluation in evaluations:
        if evaluation.status is not MatchStatus.MORE_INFO_NEEDED:
            continue
        for field in dict.fromkeys(evaluation.missing_fields):
            programs[field] += 1
    return [field for field, _ in sorted(programs.items(), key=lambda item: (-item[1], item[0]))]


async def evaluate_eligibility(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Run the deterministic engine over every candidate with usable rules."""
    update = entering(NODE)
    profile = profile_of(state)
    candidates = list(state.get("candidates") or ())
    previous = list(state.get("evaluations") or ())
    changed = list(state.get("changed_fields") or ())
    rounds = state.get("clarification_rounds", 0)

    reused = 0
    if previous and changed:
        try:
            evaluations = deps.engine.reevaluate_affected(
                previous, profile, changed, candidates=candidates
            )
        except ValueError:
            # A previously evaluated program whose rule set is no longer among
            # the candidates. Recomputing everything is correct and cheap; a
            # stale verdict would not be.
            _log.info("workflow.evaluate.full_recompute", node=NODE, reason="rules_unavailable")
            evaluations = deps.engine.evaluate_many(profile, candidates)
        else:
            reused = sum(
                1 for before, after in zip(previous, evaluations, strict=False) if before is after
            )
    else:
        evaluations = deps.engine.evaluate_many(profile, candidates)

    missing = ranked_missing_fields(evaluations)
    # Only claim critical information is missing when there is a question we
    # could honestly ask about it. Otherwise the targeted pause would open and
    # immediately close, costing the user a round trip for nothing.
    blocked = [
        evaluation
        for evaluation in evaluations
        if evaluation.status is MatchStatus.MORE_INFO_NEEDED
    ]
    askable = bool(
        not_already_declined(
            deps.engine.missing_field_questions(blocked, run_id=state["run_id"]), profile
        )
    )
    missing_critical_info = bool(blocked) and askable and rounds < MAX_CLARIFICATION_ROUNDS

    counts = Counter(evaluation.status.value for evaluation in evaluations)
    _log.info(
        "workflow.evaluate.done",
        node=NODE,
        evaluated=len(evaluations),
        reused=reused,
        candidates=len(candidates),
        statuses=dict(counts),
        missing=missing,
        missing_critical_info=missing_critical_info,
        clarification_rounds=rounds,
        model_calls=0,
    )

    update["evaluations"] = list(evaluations)
    update["eligibility_missing_fields"] = missing
    update["missing_critical_info"] = missing_critical_info
    if missing:
        update["missing_fields"] = missing

    limitations = [
        limitation for evaluation in evaluations for limitation in evaluation.limitations
    ]
    skipped = len(candidates) - len(evaluations)
    if skipped > 0:
        limitations.append(
            f"{skipped} of the programs we found do not have reviewed rules yet, "
            "so we describe them without assessing them."
        )
    if limitations:
        update["limitations"] = list(dict.fromkeys(limitations))
    return update
