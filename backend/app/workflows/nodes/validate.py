"""``validate_profile``: build the sanitised profile and decide whether to ask.

Deterministic from end to end. No model call happens here, which matters
because this is the node that decides whether to interrupt the run and put a
question in front of a person - a decision that must be reproducible and
reviewable, not a matter of what a model felt like proposing.

The bar for asking is deliberately high: a question is only worth a pause when
the answer would *materially* change matching. See
:data:`app.workflows.questions.MATERIAL_FIELDS` for what qualifies and why. A
field the user has already declined is in ``profile.unknown_fields`` and is
never asked about again.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import entering
from app.workflows.questions import intake_questions, material_missing_fields
from app.workflows.state import MAX_CLARIFICATION_ROUNDS, NavigatorState, StateUpdate

__all__ = ["NODE", "validate_profile"]

NODE = "validate_profile"

_log = get_logger(__name__)


async def validate_profile(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Assemble the profile, record what is unknown, decide on a clarification.

    Runs twice when the first clarification happens: once before the question
    and once after the answer. The second pass sees the answers at the top of
    the precedence order, which is what turns "we need one more detail" into a
    filled-in field rather than a repeated question.
    """
    update = entering(NODE)
    profile, changed = deps.assemble(
        state.get("extracted"),
        base=state.get("profile"),
        prefilled=state.get("prefilled"),
        answers=state.get("clarification_answers", ()),
        min_confidence=deps.min_confidence,
    )

    missing = material_missing_fields(profile)
    rounds = state.get("clarification_rounds", 0)
    askable = intake_questions(profile, missing, run_id=state["run_id"]) if missing else []
    # Only the first round may be spent here. The second is reserved for the
    # evidence-driven pass after evaluation, which can prove that an answer
    # would move a specific program off MORE_INFO_NEEDED.
    needs_clarification = bool(askable) and rounds < MAX_CLARIFICATION_ROUNDS - 1

    _log.info(
        "workflow.validate.done",
        node=NODE,
        missing=sorted(missing),
        unknown=sorted(profile.unknown_fields),
        changed=sorted(changed),
        needs_clarification=needs_clarification,
        clarification_rounds=rounds,
    )

    update["profile"] = profile
    update["profile_missing_fields"] = missing
    update["needs_clarification"] = needs_clarification
    update["changed_fields"] = changed
    if missing:
        update["missing_fields"] = missing
    return update
