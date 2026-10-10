"""The typed LangGraph state for the benefits-navigator workflow.

One rule governs this module, and the test suite enforces it:

    **A channel written by two nodes that can run at the same time must
    declare a reducer. Every other channel has exactly one writer.**

LangGraph raises ``InvalidUpdateError`` when two tasks in one superstep write
the same reducer-less channel, so this is not a style preference - it is the
difference between a graph that runs and a graph that dies halfway through a
fan-out. Four channels take a reducer (:data:`REDUCER_FIELDS`); everything
else names its writer in a comment above the field and in
:data:`FIELD_WRITERS`, which ``tests/integration/test_workflow_graph.py``
checks against the writes the nodes actually make.

``phase`` deserves its own note. ``docs/INTERFACES.md`` says "every node sets
its own phase on entry; last write wins", which is true of the nodes that run
*serially* but cannot be true of a fan-out branch: three concurrent writes to a
reducer-less channel is exactly the error above. So the branch nodes of the two
fan-outs deliberately do not touch ``phase``; the barrier that joins them
(:func:`~app.workflows.nodes.retrieve.join_candidates`,
:func:`~app.workflows.nodes.merge.merge_outputs`) writes the phase that
describes the whole group. See the report note in ``PHASE_BY_NODE``.

Privacy: ``narrative``, ``extracted``, ``profile`` and ``clarification_answers``
are personal content. They live in the checkpoint row for one run and one user
and are never logged. Nothing in this module is loggable except ids, counts and
enum values.
"""

from __future__ import annotations

import operator
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Final, TypedDict

from app.schemas import (
    ActionItemDraft,
    ActionPlan,
    ActionPlanDraft,
    CitationVerificationReport,
    ClarificationAnswer,
    ClarificationQuestion,
    EligibilityEvaluation,
    ExtractedProfile,
    MatchExplanation,
    ModelUsage,
    NeedCategory,
    ProfileDraft,
    ProgramCategory,
    ProgramMatch,
    RetrievedProgram,
    SharedDocument,
    UserFacingError,
    UserProfile,
    WorkflowPhase,
    WorkflowStatus,
)

__all__ = [
    "FIELD_WRITERS",
    "MAX_CLARIFICATION_ROUNDS",
    "PARALLEL_GROUPS",
    "PHASE_BY_NODE",
    "REDUCER_FIELDS",
    "NavigatorState",
    "StateUpdate",
    "build_initial_state",
    "merge_unique",
]

#: Hard ceiling on clarification rounds for one run. The first gate
#: (``validate_profile``) may spend round 1 and the second gate
#: (``evaluate_eligibility``) may spend round 2; after that the run completes
#: with whatever is known and the affected programs stay at
#: ``MORE_INFO_NEEDED``. A workflow that can ask forever is a workflow that
#: never produces an answer.
MAX_CLARIFICATION_ROUNDS: Final[int] = 2


def merge_unique(current: Sequence[str] | None, update: Sequence[str] | None) -> list[str]:
    """Order-preserving unique merge of two string lists.

    Used for ``missing_fields`` and ``limitations``, both of which are written
    by several nodes including concurrent ones. Plain ``operator.add`` would
    work but would accumulate duplicates - the same missing field reported by
    two programs, the same caveat recorded by two branches - and a user-facing
    list that repeats itself reads like a bug.
    """
    merged: dict[str, None] = {}
    for value in (*(current or ()), *(update or ())):
        merged.setdefault(value, None)
    return list(merged)


class NavigatorState(TypedDict, total=False):
    """Everything one workflow run knows, as one checkpointed dictionary."""

    # ---------------------------------------------------------- initial state
    # Sole writer: build_initial_state (called by workflow_service.create_run).
    request_id: str
    # Sole writer: build_initial_state. The verified token's subject.
    user_id: str
    # Sole writer: build_initial_state. Also the checkpointer's thread_id.
    run_id: uuid.UUID
    # Sole writer: build_initial_state. Personal content; never logged.
    narrative: str
    # Sole writer: build_initial_state. Structured answers given up front.
    prefilled: ProfileDraft | None
    # Sole writer: workflow_service, outside the graph. Nodes never write it.
    status: WorkflowStatus

    # ------------------------------------------------------------------ phase
    # Writers: every *serially* executing node, on entry. Fan-out branches must
    # not write it - see the module docstring.
    phase: WorkflowPhase

    # --------------------------------------------------------- extract_intake
    # Sole writer: extract_intake. Raw model output; personal content.
    extracted: ExtractedProfile | None
    # Sole writer: extract_intake. The projection onto UserProfile field names.
    extracted_fields: dict[str, Any]

    # -------------------------------------------------------- validate_profile
    # Writers: validate_profile, then request_targeted_clarification when it
    # applies answers. The two are strictly serial - the second runs only after
    # evaluate_eligibility, which runs only after the first - so no reducer is
    # needed and none would be correct.
    profile: UserProfile | None
    # Sole writer: validate_profile. Fields missing *before* any evaluation.
    profile_missing_fields: list[str]
    # Sole writer: validate_profile. Read by the needs_clarification predicate.
    needs_clarification: bool
    # Writers: validate_profile, request_targeted_clarification. Serial, as
    # above. Drives selective re-evaluation.
    changed_fields: list[str]

    # ------------------------------------------------- clarification handshake
    # Writers: request_clarification, request_targeted_clarification. Serial.
    clarification_questions: list[ClarificationQuestion]
    # Writers: the same two nodes, from their Command(resume=...) payload.
    # Personal content; never logged.
    clarification_answers: list[ClarificationAnswer]
    # Writers: the same two nodes. Bounded by MAX_CLARIFICATION_ROUNDS.
    clarification_rounds: int

    # ------------------------------------------- first fan-out (three branches)
    # Sole writer: retrieve_programs.
    candidates: list[RetrievedProgram]
    # Sole writer: validate_profile_evidence.
    evidence_warnings: list[str]
    # Sole writer: categorize_needs.
    need_categories: list[NeedCategory]
    # Sole writer: categorize_needs.
    program_categories: list[ProgramCategory]
    # Sole writer: join_candidates. What each branch contributed, for audit.
    branch_contributions: dict[str, int]

    # --------------------------------------------------- evaluate_eligibility
    # Sole writer: evaluate_eligibility. The engine's verdicts; no model input.
    evaluations: list[EligibilityEvaluation]
    # Sole writer: evaluate_eligibility. Separate from profile_missing_fields so
    # the two clarification passes cannot clobber each other.
    eligibility_missing_fields: list[str]
    # Sole writer: evaluate_eligibility. Read by the targeted-clarification
    # predicate.
    missing_critical_info: bool

    # ------------------------------------------ second fan-out (three branches)
    # Sole writer: generate_explanations. Keyed by program_slug.
    explanations: dict[str, MatchExplanation]
    # Sole writer: generate_action_plan.
    plan_draft: ActionPlanDraft | None
    # Sole writer: generate_action_plan. Drafts before identity and ordering.
    action_items: list[ActionItemDraft]
    # Sole writer: verify_citations.
    citation_results: CitationVerificationReport | None

    # ------------------------------------------------------- the integrity gate
    # Sole writer: merge_outputs.
    matches: list[ProgramMatch]
    # Sole writer: merge_outputs.
    plan: ActionPlan | None
    # Sole writer: merge_outputs.
    shared_documents: list[SharedDocument]
    # Sole writer: merge_outputs. Every removal the gate made, so the result is
    # auditable rather than merely clean.
    validation: dict[str, Any]

    # -------------------------------------------------------- persist_results
    # Sole writer: persist_results. Row counts only.
    persisted: dict[str, int]

    # ---------------------------------------------------------------- reducers
    #: Union of every field we would like an answer to. Written by
    #: validate_profile and evaluate_eligibility.
    missing_fields: Annotated[list[str], merge_unique]
    #: Run-level caveats. Written by any node, including concurrent ones.
    limitations: Annotated[list[str], merge_unique]
    #: Recoverable failures a node absorbed instead of killing the run.
    errors: Annotated[list[UserFacingError], operator.add]
    #: Token and cost telemetry. Written by any node that calls a model.
    model_usage: Annotated[list[ModelUsage], operator.add]

    # ------------------------------------------------------------------- fatal
    # Sole writer: run_workflow / resume_workflow, outside the node set. A node
    # that cannot recover raises instead, and the helper records the failure
    # here so no partially-built result is ever persisted.
    error: UserFacingError | None


#: A partial state returned by a node. ``dict`` rather than ``NavigatorState``
#: because mypy cannot express "some keys of this TypedDict" and a node that
#: returns three of thirty keys should not have to lie about the other
#: twenty-seven.
StateUpdate = dict[str, Any]

#: The four channels that declare a reducer, and therefore the only four that
#: more than one concurrent node may write.
REDUCER_FIELDS: Final[frozenset[str]] = frozenset(
    {"missing_fields", "limitations", "errors", "model_usage"}
)

#: Node name -> the reducer-less channels that node is allowed to write.
#: ``tests/integration/test_workflow_graph.py`` records every write a real run
#: makes and asserts it against this table, so the comments above are checked
#: rather than merely written.
FIELD_WRITERS: Final[dict[str, frozenset[str]]] = {
    "extract_intake": frozenset({"phase", "extracted", "extracted_fields"}),
    "validate_profile": frozenset(
        {"phase", "profile", "profile_missing_fields", "needs_clarification", "changed_fields"}
    ),
    "request_clarification": frozenset(
        {"phase", "clarification_questions", "clarification_answers", "clarification_rounds"}
    ),
    "retrieve_programs": frozenset({"candidates"}),
    "validate_profile_evidence": frozenset({"evidence_warnings"}),
    "categorize_needs": frozenset({"need_categories", "program_categories"}),
    "join_candidates": frozenset({"phase", "branch_contributions"}),
    "evaluate_eligibility": frozenset(
        {"phase", "evaluations", "eligibility_missing_fields", "missing_critical_info"}
    ),
    "request_targeted_clarification": frozenset(
        {
            "phase",
            "clarification_questions",
            "clarification_answers",
            "clarification_rounds",
            "profile",
            "changed_fields",
        }
    ),
    "generate_explanations": frozenset({"explanations"}),
    "generate_action_plan": frozenset({"plan_draft", "action_items"}),
    "verify_citations": frozenset({"citation_results"}),
    "merge_outputs": frozenset({"phase", "matches", "plan", "shared_documents", "validation"}),
    "persist_results": frozenset({"phase", "persisted"}),
}

#: The two sets of nodes that run concurrently. Every pair inside a group must
#: write a disjoint set of reducer-less channels.
PARALLEL_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    ("retrieve_programs", "validate_profile_evidence", "categorize_needs"),
    ("generate_explanations", "generate_action_plan", "verify_citations"),
)

#: The phase each serially-executing node publishes. A fan-out branch has no
#: entry: it cannot write ``phase`` (see the module docstring), so the barrier
#: node publishes the phase that describes its whole group.
PHASE_BY_NODE: Final[dict[str, WorkflowPhase]] = {
    "extract_intake": WorkflowPhase.EXTRACT_PROFILE,
    "validate_profile": WorkflowPhase.VALIDATE_PROFILE,
    "request_clarification": WorkflowPhase.AWAIT_CLARIFICATION,
    "join_candidates": WorkflowPhase.CATEGORIZE_NEEDS,
    "evaluate_eligibility": WorkflowPhase.EVALUATE_ELIGIBILITY,
    "request_targeted_clarification": WorkflowPhase.AWAIT_TARGETED_CLARIFICATION,
    "merge_outputs": WorkflowPhase.MERGE_OUTPUTS,
    "persist_results": WorkflowPhase.DONE,
}


def build_initial_state(
    *,
    run_id: uuid.UUID,
    user_id: str,
    request_id: str,
    narrative: str,
    prefilled: ProfileDraft | None = None,
    status: WorkflowStatus = WorkflowStatus.RUNNING,
    profile: UserProfile | None = None,
) -> NavigatorState:
    """The state a fresh run starts from.

    Every reducer channel is seeded with an empty list. LangGraph can infer an
    empty value for a ``list`` annotation, but being explicit means a node is
    never the first thing to touch a channel, which keeps the first write and
    the hundredth write identical in behaviour.

    :param profile: an existing profile to start from, for the
        "edit my information and re-run" flow. ``validate_profile`` treats it
        as the base that extraction may fill gaps in but never overrides.
    """
    return NavigatorState(
        request_id=request_id,
        user_id=user_id,
        run_id=run_id,
        narrative=narrative,
        prefilled=prefilled,
        status=status,
        phase=WorkflowPhase.PENDING,
        profile=profile,
        clarification_answers=[],
        clarification_rounds=0,
        changed_fields=[],
        missing_fields=[],
        limitations=[],
        errors=[],
        model_usage=[],
        error=None,
    )
