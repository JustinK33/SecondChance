"""The workflow graph: topology, routing predicates and the run helpers.

Topology
--------
::

    START
      -> extract_intake
      -> validate_profile
           -- needs_clarification --> request_clarification --> validate_profile
           -- otherwise -------------> retrieve_programs
                                       validate_profile_evidence   (concurrent)
                                       categorize_needs
                                         -> join_candidates
      -> evaluate_eligibility
           -- wants_targeted_clarification --> request_targeted_clarification
                                               --> evaluate_eligibility
           -- otherwise --------------------->  generate_explanations
                                                generate_action_plan   (concurrent)
                                                verify_citations
                                                  -> merge_outputs
      -> persist_results -> END

Both fan-outs are real: the three branches of each run in one superstep, and
the node they all point at runs once, after every one of them has committed.
LangGraph gives that for free - a node with three incoming edges is scheduled
when all three fire - which is why ``join_candidates`` and ``merge_outputs``
need no coordination code of their own.

Both loops are bounded by ``clarification_rounds``, which no node can push past
:data:`~app.workflows.state.MAX_CLARIFICATION_ROUNDS`. A node that cannot think
of an honest question spends the remaining budget instead of looping, so
neither cycle can run twice for the same reason.

The routing predicates are plain functions of state with no dependencies, so
``tests/integration/test_workflow_graph.py`` exercises them directly rather
than inferring their behaviour from a whole run.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Final

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.core.errors import AppError, UpstreamTimeoutError
from app.core.logging import get_logger
from app.schemas import (
    ActionPlan,
    ClarificationAnswer,
    ModelUsage,
    ProgramMatch,
    UserFacingError,
    WorkflowPhase,
    WorkflowStatus,
)
from app.workflows.checkpointer import get_checkpointer, thread_config
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import log_failure, user_facing
from app.workflows.nodes.clarify import request_clarification, request_targeted_clarification
from app.workflows.nodes.evaluate import evaluate_eligibility
from app.workflows.nodes.generate import (
    generate_action_plan,
    generate_explanations,
    verify_citations,
)
from app.workflows.nodes.intake import extract_intake
from app.workflows.nodes.merge import merge_outputs
from app.workflows.nodes.persist import persist_results
from app.workflows.nodes.retrieve import (
    categorize_needs,
    join_candidates,
    retrieve_programs,
    validate_profile_evidence,
)
from app.workflows.nodes.validate import validate_profile
from app.workflows.state import (
    MAX_CLARIFICATION_ROUNDS,
    NavigatorState,
    build_initial_state,
    merge_unique,
)

__all__ = [
    "GENERATION_BRANCHES",
    "NODE_FUNCTIONS",
    "RETRIEVAL_BRANCHES",
    "WorkflowOutcome",
    "build_graph",
    "build_initial_state",
    "clarification_budget_remaining",
    "get_compiled_graph",
    "get_workflow",
    "needs_clarification",
    "pending_interrupt",
    "resume_workflow",
    "route_after_evaluate",
    "route_after_validate",
    "run_workflow",
    "thread_config",
    "wants_targeted_clarification",
]

_log = get_logger(__name__)

#: The first fan-out. Order is declaration order, which is also the order
#: LangGraph applies their writes in - so a reducer merge is deterministic.
RETRIEVAL_BRANCHES: Final[tuple[str, ...]] = (
    "retrieve_programs",
    "validate_profile_evidence",
    "categorize_needs",
)

#: The second fan-out.
GENERATION_BRANCHES: Final[tuple[str, ...]] = (
    "generate_explanations",
    "generate_action_plan",
    "verify_citations",
)

#: Every node, by the name the graph knows it as. The single place a node name
#: is bound to an implementation.
NODE_FUNCTIONS: Final[dict[str, Any]] = {
    "extract_intake": extract_intake,
    "validate_profile": validate_profile,
    "request_clarification": request_clarification,
    "retrieve_programs": retrieve_programs,
    "validate_profile_evidence": validate_profile_evidence,
    "categorize_needs": categorize_needs,
    "join_candidates": join_candidates,
    "evaluate_eligibility": evaluate_eligibility,
    "request_targeted_clarification": request_targeted_clarification,
    "generate_explanations": generate_explanations,
    "generate_action_plan": generate_action_plan,
    "verify_citations": verify_citations,
    "merge_outputs": merge_outputs,
    "persist_results": persist_results,
}


# ---------------------------------------------------------------------------
# routing predicates
# ---------------------------------------------------------------------------
def clarification_budget_remaining(state: NavigatorState) -> bool:
    """Whether this run may still stop and ask a question."""
    return state.get("clarification_rounds", 0) < MAX_CLARIFICATION_ROUNDS


def needs_clarification(state: NavigatorState) -> bool:
    """Whether to pause before retrieval.

    ``validate_profile`` has already decided this and recorded it, because the
    decision needs the assembled profile and the list of questions it could
    actually ask. Re-deriving it here would be a second implementation of the
    same policy, which is how the two end up disagreeing.
    """
    return bool(state.get("needs_clarification")) and clarification_budget_remaining(state)


def wants_targeted_clarification(state: NavigatorState) -> bool:
    """Whether one more answer could move a program off ``MORE_INFO_NEEDED``.

    Set by ``evaluate_eligibility``, which only sets it when there is both a
    blocked program and a question the engine can honestly ask about it.
    """
    return bool(state.get("missing_critical_info")) and clarification_budget_remaining(state)


def route_after_validate(state: NavigatorState) -> str | list[str]:
    """``request_clarification``, or the three retrieval branches at once."""
    if needs_clarification(state):
        return "request_clarification"
    return list(RETRIEVAL_BRANCHES)


def route_after_evaluate(state: NavigatorState) -> str | list[str]:
    """``request_targeted_clarification``, or the three generation branches."""
    if wants_targeted_clarification(state):
        return "request_targeted_clarification"
    return list(GENERATION_BRANCHES)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def build_graph(deps: WorkflowDeps) -> StateGraph[Any, Any, Any, Any]:
    """Assemble the uncompiled graph with ``deps`` bound into every node.

    Nodes take ``(state, *, deps)`` and are bound with :func:`functools.partial`
    rather than reading a module-level singleton, which is what lets a test
    hand the whole graph a fixture-replaying model client and an in-memory
    catalog without patching anything.
    """
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(NavigatorState)
    for name, function in NODE_FUNCTIONS.items():
        graph.add_node(name, partial(function, deps=deps))

    graph.add_edge(START, "extract_intake")
    graph.add_edge("extract_intake", "validate_profile")

    graph.add_conditional_edges(
        "validate_profile",
        route_after_validate,
        ["request_clarification", *RETRIEVAL_BRANCHES],
    )
    # The answer re-enters validation rather than skipping ahead, so the
    # answers go through the same precedence and validation as everything else.
    graph.add_edge("request_clarification", "validate_profile")

    for branch in RETRIEVAL_BRANCHES:
        graph.add_edge(branch, "join_candidates")
    graph.add_edge("join_candidates", "evaluate_eligibility")

    graph.add_conditional_edges(
        "evaluate_eligibility",
        route_after_evaluate,
        ["request_targeted_clarification", *GENERATION_BRANCHES],
    )
    # Straight back to the engine: the targeted node has already re-applied the
    # answers to the profile, and only the engine may change a verdict.
    graph.add_edge("request_targeted_clarification", "evaluate_eligibility")

    for branch in GENERATION_BRANCHES:
        graph.add_edge(branch, "merge_outputs")
    graph.add_edge("merge_outputs", "persist_results")
    graph.add_edge("persist_results", END)
    return graph


@lru_cache(maxsize=4)
def get_compiled_graph(deps: WorkflowDeps, checkpointer: Any = None) -> Any:
    """A compiled graph, cached per ``(deps, checkpointer)`` pair.

    Compilation is pure work over the topology, so caching it is safe;
    ``WorkflowDeps`` is frozen and hashed by identity, which means a caller
    cannot mutate the dependencies of a graph that is already running.
    """
    compiled = build_graph(deps).compile(checkpointer, name="secondchance_navigator")
    _log.info(
        "workflow.graph.compiled",
        nodes=len(NODE_FUNCTIONS),
        checkpointer=type(checkpointer).__name__ if checkpointer is not None else "none",
    )
    return compiled


async def get_workflow(deps: WorkflowDeps, settings: Any = None) -> Any:
    """The compiled graph wired to the process-wide checkpointer.

    This is the one the API layer wants: it guarantees ``setup()`` has run and
    that paused runs survive a restart when ``CHECKPOINTER=postgres``.
    """
    handle = await get_checkpointer(settings or deps.settings)
    return get_compiled_graph(deps, handle.saver)


# ---------------------------------------------------------------------------
# running and resuming
# ---------------------------------------------------------------------------
def pending_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    """The clarification payload from an interrupted ``ainvoke``, if any.

    An interrupted run returns normally with an extra ``__interrupt__`` key
    holding ``Interrupt`` objects; it does not raise. The payload is whatever
    the node passed to ``interrupt()`` - see
    :func:`app.workflows.nodes.clarify.clarification_payload`.
    """
    pending = result.get("__interrupt__")
    if not pending:
        return None
    first = pending[0]
    value = getattr(first, "value", None)
    if not isinstance(value, dict):
        return None
    payload = dict(value)
    payload["interrupt_id"] = getattr(first, "id", None)
    return payload


@dataclass(frozen=True, slots=True)
class WorkflowOutcome:
    """What one ``ainvoke`` produced, in terms the API layer can act on.

    Deliberately not a ``WorkflowRunDetail``: that object is assembled from the
    database by the service layer, which also owns ``created_at``, ownership
    checks and pagination. This is the graph's half of the story.
    """

    run_id: uuid.UUID
    status: WorkflowStatus
    phase: WorkflowPhase
    state: NavigatorState
    interrupt: dict[str, Any] | None = None
    error: UserFacingError | None = None

    @property
    def is_awaiting_input(self) -> bool:
        return self.status is WorkflowStatus.AWAITING_INPUT

    @property
    def is_complete(self) -> bool:
        return self.status is WorkflowStatus.COMPLETED

    @property
    def matches(self) -> list[ProgramMatch]:
        return list(self.state.get("matches") or ())

    @property
    def plan(self) -> ActionPlan | None:
        return self.state.get("plan")

    @property
    def limitations(self) -> list[str]:
        return merge_unique((), self.state.get("limitations") or ())

    @property
    def model_usage(self) -> list[ModelUsage]:
        return list(self.state.get("model_usage") or ())

    @property
    def clarification_payload(self) -> dict[str, Any] | None:
        """What ``mark_awaiting_input`` should store for a poll to render."""
        return self.interrupt


#: The phase a paused run is in, by the kind of pause. Derived from the
#: interrupt payload rather than read from state, because ``interrupt()`` stops
#: the node *before* it returns: the phase it would have written is never
#: committed, so the checkpoint's last phase is the node before the pause. The
#: API layer wants the pause, so the pause is what this reports.
_PHASE_BY_INTERRUPT: Final[dict[str, WorkflowPhase]] = {
    "clarification": WorkflowPhase.AWAIT_CLARIFICATION,
    "targeted_clarification": WorkflowPhase.AWAIT_TARGETED_CLARIFICATION,
}


def _outcome(run_id: uuid.UUID, result: dict[str, Any]) -> WorkflowOutcome:
    state: NavigatorState = {key: value for key, value in result.items() if key != "__interrupt__"}  # type: ignore[assignment]
    interrupt = pending_interrupt(result)
    phase = state.get("phase", WorkflowPhase.PENDING)
    if interrupt is not None:
        return WorkflowOutcome(
            run_id=run_id,
            status=WorkflowStatus.AWAITING_INPUT,
            phase=_PHASE_BY_INTERRUPT.get(
                str(interrupt.get("kind")), WorkflowPhase.AWAIT_CLARIFICATION
            ),
            state=state,
            interrupt=interrupt,
        )
    return WorkflowOutcome(
        run_id=run_id,
        status=WorkflowStatus.COMPLETED,
        phase=WorkflowPhase.DONE if phase is WorkflowPhase.DONE else phase,
        state=state,
    )


def _failed(
    run_id: uuid.UUID, exc: BaseException, *, state: NavigatorState | None = None
) -> WorkflowOutcome:
    error = user_facing(exc)
    return WorkflowOutcome(
        run_id=run_id,
        status=WorkflowStatus.FAILED,
        phase=(state or {}).get("phase", WorkflowPhase.PENDING),
        state=state or {},
        error=error,
    )


async def _invoke(
    graph: Any,
    payload: Any,
    config: dict[str, Any],
    *,
    run_id: uuid.UUID,
    timeout_s: float | None,
) -> WorkflowOutcome:
    try:
        if timeout_s is None:
            result = await graph.ainvoke(payload, config=config)
        else:
            async with asyncio.timeout(timeout_s):
                result = await graph.ainvoke(payload, config=config)
    except TimeoutError as exc:
        # The checkpoint of the last completed superstep survives, so the run
        # is resumable with no answers re-entered.
        log_failure("workflow", exc, recoverable=False)
        return _failed(run_id, UpstreamTimeoutError(cause=exc))
    except asyncio.CancelledError:
        # Never swallowed: a cancelled run must stay cancelled, and its
        # checkpoint is already durable.
        raise
    except AppError as exc:
        log_failure("workflow", exc, recoverable=False)
        return _failed(run_id, exc)
    except Exception as exc:
        log_failure("workflow", exc, recoverable=False)
        return _failed(run_id, exc)
    return _outcome(run_id, dict(result))


async def run_workflow(
    state: NavigatorState,
    config: dict[str, Any] | None = None,
    *,
    graph: Any,
    timeout_s: float | None = None,
) -> WorkflowOutcome:
    """Start a run, returning when it finishes, pauses or fails.

    :param state: from :func:`~app.workflows.state.build_initial_state`.
    :param config: from :func:`~app.workflows.checkpointer.thread_config`;
        derived from ``state["run_id"]`` when omitted, which is the only
        correct value anyway.
    :param graph: a compiled graph, from :func:`get_workflow`.
    :param timeout_s: wall-clock budget, normally
        ``settings.workflow_timeout_s``. Exceeding it fails the run *and leaves
        it resumable*, because the last completed superstep is checkpointed.

    Never raises for a failure of the run itself: a failure is returned as
    ``status=FAILED`` with a user-safe ``error``, because the caller's next
    move is to record it and tell the user, not to handle an exception.
    """
    run_id = state["run_id"]
    return await _invoke(
        graph,
        state,
        config or thread_config(run_id),
        run_id=run_id,
        timeout_s=timeout_s,
    )


async def resume_workflow(
    run_id: uuid.UUID,
    answers: Sequence[ClarificationAnswer] = (),
    *,
    graph: Any,
    config: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> WorkflowOutcome:
    """Resume a paused run on its own thread with the user's answers.

    An empty ``answers`` list is valid and means "continue with what you have":
    the unanswered fields stay unknown and the programs they block stay at
    ``MORE_INFO_NEEDED``. That is a real choice, not a degraded one - a user is
    entitled to results without surrendering information they would rather not
    share.

    The state is not passed again. It comes from the checkpoint for
    ``thread_id == str(run_id)``, which is why a different process - or a
    different machine - can resume a run it did not start.
    """
    payload: Command[Any] = Command(
        resume={"answers": [answer.model_dump(mode="json") for answer in answers]}
    )
    return await _invoke(
        graph,
        payload,
        config or thread_config(run_id),
        run_id=run_id,
        timeout_s=timeout_s,
    )


async def continue_workflow(
    run_id: uuid.UUID,
    *,
    graph: Any,
    config: dict[str, Any] | None = None,
    timeout_s: float | None = None,
) -> WorkflowOutcome:
    """Pick up a run that stopped without asking anything - a crash or a cancel.

    Distinct from :func:`resume_workflow`, which answers an interrupt. This one
    replays the superstep that never committed, so a run killed mid-flight
    costs the work of one step rather than the whole run.
    """
    return await _invoke(
        graph,
        None,
        config or thread_config(run_id),
        run_id=run_id,
        timeout_s=timeout_s,
    )
