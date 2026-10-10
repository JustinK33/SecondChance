"""The LangGraph workflow, end to end, against Postgres and a fake model.

These tests exist to prove the graph is a graph and not a sequence of function
calls with extra ceremony. Four properties are asserted directly rather than
assumed:

1. **A full run produces real results.** Seeded catalog, fixture-replaying
   model client, and the matches and the action plan come back out of the
   database afterwards.
2. **The fan-outs really fan out.** Each branch of a fan-out waits on an
   ``asyncio.Barrier`` shared with its siblings before doing any work, so a
   graph that ran them sequentially would deadlock and the test would time out
   rather than quietly pass.
3. **Parallel branches write disjoint channels.** Every write a real run makes
   is recorded and checked against
   :data:`app.workflows.state.FIELD_WRITERS`, so the "one writer per field"
   comments in that module are enforced instead of merely written down.
4. **The model cannot move a verdict.** ``evaluate_eligibility`` is asserted to
   make zero model calls, and an explanation that claims an eligibility
   outcome is rejected by the integrity gate while the engine's status stands.

Everything seeded here is visibly fictional - ``example.org`` URLs and names
that say so - because the product rule is that a program is never fabricated
and nothing in ``backend/tests`` may be mistaken for the curated catalog.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ai.fake import FakeGeminiClient
from app.ai.schemas import GeminiTask
from app.core.config import Settings
from app.db.repositories import (
    MatchRepository,
    PlanRepository,
    SqlProgramRepository,
    WorkflowRunRepository,
)
from app.schemas import (
    AssistanceProgram,
    EligibilityEvaluation,
    GeminiResult,
    GeoScope,
    MatchExplanation,
    MatchStatus,
    ModelUsage,
    ProgramCategory,
    UserProfile,
    WorkflowPhase,
    WorkflowStatus,
)
from app.workflows import graph as graph_module
from app.workflows.deps import WorkflowDeps
from app.workflows.graph import (
    GENERATION_BRANCHES,
    RETRIEVAL_BRANCHES,
    build_graph,
    build_initial_state,
    run_workflow,
)
from app.workflows.state import FIELD_WRITERS, PARALLEL_GROUPS, REDUCER_FIELDS
from tests.conftest import make_program, make_ruleset

pytestmark = pytest.mark.db

#: A narrative in the voice of the person this product is for. Personal content
#: in production, fictional here.
NARRATIVE = (
    "I live in New Brunswick, New Jersey with my two kids and I take classes full time "
    "at Rutgers. Groceries are hard to afford this month and my electric bill is overdue."
)

#: How long a fan-out barrier waits before declaring the branches sequential.
BARRIER_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NodeRun:
    """One node execution, as observed from outside the node."""

    node: str
    entered_at: float
    started_at: float
    finished_at: float
    wrote: frozenset[str]
    model_calls: int
    saw: frozenset[str]


def _instrument(
    monkeypatch: pytest.MonkeyPatch,
    runs: list[NodeRun],
    *,
    barriers: dict[str, asyncio.Barrier] | None = None,
) -> None:
    """Record every node execution, optionally gating the fan-out branches.

    The gate is the real assertion about concurrency: a branch waits for its
    two siblings to arrive before it does anything, so a sequential scheduler
    cannot get past the first one.
    """
    gates = barriers or {}

    def wrap(name: str, function: Callable[..., Awaitable[dict[str, Any]]]):
        async def wrapped(state: dict[str, Any], *, deps: WorkflowDeps) -> dict[str, Any]:
            entered = time.perf_counter()
            gate = gates.get(name)
            if gate is not None:
                async with asyncio.timeout(BARRIER_TIMEOUT_S):
                    await gate.wait()
            before = len(getattr(deps.gemini, "calls", ()))
            started = time.perf_counter()
            update = await function(state, deps=deps)
            runs.append(
                NodeRun(
                    node=name,
                    entered_at=entered,
                    started_at=started,
                    finished_at=time.perf_counter(),
                    wrote=frozenset(update),
                    model_calls=len(getattr(deps.gemini, "calls", ())) - before,
                    saw=frozenset(key for key, value in state.items() if value not in (None, [])),
                )
            )
            return update

        return wrapped

    for name, function in list(graph_module.NODE_FUNCTIONS.items()):
        monkeypatch.setitem(graph_module.NODE_FUNCTIONS, name, wrap(name, function))


def _order(runs: Sequence[NodeRun]) -> list[str]:
    return [run.node for run in runs]


def _run_named(runs: Sequence[NodeRun], node: str) -> NodeRun:
    found = [run for run in runs if run.node == node]
    assert found, f"{node} never ran; order was {_order(runs)}"
    return found[-1]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A factory whose sessions all live on one rolled-back connection.

    The graph opens a session per task, so it needs a factory rather than the
    shared ``session`` fixture. Binding every session to the same connection
    inside one transaction keeps the production code path intact - nodes really
    commit - while nothing survives the test.
    """
    async with engine.connect() as connection:
        transaction = await connection.begin()
        factory = async_sessionmaker(
            bind=connection,
            expire_on_commit=False,
            autoflush=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            yield factory
        finally:
            await transaction.rollback()


def _program(
    slug: str,
    *,
    category: ProgramCategory,
    keywords: Sequence[str],
    counties: Sequence[str] = ("Middlesex",),
) -> AssistanceProgram:
    base = make_program(slug, category=category, keywords=keywords, counties=counties)
    return base.model_copy(
        update={
            "geo_scope": GeoScope(
                nationwide=False,
                states=["NJ"],
                counties=list(counties),
                label="Fictional scope used in tests",
            )
        }
    )


CATALOG: tuple[tuple[str, ProgramCategory, tuple[str, ...]], ...] = (
    ("example-grocery-help", ProgramCategory.FOOD, ("groceries", "nutrition assistance", "food")),
    ("example-heat-help", ProgramCategory.UTILITIES, ("electric", "utility bill", "heat")),
    ("example-campus-aid", ProgramCategory.UNIVERSITY_EMERGENCY_AID, ("student", "emergency")),
)


async def _seed(factory: async_sessionmaker[AsyncSession]) -> list[AssistanceProgram]:
    """Insert the fictional catalog and its approved rule sets."""
    programs = [
        _program(slug, category=category, keywords=keywords) for slug, category, keywords in CATALOG
    ]
    async with factory() as session:
        repository = SqlProgramRepository(session)
        for program in programs:
            await repository.upsert_program(program)
            await repository.upsert_rule_version(make_ruleset(program.slug))
        await session.commit()
    return programs


async def _create_run(factory: async_sessionmaker[AsyncSession], user_id: str) -> uuid.UUID:
    """A queued run row, because the results point at it with a foreign key."""
    async with factory() as session:
        row, _ = await WorkflowRunRepository(session).create(user_id, narrative=NARRATIVE)
        run_id = row.id
        await session.commit()
    return run_id


@pytest.fixture
def settings_local() -> Settings:
    return Settings(_env_file=None, app_env="local", auth_mode="dev", checkpointer="memory")


def _deps(
    settings: Settings,
    factory: async_sessionmaker[AsyncSession],
    *,
    gemini: Any = None,
) -> WorkflowDeps:
    return WorkflowDeps(
        gemini=gemini if gemini is not None else FakeGeminiClient(settings),
        settings=settings,
        session_factory=factory,
    )


def _compile(deps: WorkflowDeps) -> Any:
    """Compile a graph of our own, so instrumentation is never cached away."""
    return build_graph(deps).compile(InMemorySaver(), name="test_navigator")


# ---------------------------------------------------------------------------
# 1. a full run
# ---------------------------------------------------------------------------
class TestFullRun:
    """One narrative in, matches and a saved plan out."""

    async def test_a_full_run_produces_matches_and_a_persisted_plan(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _deps(settings_local, session_factory)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-full", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )

        assert outcome.status is WorkflowStatus.COMPLETED
        assert outcome.phase is WorkflowPhase.DONE
        assert outcome.error is None
        assert outcome.matches, "a seeded, approved catalog must produce matches"
        assert outcome.plan is not None
        assert outcome.plan.items

        # The status on every match is the engine's, by construction and by the
        # ProgramMatch validator. Asserted anyway: it is the product's central
        # promise and deserves an explicit line in the suite.
        for match in outcome.matches:
            assert match.status is match.evaluation.status
            assert match.status in set(MatchStatus)

        async with session_factory() as session:
            stored = await MatchRepository(session).list_for_run(user_a, run_id)
            plan = await PlanRepository(session).get_latest_for_run(user_a, run_id)
        assert {match.program.slug for match in stored} == {
            match.program.slug for match in outcome.matches
        }
        assert plan is not None
        assert len(plan.items) == len(outcome.plan.items)
        assert outcome.state["persisted"]["matches"] == len(outcome.matches)

    async def test_the_run_records_its_model_usage_and_asks_for_nothing_sensitive(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _deps(settings_local, session_factory)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-usage", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )

        assert outcome.model_usage, "every model call must record what it cost"
        assert all(isinstance(usage, ModelUsage) for usage in outcome.model_usage)
        async with session_factory() as session:
            row = await WorkflowRunRepository(session).get(user_a, run_id)
        assert row is not None
        assert len(row.model_usage) == len(outcome.model_usage)

        # No question this product can ask has anywhere to put an SSN or a bank
        # detail, because a question's field is validated against a closed set.
        asked = " ".join(
            question.question for question in outcome.state.get("clarification_questions", [])
        ).casefold()
        for term in ("social security", "ssn", "bank account", "routing number"):
            assert term not in asked


# ---------------------------------------------------------------------------
# 2 & 3. the fan-outs
# ---------------------------------------------------------------------------
class TestFanOut:
    """The two parallel groups: really parallel, and really disjoint."""

    async def test_the_retrieval_branches_run_concurrently_and_converge(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each branch waits for the other two before working.

        If the graph ran them one after another, the first branch would wait
        for siblings that cannot start, the barrier would time out and this
        test would fail. Passing it is a proof of concurrency rather than an
        observation about timings, which on a loaded machine prove nothing.
        """
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        runs: list[NodeRun] = []
        barrier = asyncio.Barrier(len(RETRIEVAL_BRANCHES))
        _instrument(monkeypatch, runs, barriers=dict.fromkeys(RETRIEVAL_BRANCHES, barrier))

        deps = _deps(settings_local, session_factory)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-fanout", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )
        assert outcome.status is WorkflowStatus.COMPLETED

        order = _order(runs)
        assert order[:2] == ["extract_intake", "validate_profile"]
        branch_runs = [_run_named(runs, name) for name in RETRIEVAL_BRANCHES]
        join = _run_named(runs, "join_candidates")

        # The barrier not timing out is the proof; this records the shape of it.
        # Every branch entered before any of them was allowed to finish, which
        # is only possible if all three were in flight at once.
        assert max(run.entered_at for run in branch_runs) <= min(
            run.finished_at for run in branch_runs
        )

        # The join is a real barrier: it ran once, after all three committed.
        assert order.count("join_candidates") == 1
        assert join.started_at >= max(run.finished_at for run in branch_runs)
        assert outcome.state["branch_contributions"] == {
            "retrieve_programs": len(outcome.state["candidates"]),
            "validate_profile_evidence": len(outcome.state["evidence_warnings"]),
            "categorize_needs": len(outcome.state["need_categories"]),
        }

        # A branch cannot see a sibling's writes, which is what makes them one
        # superstep rather than three.
        assert "candidates" not in _run_named(runs, "categorize_needs").saw
        assert "need_categories" not in _run_named(runs, "retrieve_programs").saw

    async def test_the_generation_branches_run_concurrently_and_converge(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        runs: list[NodeRun] = []
        barrier = asyncio.Barrier(len(GENERATION_BRANCHES))
        _instrument(monkeypatch, runs, barriers=dict.fromkeys(GENERATION_BRANCHES, barrier))

        deps = _deps(settings_local, session_factory)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-gen", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )
        assert outcome.status is WorkflowStatus.COMPLETED

        order = _order(runs)
        merge = _run_named(runs, "merge_outputs")
        branch_runs = [_run_named(runs, name) for name in GENERATION_BRANCHES]
        assert max(run.entered_at for run in branch_runs) <= min(
            run.finished_at for run in branch_runs
        )
        assert order.count("merge_outputs") == 1
        assert merge.started_at >= max(run.finished_at for run in branch_runs)
        assert order[-1] == "persist_results"

    async def test_no_two_parallel_nodes_write_the_same_channel(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The single-writer rule, enforced against a real run's writes.

        LangGraph raises ``InvalidUpdateError`` when two tasks in one superstep
        write a channel with no reducer, so a violation here is a graph that
        dies mid-fan-out in production. Checking the writes against
        ``FIELD_WRITERS`` also catches the reverse mistake: a node quietly
        writing a channel whose documented owner is somebody else.
        """
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        runs: list[NodeRun] = []
        _instrument(monkeypatch, runs)

        deps = _deps(settings_local, session_factory)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-writers", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )
        assert outcome.status is WorkflowStatus.COMPLETED

        for group in PARALLEL_GROUPS:
            present = [name for name in group if any(run.node == name for run in runs)]
            for first in present:
                for second in present:
                    if first >= second:
                        continue
                    shared = (
                        _run_named(runs, first).wrote & _run_named(runs, second).wrote
                    ) - REDUCER_FIELDS
                    assert not shared, (
                        f"{first} and {second} run concurrently and both wrote {sorted(shared)}; "
                        "a channel with two concurrent writers needs a reducer"
                    )

        for run in runs:
            undeclared = run.wrote - REDUCER_FIELDS - FIELD_WRITERS[run.node]
            assert not undeclared, (
                f"{run.node} wrote {sorted(undeclared)}, which FIELD_WRITERS does not "
                "declare it owns"
            )

    async def test_every_declared_writer_is_a_real_node(self) -> None:
        """``FIELD_WRITERS`` and the graph cannot drift apart."""
        assert set(FIELD_WRITERS) == set(graph_module.NODE_FUNCTIONS)
        for group in PARALLEL_GROUPS:
            for name in group:
                assert name in graph_module.NODE_FUNCTIONS


# ---------------------------------------------------------------------------
# 4. the model cannot decide anything
# ---------------------------------------------------------------------------
class _StatusClaimingGemini(FakeGeminiClient):
    """A model client that tries to award eligibility in its prose.

    Worse than the ``injected_instructions`` fixture, which the AI layer
    already refuses: this one bypasses that refusal and hands the graph a
    validated :class:`MatchExplanation` whose every line claims an outcome. It
    exists to test the integrity gate in ``merge_outputs`` rather than the
    check in ``app.ai``, because defence in depth is only real if each layer is
    tested without the others.
    """

    async def explain_match(  # type: ignore[override]
        self,
        profile: UserProfile,
        program: AssistanceProgram,
        evaluation: EligibilityEvaluation,
        *,
        request_id: str | None = None,
    ) -> GeminiResult[MatchExplanation]:
        self.calls.append(GeminiTask.EXPLAIN_MATCH)
        return GeminiResult(
            value=MatchExplanation(
                headline="You are eligible for this program and have been approved.",
                why_it_may_fit=["The program data confirmed that you qualify."],
                what_we_still_need=[],
                caveats=[],
                generated_by="gemini",
            ),
            usage=ModelUsage(model="fixture", source="live"),
        )


class TestTheEngineDecides:
    """``evaluate_eligibility`` is the only node that decides anything."""

    async def test_evaluate_eligibility_makes_no_model_calls(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        runs: list[NodeRun] = []
        _instrument(monkeypatch, runs)

        deps = _deps(settings_local, session_factory)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-engine", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )
        assert outcome.status is WorkflowStatus.COMPLETED

        for run in runs:
            if run.node in {"evaluate_eligibility", "validate_profile", "join_candidates"}:
                assert run.model_calls == 0, f"{run.node} made {run.model_calls} model calls"
        assert any(run.model_calls > 0 for run in runs), "the fake client was never exercised"

    async def test_a_status_claiming_explanation_cannot_change_the_verdict(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """Prose that awards eligibility is removed; the engine's status stands."""
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _deps(settings_local, session_factory, gemini=_StatusClaimingGemini(settings_local))
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-claim", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )

        assert outcome.status is WorkflowStatus.COMPLETED
        assert outcome.matches
        # Every explanation offered claimed an outcome, so every one was
        # rejected whole rather than patched into something plausible.
        validation = outcome.state["validation"]
        assert validation["explanations_offered"] > 0
        assert validation["explanations_kept"] == 0
        assert validation["explanations_rejected"]
        assert validation["status_source"] == "rule_engine"

        for match in outcome.matches:
            assert match.explanation is None
            assert match.status is match.evaluation.status
        # A MatchExplanation has no status field at all, so there is nowhere
        # for a model to put one even if the prose survived.
        assert "status" not in MatchExplanation.model_fields

        rendered = " ".join(
            [
                outcome.plan.title if outcome.plan else "",
                outcome.plan.summary if outcome.plan else "",
                *[item.title for item in (outcome.plan.items if outcome.plan else [])],
                *[item.detail for item in (outcome.plan.items if outcome.plan else [])],
            ]
        ).casefold()
        for banned in ("eligible", "ineligible", "you qualify", "guaranteed", "approved"):
            assert banned not in rendered

    async def test_the_fixture_model_attempting_an_outcome_is_refused_upstream(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """The AI layer refuses the same attempt before the graph sees it.

        The ``injected_instructions`` fixture is a compliant model doing what a
        poisoned program record told it to. The run still completes, with the
        engine's verdicts and a recorded limitation instead of prose.
        """
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        gemini = FakeGeminiClient(
            settings_local, cases={GeminiTask.EXPLAIN_MATCH: "injected_instructions"}
        )
        deps = _deps(settings_local, session_factory, gemini=gemini)
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-injected", narrative=NARRATIVE
            ),
            graph=_compile(deps),
        )

        assert outcome.status is WorkflowStatus.COMPLETED
        assert outcome.matches
        assert all(match.explanation is None for match in outcome.matches)
        assert any("plain-language" in limitation for limitation in outcome.limitations)
        assert outcome.state["errors"], "the refusal must be recorded, not swallowed"


# ---------------------------------------------------------------------------
# routing predicates, exercised directly
# ---------------------------------------------------------------------------
class TestRoutingPredicates:
    """Named predicates, so the topology's decisions are unit-testable."""

    def test_route_after_validate(self) -> None:
        assert (
            graph_module.route_after_validate(
                {"needs_clarification": True, "clarification_rounds": 0}
            )
            == "request_clarification"
        )
        assert graph_module.route_after_validate({"needs_clarification": False}) == list(
            RETRIEVAL_BRANCHES
        )
        # The budget wins over the flag: a run that has already asked twice
        # proceeds with what it knows.
        assert graph_module.route_after_validate(
            {"needs_clarification": True, "clarification_rounds": 2}
        ) == list(RETRIEVAL_BRANCHES)

    def test_route_after_evaluate(self) -> None:
        assert (
            graph_module.route_after_evaluate(
                {"missing_critical_info": True, "clarification_rounds": 1}
            )
            == "request_targeted_clarification"
        )
        assert graph_module.route_after_evaluate(
            {"missing_critical_info": True, "clarification_rounds": 2}
        ) == list(GENERATION_BRANCHES)
        assert graph_module.route_after_evaluate({}) == list(GENERATION_BRANCHES)

    def test_clarification_budget(self) -> None:
        assert graph_module.clarification_budget_remaining({})
        assert graph_module.clarification_budget_remaining({"clarification_rounds": 1})
        assert not graph_module.clarification_budget_remaining({"clarification_rounds": 2})
        assert not graph_module.clarification_budget_remaining({"clarification_rounds": 3})
