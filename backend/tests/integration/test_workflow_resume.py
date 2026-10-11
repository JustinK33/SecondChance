"""Interrupt and resume: the half of the workflow a user actually experiences.

A run that needs one more detail stops, the HTTP request that started it
returns, and the person answers minutes or days later - possibly against a
different process. That makes these tests the ones that decide whether the
product works at all, so they assert the properties that make a pause safe:

* the interrupt payload carries **deterministic** questions, built from author-
  written templates and a closed field set rather than from model output, so
  the same profile always produces the same questions in the same order;
* the run resumes on its own ``thread_id`` with ``Command(resume=...)``,
  completes, and the answers are visible in the resulting profile;
* the second round is **bounded**: a run can pause at most twice and then
  finishes with what it knows, leaving the blocked programs at
  ``MORE_INFO_NEEDED``;
* an empty answer list is a real answer - "continue with what you have";
* a saver that did not start the run can finish it, which is the whole reason
  the checkpoint is in Postgres rather than in memory.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import date
from typing import Any

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ai.fake import FakeGeminiClient
from app.ai.schemas import GeminiTask
from app.core.config import Settings
from app.db.repositories import SqlProgramRepository, WorkflowRunRepository
from app.schemas import (
    AllOf,
    AssistanceProgram,
    ClarificationAnswer,
    ClarificationInputKind,
    GeographyCondition,
    HouseholdSizeCondition,
    IncomeAbsoluteCondition,
    IncomePeriod,
    MatchStatus,
    ProgramCategory,
    RangeValue,
    ReviewStatus,
    RuleSet,
    UserProfile,
    WorkflowPhase,
    WorkflowStatus,
    clarification_question_id,
    utcnow,
)
from app.workflows.checkpointer import checkpointer_scope, thread_config
from app.workflows.deps import WorkflowDeps
from app.workflows.graph import (
    WorkflowOutcome,
    build_graph,
    build_initial_state,
    resume_workflow,
    run_workflow,
)
from app.workflows.questions import MATERIAL_FIELDS, intake_questions
from app.workflows.state import MAX_CLARIFICATION_ROUNDS
from tests.conftest import make_program

pytestmark = pytest.mark.db


def profile_of(outcome: WorkflowOutcome) -> UserProfile:
    """The profile a finished run assembled.

    ``NavigatorState["profile"]`` is optional because it is empty before
    extraction runs, so every read after a run has to say so once rather than
    at each use.
    """
    profile = outcome.state["profile"]
    assert profile is not None, "the run produced no profile"
    return profile


#: A narrative that says what is wrong and almost nothing else, which is the
#: common case. The ``sparse`` extraction fixture is the model's honest reading
#: of it: one need, and everything else unknown.
SPARSE_NARRATIVE = "Things are tight right now and I am having trouble keeping food in the house."

#: The monthly income limit on the fictional rule set below. Chosen so that a
#: run which never learns an income cannot resolve it either way.
INCOME_LIMIT_CENTS = 300_000


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Sessions on one rolled-back connection; see ``test_workflow_graph``."""
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


def _income_ruleset(slug: str) -> RuleSet:
    """A fictional rule set that genuinely needs an income to resolve.

    Three requirements, one of which the engine cannot settle without an income
    range. That is what makes the second clarification pass reachable: the
    engine can point at a specific program and say which answer would move it.
    """
    source_ids = [f"{slug}-source"]
    return RuleSet(
        version="2026.1",
        program_slug=slug,
        effective_from=date(2026, 1, 1),
        root=AllOf(
            description="All of these fictional requirements must be met.",
            source_ids=source_ids,
            conditions=[
                GeographyCondition(
                    description="You live in New Jersey (fictional requirement).",
                    source_ids=source_ids,
                    states=["NJ"],
                ),
                HouseholdSizeCondition(
                    description="Your household has at least one person.",
                    source_ids=source_ids,
                    min=1,
                ),
                IncomeAbsoluteCondition(
                    description="Your household brings in no more than $3,000 a month.",
                    source_ids=source_ids,
                    max_monthly_cents=INCOME_LIMIT_CENTS,
                ),
            ],
        ),
        source_ids=source_ids,
        review_status=ReviewStatus.APPROVED,
        limitations=["This rule set is fictional and encodes nothing real."],
        reviewed_by="test-fixture",
        reviewed_at=utcnow(),
    )


async def _seed(factory: async_sessionmaker[AsyncSession]) -> AssistanceProgram:
    program = make_program(
        "example-grocery-help",
        category=ProgramCategory.FOOD,
        keywords=("groceries", "food", "nutrition assistance"),
    )
    async with factory() as session:
        repository = SqlProgramRepository(session)
        await repository.upsert_program(program)
        await repository.upsert_rule_version(_income_ruleset(program.slug))
        await session.commit()
    return program


async def _create_run(factory: async_sessionmaker[AsyncSession], user_id: str) -> uuid.UUID:
    async with factory() as session:
        row, _ = await WorkflowRunRepository(session).create(user_id, narrative=SPARSE_NARRATIVE)
        run_id = row.id
        await session.commit()
    return run_id


@pytest.fixture
def settings_local() -> Settings:
    return Settings(_env_file=None, app_env="local", auth_mode="dev", checkpointer="memory")


def _sparse_deps(settings: Settings, factory: async_sessionmaker[AsyncSession]) -> WorkflowDeps:
    """Dependencies whose extraction reads almost nothing out of the narrative."""
    return WorkflowDeps(
        gemini=FakeGeminiClient(settings, cases={GeminiTask.EXTRACT_PROFILE: "sparse"}),
        settings=settings,
        session_factory=factory,
    )


def _answer(run_id: uuid.UUID, field: str, value: Any) -> ClarificationAnswer:
    """An answer keyed the way the client would key it."""
    return ClarificationAnswer(
        question_id=clarification_question_id(run_id, field), field=field, value=value
    )


class TestTheFirstPause:
    """Before retrieval: the fields almost every rule reads."""

    async def test_a_sparse_profile_interrupts_with_deterministic_questions(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _sparse_deps(settings_local, session_factory)
        graph = build_graph(deps).compile(InMemorySaver(), name="test_resume")

        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id,
                user_id=user_a,
                request_id="req-pause",
                narrative=SPARSE_NARRATIVE,
            ),
            graph=graph,
        )

        assert outcome.status is WorkflowStatus.AWAITING_INPUT
        assert outcome.phase is WorkflowPhase.AWAIT_CLARIFICATION
        payload = outcome.clarification_payload
        assert payload is not None
        assert payload["kind"] == "clarification"
        assert payload["run_id"] == str(run_id)
        assert payload["round"] == 1
        assert payload["fields"] == list(MATERIAL_FIELDS)

        # Deterministic: author-written copy, run-scoped ids, stable order.
        questions = payload["questions"]
        assert [question["id"] for question in questions] == [
            clarification_question_id(run_id, field) for field in MATERIAL_FIELDS
        ]
        for question in questions:
            assert question["why_it_matters"], "a question must say what it would change"
            assert question["is_required"] is False, "every question is skippable"
        assert {question["input_kind"] for question in questions} <= {
            kind.value for kind in ClarificationInputKind
        }

        # No results yet, and nothing persisted: a paused run has decided
        # nothing, so there is nothing to show or save.
        assert not outcome.matches
        assert outcome.plan is None

    async def test_the_questions_are_a_pure_function_of_the_profile(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """Called twice with the same inputs, it produces the same questions."""
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _sparse_deps(settings_local, session_factory)
        graph = build_graph(deps).compile(InMemorySaver(), name="test_resume_pure")
        state = build_initial_state(
            run_id=run_id, user_id=user_a, request_id="req-pure", narrative=SPARSE_NARRATIVE
        )
        first = await run_workflow(state, graph=graph)
        profile = profile_of(first)

        again = intake_questions(profile, list(MATERIAL_FIELDS), run_id=run_id)
        assert [question.id for question in again] == [
            question["id"] for question in (first.clarification_payload or {})["questions"]
        ]
        assert again == intake_questions(profile, list(MATERIAL_FIELDS), run_id=run_id)


class TestResuming:
    """Answering, on the same thread, with the state the checkpoint holds."""

    async def test_resuming_on_the_same_thread_completes_the_run(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        program = await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _sparse_deps(settings_local, session_factory)
        graph = build_graph(deps).compile(InMemorySaver(), name="test_resume_full")

        paused = await run_workflow(
            build_initial_state(
                run_id=run_id,
                user_id=user_a,
                request_id="req-resume",
                narrative=SPARSE_NARRATIVE,
            ),
            graph=graph,
        )
        assert paused.status is WorkflowStatus.AWAITING_INPUT

        resumed = await resume_workflow(
            run_id,
            [
                _answer(run_id, "state", "NJ"),
                _answer(run_id, "household_size", 3),
                _answer(run_id, "income", RangeValue(min=1200, max=1600)),
            ],
            graph=graph,
        )

        assert resumed.status is WorkflowStatus.COMPLETED
        assert resumed.phase is WorkflowPhase.DONE
        profile = profile_of(resumed)
        assert profile.state == "NJ"
        assert profile.household_size == 3
        assert profile.income_period is IncomePeriod.MONTHLY
        assert profile.income_monthly_range_cents() == (120_000, 160_000)
        assert resumed.state["clarification_rounds"] == 1

        # The income is comfortably under the limit, so every requirement
        # resolves and the engine says so.
        statuses = {match.program.slug: match.status for match in resumed.matches}
        assert statuses[program.slug] is MatchStatus.POTENTIAL_MATCH
        assert resumed.plan is not None

    async def test_a_skipped_answer_leaves_the_field_unknown_and_stops_the_asking(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """Skipping is a real answer: results, without surrendering the detail."""
        program = await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _sparse_deps(settings_local, session_factory)
        graph = build_graph(deps).compile(InMemorySaver(), name="test_resume_skip")

        await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-skip", narrative=SPARSE_NARRATIVE
            ),
            graph=graph,
        )
        resumed = await resume_workflow(
            run_id,
            [
                _answer(run_id, "state", "NJ"),
                _answer(run_id, "household_size", 2),
                ClarificationAnswer(
                    question_id=clarification_question_id(run_id, "income"),
                    field="income",
                    skipped=True,
                ),
            ],
            graph=graph,
        )

        assert resumed.status is WorkflowStatus.COMPLETED
        assert "income" in profile_of(resumed).unknown_fields
        # The income requirement cannot be resolved, so the program stays at
        # MORE_INFO_NEEDED - never at UNLIKELY_MATCH, because not knowing is
        # not the same as not qualifying.
        statuses = {match.program.slug: match.status for match in resumed.matches}
        assert statuses[program.slug] is MatchStatus.MORE_INFO_NEEDED
        # And the run did not stop again to ask the question it was told to
        # drop: one round spent, one round still in the budget.
        assert resumed.state["clarification_rounds"] == 1


class TestBoundedRounds:
    """Two pauses, never three."""

    async def test_the_second_round_is_reached_and_then_the_budget_stops_it(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """A partially answered first round opens the targeted second one.

        Round one is answered for the two cheap fields and the income question
        is left untouched - neither answered nor declined - which is what a
        half-filled form looks like. The engine then has a program it cannot
        resolve and a field that would resolve it, so the targeted pause opens.
        Resuming with no answers at all exhausts the budget and the run
        finishes with ``MORE_INFO_NEEDED`` rather than asking a third time.
        """
        program = await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _sparse_deps(settings_local, session_factory)
        graph = build_graph(deps).compile(InMemorySaver(), name="test_resume_bounded")

        first = await run_workflow(
            build_initial_state(
                run_id=run_id,
                user_id=user_a,
                request_id="req-bounded",
                narrative=SPARSE_NARRATIVE,
            ),
            graph=graph,
        )
        assert first.status is WorkflowStatus.AWAITING_INPUT
        assert (first.clarification_payload or {})["kind"] == "clarification"

        second = await resume_workflow(
            run_id,
            [_answer(run_id, "state", "NJ"), _answer(run_id, "household_size", 3)],
            graph=graph,
        )
        assert second.status is WorkflowStatus.AWAITING_INPUT
        payload = second.clarification_payload or {}
        assert payload["kind"] == "targeted_clarification"
        assert payload["round"] == 2
        # The targeted pass asks only about what is blocking a program.
        assert payload["fields"], "a targeted pause with no question is a bug"
        assert "income" in payload["fields"]

        # "Continue with what you have" - a valid, deliberate choice.
        third = await resume_workflow(run_id, [], graph=graph)
        assert third.status is WorkflowStatus.COMPLETED
        assert third.state["clarification_rounds"] == MAX_CLARIFICATION_ROUNDS
        statuses = {match.program.slug: match.status for match in third.matches}
        assert statuses[program.slug] is MatchStatus.MORE_INFO_NEEDED
        assert third.plan is not None, "a blocked program still deserves next steps"

    async def test_the_run_cannot_pause_a_third_time(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """Three pauses are unreachable by construction, not by luck."""
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        deps = _sparse_deps(settings_local, session_factory)
        graph = build_graph(deps).compile(InMemorySaver(), name="test_resume_third")

        pauses = 0
        outcome = await run_workflow(
            build_initial_state(
                run_id=run_id, user_id=user_a, request_id="req-third", narrative=SPARSE_NARRATIVE
            ),
            graph=graph,
        )
        while outcome.is_awaiting_input and pauses < 5:
            pauses += 1
            outcome = await resume_workflow(run_id, [], graph=graph)

        assert outcome.status is WorkflowStatus.COMPLETED
        assert pauses <= MAX_CLARIFICATION_ROUNDS
        assert outcome.state["clarification_rounds"] <= MAX_CLARIFICATION_ROUNDS


#: The tables ``AsyncPostgresSaver.setup()`` creates. Dropped again after the
#: Postgres tests below, because ``setup()`` is DDL committed outside this
#: test's rolled-back transaction: leaving them behind would mean the test
#: database no longer matches the state other suites find it in, and the
#: checkpointer creates them lazily anyway.
_CHECKPOINT_TABLES = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "checkpoint_migrations",
)


class TestPostgresCheckpoint:
    """The reason the checkpoint is not in memory."""

    @pytest_asyncio.fixture(autouse=True)
    async def _clean_checkpoint_tables(self, engine: AsyncEngine) -> AsyncIterator[None]:
        """Remove the checkpoint schema this class causes to be created."""
        try:
            yield
        finally:
            async with engine.begin() as connection:
                for table in _CHECKPOINT_TABLES:
                    await connection.execute(text(f"DROP TABLE IF EXISTS public.{table} CASCADE"))

    async def test_a_saver_that_did_not_start_the_run_can_finish_it(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
    ) -> None:
        """Pause with one checkpointer, resume with a different one.

        The second checkpointer has its own connection pool and its own
        compiled graph and shares no memory with the first, so the only way it
        can know what to resume is by reading the ``checkpoints`` tables -
        which is exactly what a second API worker does.

        The checkpoint rows are written with ``autocommit=True`` and so are
        real rows in the test database rather than part of this test's
        rolled-back transaction. They are deleted explicitly at the end, the
        same way the account-deletion flow deletes them.
        """
        settings_pg = Settings(
            _env_file=None,
            app_env="local",
            auth_mode="dev",
            checkpointer="postgres",
            database_url=settings_local.test_database_url,
        )
        await _seed(session_factory)
        run_id = await _create_run(session_factory, user_a)
        state = build_initial_state(
            run_id=run_id, user_id=user_a, request_id="req-pg", narrative=SPARSE_NARRATIVE
        )

        try:
            async with checkpointer_scope(settings_pg) as first_handle:
                first_deps = _sparse_deps(settings_pg, session_factory)
                paused = await run_workflow(
                    state,
                    thread_config(run_id),
                    graph=build_graph(first_deps).compile(first_handle.saver, name="worker_one"),
                )
                assert paused.status is WorkflowStatus.AWAITING_INPUT
                assert first_handle.is_persistent

            # Nothing from the first worker survives in this process except the
            # rows it committed.
            async with checkpointer_scope(settings_pg) as second_handle:
                snapshot = await second_handle.saver.aget_tuple(thread_config(run_id))
                assert snapshot is not None, "the pause must be durable"

                second_deps = _sparse_deps(settings_pg, session_factory)
                resumed = await resume_workflow(
                    run_id,
                    [
                        _answer(run_id, "state", "NJ"),
                        _answer(run_id, "household_size", 4),
                        _answer(run_id, "income", RangeValue(min=1000, max=1400)),
                    ],
                    graph=build_graph(second_deps).compile(second_handle.saver, name="worker_two"),
                )

            assert resumed.status is WorkflowStatus.COMPLETED
            assert profile_of(resumed).household_size == 4
            assert resumed.matches
        finally:
            async with checkpointer_scope(settings_pg) as cleanup:
                await cleanup.delete_threads([run_id])

    async def test_checkpoints_are_scoped_to_one_run(
        self,
        settings_local: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        user_a: str,
        user_b: str,
    ) -> None:
        """One run, one thread: another run's thread holds nothing of yours."""
        settings_pg = Settings(
            _env_file=None,
            app_env="local",
            auth_mode="dev",
            checkpointer="postgres",
            database_url=settings_local.test_database_url,
        )
        await _seed(session_factory)
        mine = await _create_run(session_factory, user_a)
        theirs = await _create_run(session_factory, user_b)

        try:
            async with checkpointer_scope(settings_pg) as handle:
                deps = _sparse_deps(settings_pg, session_factory)
                graph = build_graph(deps).compile(handle.saver, name="scoped")
                await run_workflow(
                    build_initial_state(
                        run_id=mine,
                        user_id=user_a,
                        request_id="req-mine",
                        narrative=SPARSE_NARRATIVE,
                    ),
                    thread_config(mine),
                    graph=graph,
                )
                assert await handle.saver.aget_tuple(thread_config(mine)) is not None
                assert await handle.saver.aget_tuple(thread_config(theirs)) is None
        finally:
            async with checkpointer_scope(settings_pg) as cleanup:
                await cleanup.delete_threads([mine, theirs])
