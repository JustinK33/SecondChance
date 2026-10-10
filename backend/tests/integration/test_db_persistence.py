"""Integration tests for the database layer: constraints, cascades and search.

The point of these tests is not that the repositories can write a row. It is
that the guarantees the rest of the system is allowed to assume are enforced by
*Postgres*, so they hold against a buggy service, a replayed workflow step, a
hand-written ``UPDATE`` or a future refactor:

* a repeated idempotency key cannot create a second run;
* a replayed evaluation cannot create a second match for the same program;
* two programs asking for the same document cannot create two action items;
* a program cannot have two open rule versions, so "what are its rules today"
  always has exactly one answer;
* deleting a run really removes its matches, plans and items;
* a retired program cannot be deleted out from under a saved result;
* the full-text vector is populated from the keywords and the GIN index on it
  is usable.

Every constraint test asserts on the real ``IntegrityError`` from a real
``INSERT``. Each one runs inside ``session.begin_nested()`` so that the failed
statement is confined to its own savepoint and the test can carry on afterwards.
"""

from __future__ import annotations

import re
import uuid
from datetime import date, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.core.config import Settings
from app.core.errors import ErrorCode
from app.core.logging import hash_user_id
from app.db import models
from app.db.repositories import (
    AccountDataRepository,
    AuditAction,
    AuditRepository,
    MatchRepository,
    PlanRepository,
    ProfileRepository,
    SqlProgramRepository,
    WorkflowRunRepository,
)
from app.db.session import create_session_factory, resolve_connect_args, session_scope
from app.schemas import (
    ActionItemKind,
    HouseholdComposition,
    MatchStatus,
    ProfileFlag,
    ProgramCategory,
    ReviewStatus,
    UserFacingError,
    VerificationStatus,
    WorkflowPhase,
    WorkflowStatus,
    make_dedup_key,
    utcnow,
)
from tests.conftest import make_match, make_plan, make_profile, make_program, make_ruleset

pytestmark = pytest.mark.db

NARRATIVE = "I am a fictional test user who needs help buying groceries this month."


async def _seed_program(
    session: AsyncSession,
    slug: str = "example-grocery-help",
    *,
    with_rules: bool = True,
) -> models.AssistanceProgram:
    """Insert a fictional program (and optionally its approved rule set)."""
    repo = SqlProgramRepository(session)
    program = make_program(slug)
    program_id = await repo.upsert_program(program)
    if with_rules:
        await repo.upsert_rule_version(make_ruleset(slug))
        await repo.set_active_rule_version(program_id, "2026.1")
    return await session.get_one(models.AssistanceProgram, program_id)


async def _seed_run(
    session: AsyncSession, user_id: str, *, idempotency_key: str | None = None
) -> models.WorkflowRun:
    runs = WorkflowRunRepository(session)
    run, created = await runs.create(
        user_id,
        narrative=NARRATIVE,
        idempotency_key=idempotency_key,
        status=WorkflowStatus.RUNNING,
        phase=WorkflowPhase.RETRIEVE_PROGRAMS,
    )
    assert created
    return run


class TestIdempotency:
    async def test_repeated_idempotency_key_returns_the_original_run(
        self, session: AsyncSession, user_a: str
    ) -> None:
        runs = WorkflowRunRepository(session)
        first, created_first = await runs.create(
            user_a, narrative=NARRATIVE, idempotency_key="submit-button-0001"
        )
        second, created_second = await runs.create(
            user_a, narrative=NARRATIVE, idempotency_key="submit-button-0001"
        )

        assert created_first is True
        assert created_second is False
        assert second.id == first.id

        total = await session.scalar(
            select(func.count())
            .select_from(models.WorkflowRun)
            .where(models.WorkflowRun.user_id == user_a)
        )
        assert total == 1

    async def test_duplicate_idempotency_key_is_rejected_by_the_database(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """The repository resolves the conflict; the constraint is what makes it safe."""
        await _seed_run(session, user_a, idempotency_key="duplicate-key-01")

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.WorkflowRun(
                        user_id=user_a,
                        status=WorkflowStatus.QUEUED.value,
                        phase=WorkflowPhase.PENDING.value,
                        idempotency_key="duplicate-key-01",
                        thread_id=str(uuid.uuid4()),
                        narrative_hash="0" * 64,
                    )
                )
                await session.flush()
        assert "uq_workflow_runs_user_id_idempotency_key" in str(caught.value)

    async def test_two_users_may_reuse_the_same_idempotency_key(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        """The key is scoped per user: it is a client's retry token, not a global id."""
        await _seed_run(session, user_a, idempotency_key="shared-key-0001")
        await _seed_run(session, user_b, idempotency_key="shared-key-0001")

        total = await session.scalar(select(func.count()).select_from(models.WorkflowRun))
        assert total == 2

    async def test_a_null_idempotency_key_does_not_collide(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """Postgres treats NULLs as distinct in a UNIQUE constraint.

        That is the behaviour we want and the reason there is no partial index:
        the key is optional, and two runs without one are two real runs.
        """
        await _seed_run(session, user_a)
        await _seed_run(session, user_a)
        total = await session.scalar(
            select(func.count())
            .select_from(models.WorkflowRun)
            .where(models.WorkflowRun.user_id == user_a)
        )
        assert total == 2

    async def test_the_narrative_itself_is_never_stored(
        self, session: AsyncSession, user_a: str
    ) -> None:
        run = await _seed_run(session, user_a)
        assert len(run.narrative_hash) == 64
        assert NARRATIVE not in run.narrative_hash
        assert "narrative" not in {c.name for c in models.WorkflowRun.__table__.columns}


class TestMatchConstraints:
    async def test_duplicate_run_and_program_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        match = make_match(make_program(program.slug))
        await MatchRepository(session).upsert_many(user_a, run.id, [match])

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ProgramMatch(
                        user_id=user_a,
                        run_id=run.id,
                        program_id=program.id,
                        rule_version="2026.1",
                        status=MatchStatus.POTENTIAL_MATCH.value,
                        outcome="TRUE",
                        score=0.5,
                        evaluated_at=utcnow(),
                    )
                )
                await session.flush()
        assert "uq_program_matches_run_id_program_id" in str(caught.value)

    async def test_reevaluating_updates_in_place_instead_of_duplicating(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        matches = MatchRepository(session)
        schema_program = make_program(program.slug)

        await matches.upsert_many(
            user_a, run.id, [make_match(schema_program, status=MatchStatus.MORE_INFO_NEEDED)]
        )
        await matches.upsert_many(
            user_a,
            run.id,
            [make_match(schema_program, status=MatchStatus.POTENTIAL_MATCH, score=0.9)],
        )

        stored = await matches.list_for_run(user_a, run.id)
        assert len(stored) == 1
        assert stored[0].status is MatchStatus.POTENTIAL_MATCH
        assert stored[0].score == pytest.approx(0.9)

    async def test_more_info_needed_must_name_a_missing_field(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """A "we need more information" verdict that cannot say *what* is useless."""
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ProgramMatch(
                        user_id=user_a,
                        run_id=run.id,
                        program_id=program.id,
                        rule_version="2026.1",
                        status=MatchStatus.MORE_INFO_NEEDED.value,
                        outcome="UNKNOWN",
                        missing_fields=[],
                        evaluated_at=utcnow(),
                    )
                )
                await session.flush()
        assert "more_info_needed_names_a_missing_field" in str(caught.value)

    async def test_score_outside_the_unit_interval_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ProgramMatch(
                        user_id=user_a,
                        run_id=run.id,
                        program_id=program.id,
                        rule_version="2026.1",
                        status=MatchStatus.POTENTIAL_MATCH.value,
                        outcome="TRUE",
                        score=1.5,
                        evaluated_at=utcnow(),
                    )
                )
                await session.flush()
        assert "score_in_unit_interval" in str(caught.value)

    async def test_reading_a_match_back_revalidates_the_engine_invariant(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """A status inconsistent with its outcome must not survive a read.

        ``EligibilityEvaluation`` asserts ``status == STATUS_BY_OUTCOME[outcome]``,
        so a row corrupted by a manual update fails loudly in the repository
        rather than being rendered to a person as an assessment.
        """
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        matches = MatchRepository(session)
        await matches.upsert_many(user_a, run.id, [make_match(make_program(program.slug))])

        await session.execute(
            text(
                "UPDATE program_matches SET outcome = 'FALSE' "
                "WHERE run_id = :run_id AND program_id = :program_id"
            ).bindparams(run_id=run.id, program_id=program.id)
        )

        with pytest.raises(ValueError, match="contradicts outcome"):
            await matches.list_for_run(user_a, run.id)


class TestPlanConstraints:
    async def test_duplicate_plan_and_dedup_key_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        plan = make_plan(run.id, make_program(program.slug))
        saved = await PlanRepository(session).save(user_a, plan)

        existing = plan.items[0]
        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ActionItem(
                        id=uuid.uuid4(),
                        plan_id=saved.id,
                        user_id=user_a,
                        program_id=program.id,
                        program_name=program.official_name,
                        kind=ActionItemKind.DOCUMENT.value,
                        title="A second item for the same requirement",
                        detail="This must not be storable.",
                        sort_order=99,
                        dedup_key=existing.dedup_key,
                    )
                )
                await session.flush()
        assert "uq_action_items_plan_id_dedup_key" in str(caught.value)

    async def test_two_programs_asking_for_the_same_document_collapse(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """ "Utility bill" and "proof of residency" are one requirement.

        The dedup key normalises wording through ``DOCUMENT_SUBJECT_ALIASES``
        before slugifying, so the collapse does not depend on two program
        authors - or two model outputs - choosing the same phrase. Because the
        item id is derived from that key, a plan carrying both phrasings is
        rejected before it ever reaches the database.
        """
        assert make_dedup_key(ActionItemKind.DOCUMENT, "utility bill") == make_dedup_key(
            ActionItemKind.DOCUMENT, "proof of residency"
        )
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        with pytest.raises(ValidationError, match="action item ids must be unique"):
            make_plan(
                run.id,
                make_program(program.slug),
                subjects=("utility bill", "proof of residency"),
            )

    async def test_regenerating_a_plan_versions_rather_than_overwrites(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        plans = PlanRepository(session)
        schema_program = make_program(program.slug)

        assert await plans.next_version(user_a, run.id) == 1
        await plans.save(user_a, make_plan(run.id, schema_program, version=1))
        assert await plans.next_version(user_a, run.id) == 2
        await plans.save(user_a, make_plan(run.id, schema_program, version=2))

        latest = await plans.get_latest_for_run(user_a, run.id)
        assert latest is not None
        assert latest.version == 2
        total = await session.scalar(
            select(func.count())
            .select_from(models.ActionPlan)
            .where(models.ActionPlan.run_id == run.id)
        )
        assert total == 2

        # Both plan rows survive, but the items moved to the current version
        # rather than being duplicated - an item id is derived from the run, so
        # it can only belong to one plan at a time. See the versioning note in
        # app.db.repositories.plans.
        assert len(latest.items) == 2
        assert await session.scalar(select(func.count()).select_from(models.ActionItem)) == 2

    async def test_regeneration_preserves_completion_state(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """Re-running must never un-tick something the user already did."""
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        plans = PlanRepository(session)
        schema_program = make_program(program.slug)
        plan = make_plan(run.id, schema_program, version=1)
        await plans.save(user_a, plan)
        await plans.set_item_complete(user_a, plan.items[0].id, is_complete=True)

        await plans.save(user_a, make_plan(run.id, schema_program, version=2))

        item = await plans.get_item(user_a, plan.items[0].id)
        assert item is not None
        assert item.is_complete is True
        assert item.completed_at is not None

    async def test_duplicate_run_and_version_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        await PlanRepository(session).save(user_a, make_plan(run.id, make_program(program.slug)))

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ActionPlan(
                        id=uuid.uuid4(),
                        user_id=user_a,
                        run_id=run.id,
                        version=1,
                        title="A second version 1",
                        summary="This must not be storable.",
                    )
                )
                await session.flush()
        assert "uq_action_plans_run_id_version" in str(caught.value)

    async def test_completion_flag_and_timestamp_must_agree(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        saved = await PlanRepository(session).save(
            user_a, make_plan(run.id, make_program(program.slug))
        )

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                await session.execute(
                    text(
                        "UPDATE action_items SET is_complete = true WHERE plan_id = :plan_id"
                    ).bindparams(plan_id=saved.id)
                )
        assert "completion_timestamp_matches_flag" in str(caught.value)

    async def test_set_item_complete_round_trips(self, session: AsyncSession, user_a: str) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        plans = PlanRepository(session)
        plan = make_plan(run.id, make_program(program.slug))
        await plans.save(user_a, plan)

        completed = await plans.set_item_complete(user_a, plan.items[0].id, is_complete=True)
        assert completed is not None
        assert completed.is_complete
        assert completed.completed_at is not None

        reverted = await plans.set_item_complete(user_a, plan.items[0].id, is_complete=False)
        assert reverted is not None
        assert reverted.completed_at is None

    async def test_another_users_item_cannot_be_completed(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        plan = make_plan(run.id, make_program(program.slug))
        await PlanRepository(session).save(user_a, plan)

        assert (
            await PlanRepository(session).set_item_complete(
                user_b, plan.items[0].id, is_complete=True
            )
            is None
        )


class TestRuleVersions:
    async def test_two_open_rule_versions_are_rejected(self, session: AsyncSession) -> None:
        """At most one rule version per program may be open.

        Without the partial unique index, "what are this program's rules today"
        would have two answers and the one a user got would depend on row order.
        """
        program = await _seed_program(session, with_rules=False)
        session.add(
            models.ProgramRuleVersion(
                program_id=program.id,
                version="2026.1",
                effective_from=date(2026, 1, 1),
                effective_to=None,
                root={"kind": "all_of", "description": "x", "conditions": []},
                review_status=ReviewStatus.DRAFT.value,
            )
        )
        await session.flush()

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ProgramRuleVersion(
                        program_id=program.id,
                        version="2026.2",
                        effective_from=date(2026, 6, 1),
                        effective_to=None,
                        root={"kind": "all_of", "description": "y", "conditions": []},
                        review_status=ReviewStatus.DRAFT.value,
                    )
                )
                await session.flush()
        assert "uq_program_rule_versions_program_id_active" in str(caught.value)

    async def test_superseding_closes_the_previous_version_and_keeps_it_readable(
        self, session: AsyncSession
    ) -> None:
        """Old versions stay readable so historical evaluations stay auditable."""
        program = await _seed_program(session, with_rules=False)
        repo = SqlProgramRepository(session)
        await repo.upsert_rule_version(
            make_ruleset(program.slug, version="2026.1", effective_from=date(2026, 1, 1))
        )
        await repo.upsert_rule_version(
            make_ruleset(program.slug, version="2026.2", effective_from=date(2026, 6, 1))
        )

        rows = (
            await session.scalars(
                select(models.ProgramRuleVersion)
                .where(models.ProgramRuleVersion.program_id == program.id)
                .order_by(models.ProgramRuleVersion.version)
            )
        ).all()
        assert [row.version for row in rows] == ["2026.1", "2026.2"]
        assert rows[0].effective_to == date(2026, 6, 1)
        assert rows[1].effective_to is None

        in_force_before = await repo.get_active_ruleset(program.id, on=date(2026, 3, 1))
        in_force_after = await repo.get_active_ruleset(program.id, on=date(2026, 8, 1))
        assert in_force_before is not None
        assert in_force_after is not None
        assert in_force_before.version == "2026.1"
        assert in_force_after.version == "2026.2"

    async def test_an_unapproved_rule_set_is_never_in_force(self, session: AsyncSession) -> None:
        """ "Cannot assess" and "does not qualify" are different answers.

        A draft rule set must not influence anybody's status, so
        ``get_active_ruleset`` returns ``None`` and the caller reports a
        limitation rather than a verdict.
        """
        program = await _seed_program(session, with_rules=False)
        repo = SqlProgramRepository(session)
        await repo.upsert_rule_version(make_ruleset(program.slug, review_status=ReviewStatus.DRAFT))
        assert await repo.get_active_ruleset(program.id, on=date(2026, 3, 1)) is None

    async def test_effective_to_must_be_after_effective_from(self, session: AsyncSession) -> None:
        program = await _seed_program(session, with_rules=False)
        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.ProgramRuleVersion(
                        program_id=program.id,
                        version="bad-dates",
                        effective_from=date(2026, 6, 1),
                        effective_to=date(2026, 1, 1),
                        root={"kind": "all_of", "description": "z", "conditions": []},
                    )
                )
                await session.flush()
        assert "effective_to_after_effective_from" in str(caught.value)


class TestCascades:
    async def test_deleting_a_run_removes_its_matches_plans_and_items(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        schema_program = make_program(program.slug)
        await MatchRepository(session).upsert_many(user_a, run.id, [make_match(schema_program)])
        await PlanRepository(session).save(user_a, make_plan(run.id, schema_program))

        await session.execute(
            text("DELETE FROM workflow_runs WHERE id = :run_id").bindparams(run_id=run.id)
        )
        await session.flush()

        for table in ("program_matches", "action_plans", "action_items"):
            remaining = await session.scalar(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            assert remaining == 0, table

    async def test_deleting_a_plan_removes_its_items(
        self, session: AsyncSession, user_a: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        saved = await PlanRepository(session).save(
            user_a, make_plan(run.id, make_program(program.slug))
        )

        await session.execute(
            text("DELETE FROM action_plans WHERE id = :plan_id").bindparams(plan_id=saved.id)
        )
        await session.flush()
        assert await session.scalar(text("SELECT count(*) FROM action_items")) == 0

    async def test_deleting_a_program_removes_its_sources_and_rules(
        self, session: AsyncSession
    ) -> None:
        program = await _seed_program(session)
        assert await session.scalar(text("SELECT count(*) FROM program_sources")) == 1
        assert await session.scalar(text("SELECT count(*) FROM program_rule_versions")) == 1

        await session.execute(
            text("DELETE FROM assistance_programs WHERE id = :pid").bindparams(pid=program.id)
        )
        await session.flush()
        assert await session.scalar(text("SELECT count(*) FROM program_sources")) == 0
        assert await session.scalar(text("SELECT count(*) FROM program_rule_versions")) == 0

    async def test_a_program_with_saved_matches_cannot_be_deleted(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """``ON DELETE RESTRICT``: a saved result must keep naming its program.

        Retiring a program is a soft delete, which hides it from search while
        leaving an action plan from last month intact.
        """
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        await MatchRepository(session).upsert_many(
            user_a, run.id, [make_match(make_program(program.slug))]
        )

        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                await session.execute(
                    text("DELETE FROM assistance_programs WHERE id = :pid").bindparams(
                        pid=program.id
                    )
                )
        assert "fk_program_matches_program_id_assistance_programs" in str(caught.value)

        await SqlProgramRepository(session).soft_delete_program(program.id)
        page = await SqlProgramRepository(session).list_programs()
        assert page.total == 0
        assert await SqlProgramRepository(session).get_program(program.id) is not None


class TestFullTextSearch:
    async def test_the_search_vector_is_populated_from_every_source_field(
        self, session: AsyncSession
    ) -> None:
        program = await _seed_program(session)
        vector = await session.scalar(
            text("SELECT search_vector::text FROM assistance_programs WHERE id = :pid").bindparams(
                pid=program.id
            )
        )
        assert vector
        # 'nutrition' appears only in ``keywords``, so matching it proves the
        # array is folded into the generated column.
        assert "nutrit" in vector
        # Weighted, so a term in the official name outranks the same term in
        # the body: A for the name, B for the summary and keywords, C for the
        # description.
        assert re.search(r"\d+A\b", vector), vector
        assert re.search(r"\d+B\b", vector), vector
        assert re.search(r"\d+C\b", vector), vector

    async def test_the_generated_column_tracks_an_update(self, session: AsyncSession) -> None:
        """A generated column cannot go stale, which is why it is not a trigger."""
        program = await _seed_program(session)
        await session.execute(
            text(
                "UPDATE assistance_programs SET keywords = ARRAY['kerosene'] WHERE id = :pid"
            ).bindparams(pid=program.id)
        )
        await session.flush()
        vector = await session.scalar(
            text("SELECT search_vector::text FROM assistance_programs WHERE id = :pid").bindparams(
                pid=program.id
            )
        )
        assert "kerosen" in vector
        assert "nutrit" not in vector

    async def test_search_matches_a_keyword_only_term(self, session: AsyncSession) -> None:
        await _seed_program(session)
        repo = SqlProgramRepository(session)
        assert (await repo.list_programs(query="nutrition")).total == 1
        assert (await repo.list_programs(query="orthodontics")).total == 0

    async def test_the_gin_index_can_serve_the_search(self, session: AsyncSession) -> None:
        """Assert the planner *can* use the GIN index.

        Sequential scans are disabled for the length of this statement because
        a table holding one fixture row will always be cheaper to scan than to
        index, and the thing worth testing is that the index exists, covers the
        column and matches the operator - not what the planner does with two
        rows.
        """
        await _seed_program(session)
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (
            (
                await session.execute(
                    text(
                        "EXPLAIN SELECT id FROM assistance_programs "
                        "WHERE search_vector @@ websearch_to_tsquery('english', 'groceries')"
                    )
                )
            )
            .scalars()
            .all()
        )
        rendered = "\n".join(plan)
        assert "ix_assistance_programs_search_vector" in rendered, rendered


class TestProfiles:
    async def test_saving_a_profile_appends_a_version(
        self, session: AsyncSession, user_a: str
    ) -> None:
        repo = ProfileRepository(session)
        first = await repo.save_new_version(user_a, make_profile())
        second = await repo.save_new_version(
            user_a, make_profile(household_size=4), changed_fields=["household_size"]
        )

        assert first.version == 1
        assert second.version == 2
        latest = await repo.get_latest(user_a)
        assert latest is not None
        assert latest.version == 2
        assert latest.profile.household_size == 4
        assert latest.changed_fields == ["household_size"]

        original = await repo.get_version(user_a, 1)
        assert original is not None
        assert original.profile.household_size == 3

    async def test_a_profile_round_trips_through_every_column(
        self, session: AsyncSession, user_a: str
    ) -> None:
        profile = make_profile(
            household_composition=HouseholdComposition(adults=2, children_under_18=1),
            flags={ProfileFlag.RECEIVES_SNAP: True, ProfileFlag.HAS_CHILDREN: True},
            extras={"preferred_language": "es"},
        )
        repo = ProfileRepository(session)
        await repo.save_new_version(user_a, profile)
        loaded = await repo.get_latest(user_a)
        assert loaded is not None
        assert loaded.profile == profile

    async def test_duplicate_user_and_version_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        await ProfileRepository(session).save_new_version(user_a, make_profile())
        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(models.Profile(user_id=user_a, version=1))
                await session.flush()
        assert "uq_profiles_user_id_version" in str(caught.value)

    async def test_an_income_amount_without_a_period_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """An amount with no period cannot be normalised and must not be assumed
        to be monthly - a wrong period turns "we should ask" into "do not bother"."""
        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(models.Profile(user_id=user_a, version=1, income_min_cents=200_000))
                await session.flush()
        assert "income_amount_requires_a_period" in str(caught.value)

    async def test_an_inverted_income_range_is_rejected(
        self, session: AsyncSession, user_a: str
    ) -> None:
        with pytest.raises(IntegrityError) as caught:
            async with session.begin_nested():
                session.add(
                    models.Profile(
                        user_id=user_a,
                        version=1,
                        income_min_cents=500_000,
                        income_max_cents=100_000,
                        income_period="MONTHLY",
                    )
                )
                await session.flush()
        assert "income_range_is_ordered" in str(caught.value)

    async def test_another_users_profile_is_not_readable_by_id(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        repo = ProfileRepository(session)
        row = await repo.save_new_version(user_a, make_profile())
        assert await repo.get_by_id(user_a, row.id) is not None
        assert await repo.get_by_id(user_b, row.id) is None


class TestCatalogReads:
    async def test_program_detail_renders_requirements_from_the_approved_rules(
        self, session: AsyncSession
    ) -> None:
        program = await _seed_program(session)
        detail = await SqlProgramRepository(session).get_program(program.id)
        assert detail is not None
        assert detail.rule_version == "2026.1"
        assert [bullet.text for bullet in detail.requirements] == [
            "You live in New Jersey (fictional requirement).",
            "Your household has at least one person.",
            # Known gaps are rendered as bullets too, marked non-evaluable, so
            # a limitation is shown to the user rather than quietly dropped.
            "This rule set is fictional and encodes nothing real.",
        ]
        assert [bullet.is_evaluable for bullet in detail.requirements] == [True, True, False]

    async def test_program_detail_without_approved_rules_shows_no_requirements(
        self, session: AsyncSession
    ) -> None:
        program = await _seed_program(session, with_rules=False)
        detail = await SqlProgramRepository(session).get_program(program.id)
        assert detail is not None
        assert detail.requirements == []
        assert detail.rule_version is None
        assert detail.limitations

    async def test_filters_compose(self, session: AsyncSession) -> None:
        await _seed_program(session, "example-grocery-help")
        await _seed_program(session, "example-heating-help", with_rules=False)
        await session.execute(
            text(
                "UPDATE assistance_programs SET category = 'UTILITIES', "
                "verification_status = 'NEEDS_REVIEW', last_verified_at = NULL "
                "WHERE slug = 'example-heating-help'"
            )
        )
        await session.flush()
        repo = SqlProgramRepository(session)

        assert (await repo.list_programs()).total == 2
        assert (await repo.list_programs(categories=[ProgramCategory.FOOD])).total == 1
        assert (
            await repo.list_programs(verification_statuses=[VerificationStatus.VERIFIED])
        ).total == 1
        assert (await repo.list_programs(state="NJ")).total == 2
        assert (await repo.list_programs(state="NY")).total == 0
        assert (await repo.list_programs(county="Middlesex")).total == 2
        assert (await repo.list_programs(county="Cape May")).total == 0

    async def test_get_programs_by_ids_preserves_the_requested_order(
        self, session: AsyncSession
    ) -> None:
        first = await _seed_program(session, "example-aaa-help", with_rules=False)
        second = await _seed_program(session, "example-zzz-help", with_rules=False)
        repo = SqlProgramRepository(session)
        got = await repo.get_programs_by_ids([second.id, first.id])
        assert [program.id for program in got] == [second.id, first.id]

    async def test_source_search_is_substring_not_stemmed(self, session: AsyncSession) -> None:
        await _seed_program(session)
        repo = SqlProgramRepository(session)
        found = await repo.search_sources("fictional excerpt")
        assert len(found) == 1
        assert "fictional excerpt" in found[0].excerpt
        assert await repo.search_sources("no such passage") == []

    async def test_upsert_is_idempotent_on_slug(self, session: AsyncSession) -> None:
        repo = SqlProgramRepository(session)
        program = make_program()
        first_id = await repo.upsert_program(program)
        second_id = await repo.upsert_program(program)
        assert first_id == second_id
        assert await session.scalar(text("SELECT count(*) FROM assistance_programs")) == 1
        assert await session.scalar(text("SELECT count(*) FROM program_sources")) == 1


class TestRunHistory:
    async def test_run_summaries_aggregate_counts_and_the_latest_plan(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        program = await _seed_program(session)
        schema_program = make_program(program.slug)
        other = await _seed_program(session, "example-second-help", with_rules=False)
        runs = WorkflowRunRepository(session)

        run = await _seed_run(session, user_a)
        await MatchRepository(session).upsert_many(
            user_a,
            run.id,
            [
                make_match(schema_program, status=MatchStatus.POTENTIAL_MATCH),
                make_match(make_program(other.slug), status=MatchStatus.UNLIKELY_MATCH),
            ],
        )
        await PlanRepository(session).save(user_a, make_plan(run.id, schema_program, version=1))
        await PlanRepository(session).save(user_a, make_plan(run.id, schema_program, version=2))
        await _seed_run(session, user_b)

        page = await runs.list_summaries(user_a)
        assert page.total == 1
        summary = page.items[0]
        assert summary.match_count == 2
        assert summary.potential_match_count == 1
        latest = await PlanRepository(session).get_latest_for_run(user_a, run.id)
        assert latest is not None
        assert summary.plan_id == latest.id

    async def test_another_users_run_is_invisible(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        run = await _seed_run(session, user_a)
        runs = WorkflowRunRepository(session)
        assert await runs.get(user_a, run.id) is not None
        assert await runs.get(user_b, run.id) is None

    async def test_failure_is_recorded_as_user_safe_structured_data(
        self, session: AsyncSession, user_a: str
    ) -> None:
        run = await _seed_run(session, user_a)
        runs = WorkflowRunRepository(session)
        failed = await runs.mark_failed(
            user_a,
            run.id,
            UserFacingError(
                code=ErrorCode.UPSTREAM_TIMEOUT,
                message="We could not reach a service we needed. Your answers are saved.",
                is_retryable=True,
                occurred_at=utcnow(),
            ),
        )
        assert failed is not None
        assert failed.status == WorkflowStatus.FAILED.value
        assert failed.error is not None
        assert failed.error["code"] == ErrorCode.UPSTREAM_TIMEOUT.value
        assert failed.completed_at is not None


class TestAuditEvents:
    async def test_metadata_is_redacted_with_the_same_rules_as_the_logs(
        self, session: AsyncSession, user_a: str
    ) -> None:
        event = await AuditRepository(session).record(
            user_a,
            AuditAction.RUN_CREATED,
            resource_type="workflow_run",
            resource_id=uuid.uuid4(),
            metadata={
                "narrative": "I lost my job last month and cannot buy food.",
                "income_min_cents": 180_000,
                "match_count": 4,
            },
        )
        assert event.event_metadata["narrative"] == "[redacted]"
        assert event.event_metadata["income_min_cents"] == "[redacted]"
        assert event.event_metadata["match_count"] == 4

    async def test_both_identities_are_stored(self, session: AsyncSession, user_a: str) -> None:
        event = await AuditRepository(session).record(user_a, AuditAction.PLAN_VIEWED)
        assert event.user_id == user_a
        assert event.user_id_hash == hash_user_id(user_a)
        assert event.user_id_hash.startswith("u_")

    async def test_only_the_owner_reads_their_trail(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        audit = AuditRepository(session)
        await audit.record(user_a, AuditAction.PLAN_VIEWED)
        await audit.record(user_b, AuditAction.PLAN_VIEWED)
        assert len(await audit.list_for_user(user_a)) == 1


class TestAccountDeletion:
    async def test_everything_including_checkpoints_is_removed(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        """The checkpoint rows are the test that matters.

        They hold the narrative, and they have no ``user_id`` of their own, so
        no cascade can reach them. Deleting a user's runs without deleting their
        checkpoints would leave everything they ever wrote in the database.
        """
        program = await _seed_program(session)
        schema_program = make_program(program.slug)
        run = await _seed_run(session, user_a)
        other_run = await _seed_run(session, user_b)
        await MatchRepository(session).upsert_many(user_a, run.id, [make_match(schema_program)])
        await MatchRepository(session).upsert_many(
            user_b, other_run.id, [make_match(schema_program)]
        )
        await PlanRepository(session).save(user_a, make_plan(run.id, schema_program))
        await ProfileRepository(session).save_new_version(user_a, make_profile())
        await AuditRepository(session).record(user_a, AuditAction.RUN_CREATED)

        # Stand in for the tables ``AsyncPostgresSaver.setup()`` creates. Only
        # ``thread_id`` matters to the deletion routine.
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            await session.execute(
                text(f"CREATE TABLE IF NOT EXISTS {table} (thread_id text, payload text)")
            )
            await session.execute(
                text(
                    f"INSERT INTO {table} (thread_id, payload) VALUES (:a, 'x'), (:b, 'y')"  # noqa: S608
                ).bindparams(a=run.thread_id, b=other_run.thread_id)
            )
        await session.flush()

        receipt = await AccountDataRepository(session).delete_user_data(user_a)

        assert receipt.deleted_counts["profiles"] == 1
        assert receipt.deleted_counts["workflow_runs"] == 1
        assert receipt.deleted_counts["program_matches"] == 1
        assert receipt.deleted_counts["action_plans"] == 1
        assert receipt.deleted_counts["action_items"] == 2
        assert receipt.deleted_counts["audit_events"] == 1
        assert receipt.deleted_counts["checkpoints"] == 1
        assert receipt.deleted_counts["checkpoint_blobs"] == 1
        assert receipt.deleted_counts["checkpoint_writes"] == 1
        assert receipt.retained

        # The other user is untouched.
        assert await WorkflowRunRepository(session).get(user_b, other_run.id) is not None
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            remaining = await session.scalar(
                text(f"SELECT count(*) FROM {table}")  # noqa: S608
            )
            assert remaining == 1, table

        # And the deletion itself is audited, with nothing personal in it.
        trail = await AuditRepository(session).list_for_user(user_a)
        assert [event.action for event in trail] == [AuditAction.ACCOUNT_DATA_DELETED.value]
        assert user_a not in str(trail[0].event_metadata)

        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            await session.execute(text(f"DROP TABLE {table}"))

    async def test_deletion_works_before_any_workflow_has_ever_run(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """The checkpoint tables are created lazily by the checkpointer.

        A user who signed up, changed their mind and asked for deletion must
        not get an error because a table they never caused to exist is missing.
        """
        await ProfileRepository(session).save_new_version(user_a, make_profile())
        receipt = await AccountDataRepository(session).delete_user_data(user_a)
        assert receipt.deleted_counts["profiles"] == 1
        assert receipt.deleted_counts["checkpoints"] == 0

    async def test_model_call_telemetry_survives_with_its_run_reference_nulled(
        self, session: AsyncSession, user_a: str
    ) -> None:
        run = await _seed_run(session, user_a)
        session.add(
            models.ModelCall(
                run_id=run.id,
                model="gemini-2.5-flash-lite",
                source="fallback",
                prompt_tokens=10,
                output_tokens=20,
                total_tokens=30,
            )
        )
        await session.flush()

        await AccountDataRepository(session).delete_user_data(user_a)

        rows = (await session.scalars(select(models.ModelCall))).all()
        assert len(rows) == 1
        assert rows[0].run_id is None
        assert rows[0].total_tokens == 30


class TestTimestamps:
    async def test_timestamps_are_timezone_aware(self, session: AsyncSession, user_a: str) -> None:
        """A naive timestamp anywhere here would corrupt freshness the first
        time the server crossed a DST boundary."""
        run = await _seed_run(session, user_a)
        assert run.created_at.tzinfo is not None
        assert run.updated_at.tzinfo is not None
        assert abs(utcnow() - run.created_at) < timedelta(minutes=5)


class TestSessionManagement:
    """``app.db.session`` itself: the transaction boundaries and the pooler case."""

    async def test_session_scope_commits_on_clean_exit(
        self, engine: AsyncEngine, user_a: str
    ) -> None:
        """A background task's writes must be durable without the task committing.

        This one deliberately does not use the rolled-back ``session`` fixture:
        the whole point is to observe a real ``COMMIT`` from a second session,
        so it cleans up after itself instead.
        """
        factory = create_session_factory(engine)
        try:
            async with session_scope(factory) as scoped:
                await AuditRepository(scoped).record(user_a, AuditAction.RUN_CREATED)

            async with factory() as reader:
                committed = await AuditRepository(reader).list_for_user(user_a)
                assert len(committed) == 1
        finally:
            async with session_scope(factory) as cleanup:
                await cleanup.execute(
                    text("DELETE FROM audit_events WHERE user_id = :uid").bindparams(uid=user_a)
                )

    async def test_session_scope_rolls_back_on_an_exception(
        self, engine: AsyncEngine, user_a: str
    ) -> None:
        """A failed background task must leave nothing behind."""
        factory = create_session_factory(engine)

        with pytest.raises(RuntimeError, match="deliberate"):
            async with session_scope(factory) as scoped:
                await AuditRepository(scoped).record(user_a, AuditAction.RUN_CREATED)
                msg = "deliberate failure after a write"
                raise RuntimeError(msg)

        async with factory() as reader:
            assert await AuditRepository(reader).list_for_user(user_a) == []

    async def test_get_session_rolls_back_a_failed_request(
        self, engine: AsyncEngine, user_a: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The FastAPI dependency does not commit, and cleans up after a raise.

        Not committing is deliberate: a dependency that committed on the way
        out would eventually persist half of a request that failed after its
        last successful statement.
        """
        from app.db import session as session_module

        monkeypatch.setattr(session_module, "_session_factory", create_session_factory(engine))
        monkeypatch.setattr(session_module, "_engine", engine)

        generator = session_module.get_session()
        request_session = await anext(generator)
        await AuditRepository(request_session).record(user_a, AuditAction.PLAN_VIEWED)
        with pytest.raises(RuntimeError, match="handler blew up"):
            await generator.athrow(RuntimeError("handler blew up"))

        async with create_session_factory(engine)() as reader:
            assert await AuditRepository(reader).list_for_user(user_a) == []

    def test_a_pooler_url_disables_the_prepared_statement_cache(self, settings: Settings) -> None:
        """Supabase's transaction-mode pooler cannot carry prepared statements.

        Each transaction may land on a different backend, so a cached
        server-side prepared statement fails with ``prepared statement
        "__asyncpg_stmt_1__" does not exist``. Detecting the endpoint and
        disabling the cache is what will let the production pooler URL be
        swapped in without a code change.
        """
        args = resolve_connect_args(
            settings,
            "postgresql+asyncpg://user:pw@aws-0-us-east-1.pooler.supabase.com:6543/postgres",
        )
        assert args["statement_cache_size"] == 0
        # With the cache off asyncpg still prepares each statement once, and
        # reusing a name across pooled backends is the failure being avoided.
        assert "prepared_statement_name_func" in args
        first = args["prepared_statement_name_func"]()
        assert first != args["prepared_statement_name_func"]()

    def test_the_pooler_port_alone_is_enough_to_detect_one(self, settings: Settings) -> None:
        args = resolve_connect_args(settings, "postgresql+asyncpg://u:p@db.internal:6543/postgres")
        assert args["statement_cache_size"] == 0

    def test_a_direct_url_keeps_the_cache(self, settings: Settings) -> None:
        args = resolve_connect_args(settings, settings.test_database_url_async)
        assert args["statement_cache_size"] > 0
        assert "prepared_statement_name_func" not in args
        # The statement timeout is a server setting, the only way to bound a
        # query that has already left the process.
        assert args["server_settings"]["statement_timeout"] == str(
            int(settings.db_statement_timeout_s * 1000)
        )

    def test_the_cache_size_can_be_forced(self, settings: Settings) -> None:
        """An unrecognised pooler host must still be supportable by config."""
        args = resolve_connect_args(
            settings,
            "postgresql+asyncpg://u:p@proxy.internal:5432/postgres",
            statement_cache_size=0,
        )
        assert args["statement_cache_size"] == 0
        assert "prepared_statement_name_func" in args


class TestRunTransitions:
    """Every state a run can move through, and what each transition clears.

    These are not getter/setter tests. Each transition has one job beyond
    writing a status: ``mark_running`` clears a stale clarification payload and
    a stale error so a resumed run does not keep showing the question it has
    already been answered; ``mark_awaiting_input`` stores only editorial text;
    the terminal transitions stamp ``completed_at``, which
    ``WorkflowRunDetail`` requires for a terminal status.
    """

    async def test_the_full_happy_path(self, session: AsyncSession, user_a: str) -> None:
        runs = WorkflowRunRepository(session)
        run = await _seed_run(session, user_a, idempotency_key="transition-0001")
        profile = await ProfileRepository(session).save_new_version(user_a, make_profile())

        running = await runs.mark_running(user_a, run.id, phase=WorkflowPhase.EXTRACT_PROFILE)
        assert running is not None
        assert running.status == WorkflowStatus.RUNNING.value
        assert running.started_at is not None

        attached = await runs.attach_profile(user_a, run.id, profile.id)
        assert attached is not None
        assert attached.profile_id == profile.id

        advanced = await runs.set_phase(user_a, run.id, WorkflowPhase.EVALUATE_ELIGIBILITY)
        assert advanced is not None
        assert advanced.phase == WorkflowPhase.EVALUATE_ELIGIBILITY.value

        with_usage = await runs.add_model_usage(
            user_a, run.id, [{"model": "gemini-2.5-flash-lite", "source": "fallback"}]
        )
        assert with_usage is not None
        assert len(with_usage.model_usage) == 1

        limited = await runs.add_limitations(
            user_a,
            run.id,
            ["One program has no encoded rules.", "One program has no encoded rules."],
        )
        assert limited is not None
        assert limited.limitations == ["One program has no encoded rules."]

        done = await runs.mark_completed(
            user_a, run.id, limitations=["A source is more than a year old."]
        )
        assert done is not None
        assert done.status == WorkflowStatus.COMPLETED.value
        assert done.phase == WorkflowPhase.DONE.value
        assert done.completed_at is not None
        assert len(done.limitations) == 2

    async def test_resuming_clears_the_open_question_and_any_earlier_error(
        self, session: AsyncSession, user_a: str
    ) -> None:
        runs = WorkflowRunRepository(session)
        run = await _seed_run(session, user_a)

        parked = await runs.mark_awaiting_input(
            user_a,
            run.id,
            phase=WorkflowPhase.AWAIT_CLARIFICATION,
            clarification_payload={"questions": [{"field": "household_size"}]},
        )
        assert parked is not None
        assert parked.status == WorkflowStatus.AWAITING_INPUT.value
        assert parked.clarification_payload is not None

        resumed = await runs.mark_running(user_a, run.id, phase=WorkflowPhase.VALIDATE_PROFILE)
        assert resumed is not None
        assert resumed.clarification_payload is None
        assert resumed.error is None

    async def test_cancelling_is_terminal(self, session: AsyncSession, user_a: str) -> None:
        runs = WorkflowRunRepository(session)
        run = await _seed_run(session, user_a)
        cancelled = await runs.mark_cancelled(user_a, run.id)
        assert cancelled is not None
        assert cancelled.status == WorkflowStatus.CANCELLED.value
        assert cancelled.completed_at is not None

    async def test_no_transition_touches_another_users_run(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        """Every mutator is user-scoped, so a guessed run id changes nothing."""
        runs = WorkflowRunRepository(session)
        run = await _seed_run(session, user_a)
        assert await runs.mark_running(user_b, run.id, phase=WorkflowPhase.DONE) is None
        assert await runs.set_phase(user_b, run.id, WorkflowPhase.DONE) is None
        assert await runs.mark_completed(user_b, run.id) is None
        assert await runs.mark_cancelled(user_b, run.id) is None
        assert await runs.attach_profile(user_b, run.id, uuid.uuid4()) is None
        assert await runs.add_model_usage(user_b, run.id, [{"model": "x"}]) is None
        assert await runs.add_limitations(user_b, run.id, ["x"]) is None
        assert (
            await runs.mark_awaiting_input(
                user_b, run.id, phase=WorkflowPhase.AWAIT_CLARIFICATION, clarification_payload={}
            )
            is None
        )
        untouched = await runs.get(user_a, run.id)
        assert untouched is not None
        assert untouched.status == WorkflowStatus.RUNNING.value

    async def test_a_thread_is_scoped_to_one_run_and_one_user(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        """``thread_id == str(run_id)``, which is what links a user to the
        checkpoint rows holding their narrative."""
        runs = WorkflowRunRepository(session)
        run = await _seed_run(session, user_a)
        assert run.thread_id == str(run.id)
        assert await runs.get_by_thread_id(user_a, run.thread_id) is not None
        assert await runs.get_by_thread_id(user_b, run.thread_id) is None
        assert await runs.thread_ids_for_user(user_a) == [str(run.id)]
        assert await runs.thread_ids_for_user(user_b) == []


class TestRepositoryReadPaths:
    async def test_explanations_are_attached_without_touching_the_verdict(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """The only write path for prose cannot reach ``status``."""
        from app.schemas import MatchExplanation

        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        matches = MatchRepository(session)
        await matches.upsert_many(user_a, run.id, [make_match(make_program(program.slug))])

        written = await matches.set_explanation(
            user_a,
            run.id,
            program.id,
            MatchExplanation(
                headline="You may be able to get help from this fictional program.",
                why_it_may_fit=["You live in the fictional service area."],
                generated_by="deterministic_fallback",
            ),
        )
        assert written is True

        stored = await matches.list_for_run(user_a, run.id)
        assert stored[0].explanation is not None
        assert stored[0].status is MatchStatus.POTENTIAL_MATCH

    async def test_another_users_match_cannot_be_explained(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        from app.schemas import MatchExplanation

        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        matches = MatchRepository(session)
        await matches.upsert_many(user_a, run.id, [make_match(make_program(program.slug))])
        assert (
            await matches.set_explanation(
                user_b,
                run.id,
                program.id,
                MatchExplanation(headline="Not yours to word."),
            )
            is False
        )

    async def test_matches_are_ordered_most_actionable_first(
        self, session: AsyncSession, user_a: str
    ) -> None:
        """The order a user reads is decided in SQL, not by the caller.

        Two screens assembling the same results through different code paths
        would otherwise be free to rank them differently.
        """
        unlikely = await _seed_program(session, "example-aaa-help", with_rules=False)
        potential = await _seed_program(session, "example-zzz-help", with_rules=False)
        run = await _seed_run(session, user_a)
        matches = MatchRepository(session)
        await matches.upsert_many(
            user_a,
            run.id,
            [
                make_match(make_program(unlikely.slug), status=MatchStatus.UNLIKELY_MATCH),
                make_match(make_program(potential.slug), status=MatchStatus.POTENTIAL_MATCH),
            ],
        )
        ordered = await matches.list_for_run(user_a, run.id)
        assert [match.status for match in ordered] == [
            MatchStatus.POTENTIAL_MATCH,
            MatchStatus.UNLIKELY_MATCH,
        ]

    async def test_counts_and_raw_rows_and_deletion(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        matches = MatchRepository(session)
        await matches.upsert_many(user_a, run.id, [make_match(make_program(program.slug))])

        assert await matches.counts_for_run(user_a, run.id) == {MatchStatus.POTENTIAL_MATCH: 1}
        assert len(await matches.list_rows_for_run(user_a, run.id)) == 1
        assert await matches.upsert_many(user_a, run.id, []) == 0

        assert await matches.delete_for_run(user_b, run.id) == 0
        assert await matches.delete_for_run(user_a, run.id) == 1

    async def test_plan_listing_is_paged_and_user_scoped(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        program = await _seed_program(session)
        schema_program = make_program(program.slug)
        plans = PlanRepository(session)
        for user in (user_a, user_b):
            for _ in range(3):
                run = await _seed_run(session, user)
                await plans.save(user, make_plan(run.id, schema_program))

        first_page = await plans.list_for_user(user_a, limit=2)
        assert first_page.total == 3
        assert len(first_page.items) == 2
        assert first_page.has_more is True

        second_page = await plans.list_for_user(user_a, limit=2, offset=2)
        assert len(second_page.items) == 1
        assert second_page.has_more is False

        # Pagination is not a security boundary, so an absurd window is
        # clamped rather than rejected.
        clamped = await plans.list_for_user(user_a, limit=10_000, offset=-5)
        assert clamped.limit == 200
        assert clamped.offset == 0

    async def test_plan_and_item_reads_are_user_scoped(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        program = await _seed_program(session)
        run = await _seed_run(session, user_a)
        plans = PlanRepository(session)
        plan = make_plan(run.id, make_program(program.slug))
        saved = await plans.save(user_a, plan)

        assert await plans.get(user_a, saved.id) is not None
        assert await plans.get(user_b, saved.id) is None
        assert await plans.get_latest_for_run(user_b, run.id) is None
        assert await plans.get_item(user_a, plan.items[0].id) is not None
        assert await plans.get_item(user_b, plan.items[0].id) is None
        assert len(await plans.list_items_for_plan(user_a, saved.id)) == 2
        assert await plans.list_items_for_plan(user_b, saved.id) == []

    async def test_profile_reads_cover_the_row_and_version_list(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        profiles = ProfileRepository(session)
        assert await profiles.get_latest_row(user_a) is None
        assert await profiles.get_latest(user_a) is None

        await profiles.save_new_version(user_a, make_profile())
        await profiles.save_new_version(user_a, make_profile(household_size=5))

        row = await profiles.get_latest_row(user_a)
        assert row is not None
        assert row.version == 2
        assert [snapshot.version for snapshot in await profiles.list_versions(user_a)] == [2, 1]
        assert await profiles.get_version(user_b, 1) is None

    async def test_catalog_reads_by_slug_and_by_source_id(self, session: AsyncSession) -> None:
        program = await _seed_program(session)
        repo = SqlProgramRepository(session)

        by_slug = await repo.get_program_by_slug(program.slug)
        assert by_slug is not None
        assert by_slug.program.id == program.id
        assert await repo.get_program_by_slug("no-such-program") is None
        assert await repo.get_program(uuid.uuid4()) is None

        all_sources = await repo.get_sources(program.id)
        assert len(all_sources) == 1
        wanted = await repo.get_sources(program.id, source_ids=[all_sources[0].id])
        assert len(wanted) == 1
        assert await repo.get_sources(program.id, source_ids=[]) == []
        assert await repo.get_programs_by_ids([]) == []
        assert await repo.search_sources("   ") == []

    async def test_making_a_rule_version_live_is_a_separate_decision(
        self, session: AsyncSession
    ) -> None:
        """Loading a rule set must not make it live.

        Otherwise an ingestion run could change people's results before a
        human had read the new requirements.
        """
        repo = SqlProgramRepository(session)
        program_id = await repo.upsert_program(
            make_program("example-unrated-help", active_rule_version=None)
        )
        program = await session.get_one(models.AssistanceProgram, program_id)
        await repo.upsert_rule_version(make_ruleset(program.slug))
        await session.refresh(program)
        assert program.active_rule_version is None

        await repo.set_active_rule_version(program.id, "2026.1")
        await session.refresh(program)
        assert program.active_rule_version == "2026.1"

    async def test_a_rule_version_for_an_unknown_program_is_an_error(
        self, session: AsyncSession
    ) -> None:
        with pytest.raises(LookupError, match="no program with slug"):
            await SqlProgramRepository(session).upsert_rule_version(
                make_ruleset("example-never-loaded")
            )
