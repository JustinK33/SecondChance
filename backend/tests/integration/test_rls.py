"""Row-level security: user B's rows are invisible and unmodifiable to user A.

This is the test the whole RLS migration exists for, and it only means
something because of two details.

**It runs as a non-owner role.** RLS does not apply to a table's owner unless
``FORCE ROW LEVEL SECURITY`` is set, and it is deliberately not set, because the
local ``AUTH_MODE=dev`` path connects as the owner. So every assertion here is
made after ``SET LOCAL ROLE secondchance_rls_test`` - a role created
``NOSUPERUSER NOBYPASSRLS NOINHERIT`` by the ``rls_role`` fixture. Without the
role switch this file would pass against a database with no policies at all,
which is the most expensive kind of green test.

**It sets ``request.jwt.claims``, not a custom variable.** That is the setting
Supabase's PostgREST populates and the one ``auth.uid()`` reads, on Supabase and
through the local fallback the migration installs. So the policies exercised
here are the policies that will be exercised in production, rather than a
local-only approximation of them.

What is checked, for all six user-owned tables: the other user's rows cannot be
read, updated or deleted, and a row cannot be *written* carrying someone else's
id. Plus the fail-closed cases - no claims and malformed claims both yield no
access rather than full access - and the catalog's public-read,
service-role-write split.

Supabase itself is not provisioned yet, so what is verified here is the policy
logic against a compatible ``auth.uid()``. The policies have not been run
against a live Supabase project; see the handoff notes in ``docs/``.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.repositories import (
    AuditAction,
    AuditRepository,
    MatchRepository,
    PlanRepository,
    ProfileRepository,
    SqlProgramRepository,
    WorkflowRunRepository,
)
from app.schemas import WorkflowPhase, WorkflowStatus
from tests.conftest import make_match, make_plan, make_profile, make_program, make_ruleset

pytestmark = pytest.mark.db

NARRATIVE = "A fictional narrative written by the test suite, not by a person."

#: The six tables the migration protects, with the column holding the row id,
#: so one parametrised assertion covers all of them.
PROTECTED_TABLES: tuple[str, ...] = (
    "profiles",
    "workflow_runs",
    "program_matches",
    "action_plans",
    "action_items",
    "audit_events",
)

CATALOG_TABLES: tuple[str, ...] = (
    "assistance_programs",
    "program_sources",
    "program_rule_versions",
)


async def _set_claims(session: AsyncSession, user_id: str | None) -> None:
    """Set (or clear) the JWT claims for the rest of this transaction."""
    claims = json.dumps({"sub": user_id, "role": "authenticated"}) if user_id else ""
    await session.execute(
        text("SELECT set_config('request.jwt.claims', :claims, true)").bindparams(claims=claims)
    )


async def _act_as(session: AsyncSession, role: str, user_id: str | None) -> None:
    """Become ``role`` with ``user_id``'s claims, for the rest of this transaction."""
    await session.execute(text("RESET ROLE"))
    await _set_claims(session, user_id)
    await session.execute(text(f"SET LOCAL ROLE {role}"))


async def _seed_two_users(session: AsyncSession, user_a: str, user_b: str) -> dict[str, object]:
    """Give both users one row in every protected table, as the table owner."""
    programs = models.AssistanceProgram
    catalog = SqlProgramRepository(session)
    program = make_program("example-grocery-help")
    program_id = await catalog.upsert_program(program)
    await catalog.upsert_rule_version(make_ruleset("example-grocery-help"))
    await catalog.set_active_rule_version(program_id, "2026.1")
    program_row = await session.get_one(programs, program_id)

    seeded: dict[str, object] = {"program_id": program_id}
    for label, user_id in (("a", user_a), ("b", user_b)):
        profile = await ProfileRepository(session).save_new_version(user_id, make_profile())
        run, _ = await WorkflowRunRepository(session).create(
            user_id,
            narrative=NARRATIVE,
            status=WorkflowStatus.RUNNING,
            phase=WorkflowPhase.EVALUATE_ELIGIBILITY,
        )
        await MatchRepository(session).upsert_many(
            user_id, run.id, [make_match(make_program(program_row.slug))]
        )
        plan = make_plan(run.id, make_program(program_row.slug))
        saved_plan = await PlanRepository(session).save(user_id, plan)
        await AuditRepository(session).record(user_id, AuditAction.RUN_CREATED, run_id=run.id)
        seeded[f"profile_{label}"] = profile.id
        seeded[f"run_{label}"] = run.id
        seeded[f"plan_{label}"] = saved_plan.id
        seeded[f"item_{label}"] = plan.items[0].id
    await session.flush()
    return seeded


class TestSetup:
    async def test_the_test_role_cannot_bypass_rls(
        self, session: AsyncSession, rls_role: str, user_a: str
    ) -> None:
        """If this fails, every other assertion in this file is meaningless."""
        await _act_as(session, rls_role, user_a)
        current = await session.scalar(text("SELECT current_user"))
        assert current == rls_role
        attributes = (
            await session.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).one()
        assert attributes.rolsuper is False
        assert attributes.rolbypassrls is False

    async def test_every_protected_table_has_rls_enabled(self, session: AsyncSession) -> None:
        rows = (
            await session.execute(
                text(
                    "SELECT tablename, rowsecurity FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename = ANY(:names)"
                ).bindparams(names=list(PROTECTED_TABLES + CATALOG_TABLES + ("model_calls",)))
            )
        ).all()
        enabled = {row.tablename: row.rowsecurity for row in rows}
        assert len(enabled) == len(PROTECTED_TABLES) + len(CATALOG_TABLES) + 1
        assert all(enabled.values()), enabled

    async def test_auth_uid_reads_the_jwt_sub_claim(
        self, session: AsyncSession, user_a: str
    ) -> None:
        await _set_claims(session, user_a)
        assert str(await session.scalar(text("SELECT auth.uid()"))) == user_a

    async def test_the_owner_connection_still_sees_everything(
        self, session: AsyncSession, user_a: str, user_b: str
    ) -> None:
        """Documents the local ``AUTH_MODE=dev`` behaviour rather than hiding it.

        The application role owns these tables, so RLS is inert for it. That is
        why the repositories filter by ``user_id`` themselves and why
        ``tests/security/`` has to assert isolation at the service layer too -
        in local development the database is not the thing stopping a leak.
        """
        await _seed_two_users(session, user_a, user_b)
        total = await session.scalar(text("SELECT count(*) FROM profiles"))
        assert total == 2


class TestReadIsolation:
    @pytest.mark.parametrize("table", PROTECTED_TABLES)
    async def test_only_the_acting_users_rows_are_visible(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, table: str
    ) -> None:
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)

        visible = (
            (
                await session.execute(
                    text(f"SELECT user_id FROM {table}")  # noqa: S608 - module constant
                )
            )
            .scalars()
            .all()
        )
        # Row counts differ per table (a plan has several items), so assert on
        # ownership: every visible row belongs to the acting user, and at least
        # one is visible - otherwise a missing GRANT would pass as isolation.
        assert visible, table
        assert set(visible) == {user_a}, table
        assert user_b not in visible

    @pytest.mark.parametrize("table", PROTECTED_TABLES)
    async def test_no_claims_means_no_access(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, table: str
    ) -> None:
        """Fail closed. An unauthenticated connection must see nothing at all."""
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, None)
        count = await session.scalar(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
        assert count == 0, table

    @pytest.mark.parametrize("claims", ["not json at all", '{"sub": "not-a-uuid"}', "{}"])
    async def test_malformed_claims_mean_no_access(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, claims: str
    ) -> None:
        """A broken token must not raise, and must not grant anything.

        An exception here would be worse than it looks: a policy that errors is
        a policy whose behaviour depends on how the caller handles the error,
        and the fallback ``auth.uid()`` therefore returns ``NULL`` for anything
        it cannot parse.
        """
        await _seed_two_users(session, user_a, user_b)
        await session.execute(text("RESET ROLE"))
        await session.execute(
            text("SELECT set_config('request.jwt.claims', :claims, true)").bindparams(claims=claims)
        )
        await session.execute(text(f"SET LOCAL ROLE {rls_role}"))
        assert await session.scalar(text("SELECT auth.uid()")) is None
        assert await session.scalar(text("SELECT count(*) FROM profiles")) == 0

    async def test_switching_users_switches_the_visible_rows(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str
    ) -> None:
        await _seed_two_users(session, user_a, user_b)

        await _act_as(session, rls_role, user_a)
        assert await session.scalar(text("SELECT count(*) FROM workflow_runs")) == 1
        await _act_as(session, rls_role, user_b)
        assert await session.scalar(text("SELECT count(*) FROM workflow_runs")) == 1
        seen = await session.scalar(text("SELECT user_id FROM workflow_runs"))
        assert seen == user_b

    async def test_a_repository_read_of_another_users_row_finds_nothing(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str
    ) -> None:
        """Belt and braces: the repository filter and the policy agree."""
        seeded = await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)
        runs = WorkflowRunRepository(session)
        assert await runs.get(user_a, seeded["run_a"]) is not None  # type: ignore[arg-type]
        assert await runs.get(user_a, seeded["run_b"]) is None  # type: ignore[arg-type]


class TestWriteIsolation:
    @pytest.mark.parametrize("table", PROTECTED_TABLES)
    async def test_another_users_rows_cannot_be_updated(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, table: str
    ) -> None:
        """An invisible row is also an unmodifiable one.

        ``USING`` is what makes this true: the ``UPDATE`` matches no rows
        rather than being refused, so an attacker learns nothing about whether
        the row exists.
        """
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)
        result = await session.execute(
            text(f"UPDATE {table} SET updated_at = now() WHERE user_id = :victim")  # noqa: S608
            if table != "audit_events"
            else text(f"UPDATE {table} SET request_id = 'x' WHERE user_id = :victim"),  # noqa: S608
            {"victim": user_b},
        )
        assert result.rowcount == 0, table

    @pytest.mark.parametrize("table", PROTECTED_TABLES)
    async def test_another_users_rows_cannot_be_deleted(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, table: str
    ) -> None:
        await _seed_two_users(session, user_a, user_b)

        await _act_as(session, rls_role, user_b)
        before = await session.scalar(
            text(f"SELECT count(*) FROM {table}")  # noqa: S608
        )
        assert before and before > 0, table

        await _act_as(session, rls_role, user_a)
        result = await session.execute(
            text(f"DELETE FROM {table} WHERE user_id = :victim").bindparams(  # noqa: S608
                victim=user_b
            )
        )
        assert result.rowcount == 0, table

        await _act_as(session, rls_role, user_b)
        after = await session.scalar(
            text(f"SELECT count(*) FROM {table}")  # noqa: S608
        )
        assert after == before, table

    async def test_a_row_cannot_be_inserted_with_someone_elses_id(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str
    ) -> None:
        """``WITH CHECK`` is the half that is easy to forget.

        Without it, a user could insert rows stamped with another user's id -
        invisible to themselves, visible to the victim. Planting data in
        somebody else's account is a strange attack but a real one.
        """
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)

        with pytest.raises(DBAPIError) as caught:
            await session.execute(
                text(
                    "INSERT INTO profiles (id, user_id, version) VALUES (:id, :victim, 99)"
                ).bindparams(id=uuid.uuid4(), victim=user_b)
            )
        assert "row-level security" in str(caught.value).lower()
        await session.rollback()

    async def test_a_row_may_be_inserted_with_the_acting_users_own_id(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str
    ) -> None:
        """The policy has to allow the legitimate case, or it is just a denial."""
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)
        await session.execute(
            text("INSERT INTO profiles (id, user_id, version) VALUES (:id, :me, 99)").bindparams(
                id=uuid.uuid4(), me=user_a
            )
        )
        assert await session.scalar(text("SELECT count(*) FROM profiles")) == 2

    async def test_a_row_cannot_be_reassigned_to_another_user(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str
    ) -> None:
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)
        with pytest.raises(DBAPIError) as caught:
            await session.execute(
                text("UPDATE profiles SET user_id = :victim WHERE user_id = :me").bindparams(
                    victim=user_b, me=user_a
                )
            )
        assert "row-level security" in str(caught.value).lower()
        await session.rollback()


class TestCatalogPolicies:
    @pytest.mark.parametrize("table", CATALOG_TABLES)
    async def test_the_catalog_is_publicly_readable(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, table: str
    ) -> None:
        """Programs are public information. ``GET /api/v1/programs`` needs no token."""
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)
        count = await session.scalar(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
        assert count == 1, table

    async def test_the_catalog_is_readable_with_no_claims_at_all(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str
    ) -> None:
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, None)
        assert await session.scalar(text("SELECT count(*) FROM assistance_programs")) == 1

    @pytest.mark.parametrize("table", CATALOG_TABLES)
    async def test_the_catalog_is_not_writable(
        self, session: AsyncSession, rls_role: str, user_a: str, user_b: str, table: str
    ) -> None:
        """Public read, service-role write.

        There is no ``INSERT``/``UPDATE``/``DELETE`` policy on these tables, and
        with RLS enabled the absence of a permissive policy is a denial. Only a
        role that bypasses RLS - Supabase's ``service_role``, or the owner - can
        change the catalog.
        """
        await _seed_two_users(session, user_a, user_b)
        await _act_as(session, rls_role, user_a)
        with pytest.raises((DBAPIError, ProgrammingError)):
            await session.execute(
                text(f"DELETE FROM {table}")  # noqa: S608
            )
        await session.rollback()


class TestServiceOnlyTables:
    async def test_model_call_telemetry_is_unreachable(
        self, session: AsyncSession, rls_role: str, user_a: str
    ) -> None:
        """``model_calls`` is operator telemetry: RLS on, no policies, no grants."""
        await _act_as(session, rls_role, user_a)
        with pytest.raises((DBAPIError, ProgrammingError)) as caught:
            await session.execute(text("SELECT count(*) FROM model_calls"))
        assert "permission denied" in str(caught.value).lower()
        await session.rollback()
