"""``DELETE /api/v1/account/data`` and the system routes.

Deletion is the one promise in this product that cannot be partially kept, so
the tests are about the receipt being *true*: the counts have to match what
actually went, the checkpoint rows that hold the narrative have to be included,
and a second call has to be a no-op rather than an error.

The system routes are in the same file because they are the other two things a
deployment depends on and neither needs a fixture: ``/healthz`` must touch no
dependency (a liveness probe that queries Postgres turns a blip into a restart
loop) and ``/readyz`` must check one.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.repositories import MatchRepository, PlanRepository, WorkflowRunRepository
from app.schemas import AccountDeletionReceipt, HealthResponse, ReadinessResponse
from tests.conftest import make_match, make_plan, make_program, make_ruleset
from tests.security.harness import ApiHarness, rolled_back_api, seed_catalog

__all__ = ["rolled_back_api"]

pytestmark = pytest.mark.db

NARRATIVE = "I need help with groceries and my electric bill this month, things are tight."

USER_TABLES = (
    ("workflow_runs", models.WorkflowRun),
    ("profiles", models.Profile),
    ("program_matches", models.ProgramMatch),
    ("action_plans", models.ActionPlan),
    ("action_items", models.ActionItem),
)


@pytest.fixture
async def populated(session: AsyncSession, rolled_back_api: ApiHarness, user_a: str) -> dict:
    """A user with a profile, a run, a match and a plan, made through the API
    where the API can make it."""
    program = make_program()
    await seed_catalog(session, [program], [make_ruleset(program.slug)])

    intake = await rolled_back_api.client.post(
        "/api/v1/intake", json={"narrative": NARRATIVE}, headers=rolled_back_api.dev(user_a)
    )
    assert intake.status_code == 200, intake.text

    runs = WorkflowRunRepository(session)
    row, _ = await runs.create(user_a, narrative=NARRATIVE)
    await MatchRepository(session).upsert_many(user_a, row.id, [make_match(program)])
    plan = make_plan(row.id, program)
    await PlanRepository(session).save(user_a, plan)
    await session.commit()
    return {"run_id": row.id, "plan": plan, "program": program}


async def _counts(session: AsyncSession, user_id: str) -> dict[str, int]:
    return {
        name: int(
            await session.scalar(
                select(func.count()).select_from(model).where(model.user_id == user_id)
            )
            or 0
        )
        for name, model in USER_TABLES
    }


class TestDeletion:
    async def test_the_receipt_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, populated: dict, user_a: str
    ) -> None:
        response = await rolled_back_api.client.delete(
            "/api/v1/account/data", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        receipt = AccountDeletionReceipt.model_validate(response.json())
        assert receipt.deleted_at
        assert receipt.retained, "a receipt must say what deliberately survives"
        assert all(count >= 0 for count in receipt.deleted_counts.values())

    async def test_the_counts_match_what_actually_went(
        self, rolled_back_api: ApiHarness, session: AsyncSession, populated: dict, user_a: str
    ) -> None:
        before = await _counts(session, user_a)
        assert all(value > 0 for value in before.values()), before

        response = await rolled_back_api.client.delete(
            "/api/v1/account/data", headers=rolled_back_api.dev(user_a)
        )
        receipt = AccountDeletionReceipt.model_validate(response.json())
        for name, count in before.items():
            assert receipt.deleted_counts[name] == count, (
                f"the receipt claims {receipt.deleted_counts[name]} {name} rows, "
                f"but there were {count}"
            )
        assert await _counts(session, user_a) == dict.fromkeys(before, 0)

    async def test_the_checkpoint_tables_are_included_in_the_receipt(
        self, rolled_back_api: ApiHarness, populated: dict, user_a: str
    ) -> None:
        """They hold the narrative and the extracted profile, and no cascade
        reaches them, so a receipt that omitted them would be the worst kind of
        wrong: reassuring and false."""
        response = await rolled_back_api.client.delete(
            "/api/v1/account/data", headers=rolled_back_api.dev(user_a)
        )
        counts = response.json()["deleted_counts"]
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert table in counts, f"{table} is not accounted for in the receipt"

    async def test_a_second_deletion_removes_nothing_but_the_first_receipt(
        self, rolled_back_api: ApiHarness, session: AsyncSession, populated: dict, user_a: str
    ) -> None:
        """Repeating it must be safe, and must not resurrect a count.

        One audit row always survives a deletion: the record that it happened.
        So a second call removes exactly that one row and writes its own, and
        every other table stays at zero. Worth stating explicitly, because
        "deleted_counts is all zeroes" is the intuitive expectation and it is
        wrong by one.
        """
        headers = rolled_back_api.dev(user_a)
        first = await rolled_back_api.client.delete("/api/v1/account/data", headers=headers)
        second = await rolled_back_api.client.delete("/api/v1/account/data", headers=headers)
        assert first.status_code == second.status_code == 200

        counts = second.json()["deleted_counts"]
        assert counts.pop("audit_events") == 1, "the first receipt should have been cleared"
        assert set(counts.values()) == {0}, counts
        assert await _counts(session, user_a) == dict.fromkeys(dict(USER_TABLES), 0)

        remaining = await session.scalar(
            select(func.count())
            .select_from(models.AuditEvent)
            .where(models.AuditEvent.user_id == user_a)
        )
        assert remaining == 1, "exactly one record of the deletion may remain"

    async def test_deleting_with_nothing_to_delete_still_succeeds(
        self, rolled_back_api: ApiHarness, user_b: str
    ) -> None:
        response = await rolled_back_api.client.delete(
            "/api/v1/account/data", headers=rolled_back_api.dev(user_b)
        )
        assert response.status_code == 200
        assert sum(response.json()["deleted_counts"].values()) == 0

    async def test_the_run_is_gone_from_the_api_too(
        self, rolled_back_api: ApiHarness, populated: dict, user_a: str
    ) -> None:
        headers = rolled_back_api.dev(user_a)
        assert (
            await rolled_back_api.client.get(
                f"/api/v1/workflows/{populated['run_id']}", headers=headers
            )
        ).status_code == 200
        assert (
            await rolled_back_api.client.delete("/api/v1/account/data", headers=headers)
        ).status_code == 200
        gone = await rolled_back_api.client.get(
            f"/api/v1/workflows/{populated['run_id']}", headers=headers
        )
        assert gone.status_code == 404
        assert (await rolled_back_api.client.get("/api/v1/plans", headers=headers)).json()[
            "total"
        ] == 0

    async def test_the_catalog_is_not_touched(
        self, rolled_back_api: ApiHarness, session: AsyncSession, populated: dict, user_a: str
    ) -> None:
        """A user deleting their data must not delete public program records."""
        before = await session.scalar(select(func.count()).select_from(models.AssistanceProgram))
        await rolled_back_api.client.delete(
            "/api/v1/account/data", headers=rolled_back_api.dev(user_a)
        )
        after = await session.scalar(select(func.count()).select_from(models.AssistanceProgram))
        assert after == before


class TestTheProbes:
    async def test_liveness_touches_no_dependency(self, rolled_back_api: ApiHarness) -> None:
        for path in ("/healthz", "/api/v1/health"):
            response = await rolled_back_api.client.get(path)
            assert response.status_code == 200, path
            health = HealthResponse.model_validate(response.json())
            assert health.status == "ok"
            assert health.app_env == "local"

    async def test_readiness_checks_the_database(self, rolled_back_api: ApiHarness) -> None:
        for path in ("/readyz", "/api/v1/ready"):
            response = await rolled_back_api.client.get(path)
            assert response.status_code == 200, path
            ready = ReadinessResponse.model_validate(response.json())
            assert ready.ready is True
            names = {dependency.name for dependency in ready.dependencies}
            assert "database" in names
            database = next(d for d in ready.dependencies if d.name == "database")
            assert database.ok is True
            assert database.latency_ms is not None

    async def test_no_probe_leaks_a_connection_string_or_a_version(
        self, rolled_back_api: ApiHarness
    ) -> None:
        """An unauthenticated endpoint should not be a reconnaissance tool."""
        for path in ("/healthz", "/readyz"):
            text = (await rolled_back_api.client.get(path)).text
            assert "postgres" not in text.lower()
            assert "5433" not in text
            assert "password" not in text.lower()
            assert "secondchance:" not in text

    async def test_model_access_is_reported_but_not_required(
        self, rolled_back_api: ApiHarness
    ) -> None:
        """The whole product works without it, so it must not fail readiness."""
        response = await rolled_back_api.client.get("/readyz")
        ready = ReadinessResponse.model_validate(response.json())
        gemini = next(d for d in ready.dependencies if d.name == "gemini")
        assert gemini.ok is False, "the harness configures no key"
        assert ready.ready is True
        assert gemini.note


class TestTheErrorEnvelope:
    async def test_an_unknown_path_uses_the_standard_envelope(
        self, rolled_back_api: ApiHarness
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"error"}
        assert body["error"]["code"] == "not_found"
        assert body["error"]["message"]

    async def test_a_wrong_method_uses_the_standard_envelope(
        self, rolled_back_api: ApiHarness
    ) -> None:
        response = await rolled_back_api.client.delete("/api/v1/programs")
        assert response.status_code == 405
        assert set(response.json()) == {"error"}

    async def test_an_inbound_request_id_is_echoed_back(self, rolled_back_api: ApiHarness) -> None:
        """So a user can quote one id across a log, a trace and a screenshot."""
        supplied = "0cd7e887a543be7f3bb3f046fbd6f51f"
        response = await rolled_back_api.client.get("/healthz", headers={"X-Request-ID": supplied})
        assert response.headers["X-Request-ID"] == supplied

    async def test_an_error_body_carries_the_same_request_id_as_the_header(
        self, rolled_back_api: ApiHarness
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/does-not-exist")
        assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]

    async def test_the_metrics_endpoint_carries_no_user_derived_label(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        await rolled_back_api.client.get("/api/v1/workflows", headers=rolled_back_api.dev(user_a))
        response = await rolled_back_api.client.get("/internal/metrics")
        assert response.status_code == 200
        assert user_a not in response.text
        assert NARRATIVE not in response.text
        snapshot = await rolled_back_api.client.get("/internal/metrics.json")
        assert snapshot.status_code == 200
        assert user_a not in snapshot.text
