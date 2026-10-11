"""Two users, one database: nothing of A's is reachable, readable or writable by B.

The property under test is stricter than "B gets an error". It is:

* **404, not 403.** A resource belonging to somebody else must be
  indistinguishable from one that does not exist, or the API becomes an oracle
  for which ids are real.
* **No write happened.** A rejected request has to be inert. Every test here
  snapshots A's rows before B's attempt and compares them afterwards, because
  "it returned 404" and "it returned 404 after updating the row" look
  identical from the outside.
* **Identity comes from the token.** A ``user_id`` in a query string is
  ignored, a ``user_id`` in a body is refused by the schema, and a ``run_id``
  the caller does not own scopes nothing.

``AUTH_MODE=dev`` connects as the database owner and therefore bypasses RLS,
which is exactly why these assertions live at the API and service layer rather
than relying on the policies in the migrations. ``tests/integration/test_rls.py``
covers the policies themselves.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.repositories import (
    MatchRepository,
    PlanRepository,
    ProfileRepository,
    WorkflowRunRepository,
)
from app.schemas import AssistanceProgram, MatchStatus, WorkflowPhase, WorkflowStatus
from tests.conftest import make_match, make_plan, make_profile, make_program, make_ruleset
from tests.security.harness import ApiHarness, rolled_back_api, seed_catalog

__all__ = ["rolled_back_api"]

pytestmark = pytest.mark.db

NARRATIVE = "I need help with groceries and my electric bill this month, things are tight."


@dataclass(slots=True)
class OwnedData:
    """Everything user A owns, and the shape of it before B interferes."""

    program: AssistanceProgram
    run_id: uuid.UUID
    plan_id: uuid.UUID
    item_id: uuid.UUID
    profile_version: int


async def _give_user_data(session: AsyncSession, user_id: str) -> OwnedData:
    """Create a completed run with a match and a two-step plan for ``user_id``.

    Built through the repositories rather than through the API because this is
    the *fixture*, not the thing under test: a full workflow run per case would
    make the suite slow without making the isolation assertions any stronger.
    """
    program = make_program()
    await seed_catalog(session, [program], [make_ruleset(program.slug)])

    runs = WorkflowRunRepository(session)
    row, _ = await runs.create(user_id, narrative=NARRATIVE)
    profile_row = await ProfileRepository(session).save_new_version(
        user_id, make_profile(), changed_fields=["state"]
    )
    await runs.attach_profile(user_id, row.id, profile_row.id)
    await MatchRepository(session).upsert_many(
        user_id, row.id, [make_match(program, with_explanation=True)]
    )
    plan = make_plan(row.id, program)
    await PlanRepository(session).save(user_id, plan)
    await runs.mark_completed(user_id, row.id)
    await session.commit()
    return OwnedData(
        program=program,
        run_id=row.id,
        plan_id=plan.id,
        item_id=plan.items[0].id,
        profile_version=profile_row.version,
    )


async def _snapshot(session: AsyncSession, data: OwnedData, user_id: str) -> dict[str, Any]:
    """Everything about A's rows that an intrusion could plausibly change."""
    run = await WorkflowRunRepository(session).get(user_id, data.run_id)
    assert run is not None
    item = await PlanRepository(session).get_item(user_id, data.item_id)
    assert item is not None
    matches = await MatchRepository(session).list_for_run(user_id, data.run_id)
    counts = {
        table: await session.scalar(
            select(func.count()).select_from(model).where(model.user_id == user_id)
        )
        for table, model in (
            ("workflow_runs", models.WorkflowRun),
            ("profiles", models.Profile),
            ("program_matches", models.ProgramMatch),
            ("action_plans", models.ActionPlan),
            ("action_items", models.ActionItem),
        )
    }
    return {
        "status": run.status,
        "phase": run.phase,
        "updated_at": run.updated_at,
        "item_complete": item.is_complete,
        "item_completed_at": item.completed_at,
        "match_statuses": sorted(match.status.value for match in matches),
        "counts": counts,
    }


@pytest.fixture
async def owned(session: AsyncSession, user_a: str) -> OwnedData:
    return await _give_user_data(session, user_a)


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
class TestUserBCannotRead:
    """Every read of somebody else's resource is a 404 with no detail."""

    @pytest.mark.parametrize("resource", ["run", "plan"])
    async def test_a_resource_of_anothers_is_not_found(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        user_b: str,
        resource: str,
    ) -> None:
        path = (
            f"/api/v1/workflows/{owned.run_id}"
            if resource == "run"
            else f"/api/v1/plans/{owned.plan_id}"
        )
        response = await rolled_back_api.client.get(path, headers=rolled_back_api.dev(user_b))
        assert response.status_code == 404, response.text
        error = response.json()["error"]
        assert error["code"] == "not_found"
        # Nothing in the message may confirm the id exists.
        assert str(owned.run_id) not in response.text
        assert str(owned.plan_id) not in response.text

    async def test_the_owner_can_read_the_same_resources(
        self, rolled_back_api: ApiHarness, owned: OwnedData, user_a: str
    ) -> None:
        """The counterpart, so the 404s above are isolation and not breakage."""
        headers = rolled_back_api.dev(user_a)
        run = await rolled_back_api.client.get(f"/api/v1/workflows/{owned.run_id}", headers=headers)
        assert run.status_code == 200, run.text
        assert run.json()["matches"], "A's own run must show A's matches"
        assert run.json()["profile"] is not None, "A's own run must show A's profile"
        plan = await rolled_back_api.client.get(f"/api/v1/plans/{owned.plan_id}", headers=headers)
        assert plan.status_code == 200, plan.text
        assert len(plan.json()["items"]) == 2

    async def test_listing_shows_only_your_own(
        self, rolled_back_api: ApiHarness, owned: OwnedData, user_a: str, user_b: str
    ) -> None:
        for path in ("/api/v1/workflows", "/api/v1/plans"):
            mine = await rolled_back_api.client.get(path, headers=rolled_back_api.dev(user_a))
            theirs = await rolled_back_api.client.get(path, headers=rolled_back_api.dev(user_b))
            assert mine.status_code == theirs.status_code == 200
            assert mine.json()["total"] == 1, f"{path} lost A's own row"
            assert theirs.json()["total"] == 0, f"{path} leaked A's row to B"
            assert theirs.json()["items"] == []

    async def test_a_user_id_in_the_query_string_is_ignored(
        self, rolled_back_api: ApiHarness, owned: OwnedData, user_a: str, user_b: str
    ) -> None:
        """Identity comes from the token, so this parameter means nothing."""
        for path in ("/api/v1/workflows", "/api/v1/plans"):
            response = await rolled_back_api.client.get(
                f"{path}?user_id={user_a}", headers=rolled_back_api.dev(user_b)
            )
            assert response.status_code == 200
            assert response.json()["total"] == 0, f"{path} honoured a user_id in the query"

    async def test_a_run_id_of_anothers_scopes_nothing_in_the_assistant(
        self, rolled_back_api: ApiHarness, owned: OwnedData, user_b: str
    ) -> None:
        """Ignored rather than rejected: refusing would confirm the id exists.

        "Ignored" is asserted by comparison rather than by inspection. The same
        question asked with and without A's ``run_id`` has to come back with the
        same answer and the same citations; anything A's run contributed would
        show up as a difference.
        """
        question = "What documents does this program ask for?"
        borrowed = await rolled_back_api.client.post(
            "/api/v1/assistant/chat",
            json={"message": question, "run_id": str(owned.run_id)},
            headers=rolled_back_api.dev(user_b),
        )
        unscoped = await rolled_back_api.client.post(
            "/api/v1/assistant/chat",
            json={"message": question},
            headers=rolled_back_api.dev(user_b),
        )

        assert borrowed.status_code == unscoped.status_code == 200, borrowed.text
        assert str(owned.run_id) not in borrowed.text

        def grounding(response: Any) -> tuple[str, list[str], list[str]]:
            body = response.json()
            return (
                body["answer"],
                sorted(citation["source_id"] for citation in body["citations"]),
                sorted(fact["text"] for fact in body["facts"]),
            )

        assert grounding(borrowed) == grounding(unscoped), "A's run changed the answer B received"


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------
class TestUserBCannotWrite:
    """Every write against somebody else's resource is a 404 and a no-op."""

    async def test_patching_anothers_action_item_changes_nothing(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{owned.item_id}",
            json={"is_complete": True},
            headers=rolled_back_api.dev(user_b),
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
        assert await _snapshot(session, owned, user_a) == before

    async def test_resuming_anothers_run_changes_nothing(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{owned.run_id}/resume",
            json={"answers": []},
            headers=rolled_back_api.dev(user_b),
        )
        assert response.status_code == 404
        assert await _snapshot(session, owned, user_a) == before
        assert rolled_back_api.runner.resumed == [], "B's attempt scheduled work on A's run"
        assert rolled_back_api.runner.continued == []

    async def test_cancelling_anothers_run_changes_nothing(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{owned.run_id}/cancel", headers=rolled_back_api.dev(user_b)
        )
        assert response.status_code == 404
        assert await _snapshot(session, owned, user_a) == before
        assert rolled_back_api.runner.cancelled == []

    async def test_re_evaluating_anothers_run_creates_nothing(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(owned.run_id), "changes": {"household_size": 9}},
            headers=rolled_back_api.dev(user_b),
        )
        assert response.status_code == 404
        assert await _snapshot(session, owned, user_a) == before
        b_runs = await session.scalar(
            select(func.count())
            .select_from(models.WorkflowRun)
            .where(models.WorkflowRun.user_id == user_b)
        )
        assert b_runs == 0, "B got a run out of A's profile"

    async def test_a_user_id_in_the_body_is_refused_by_the_schema(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        """Not silently dropped - refused, because there is nowhere to put it."""
        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE, "user_id": user_a},
            headers=rolled_back_api.dev(user_b),
        )
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert error["detail"]["fields"] == ["user_id"]
        assert user_a not in response.text, "a rejected field must not be echoed back"
        assert await _snapshot(session, owned, user_a) == before

    async def test_deleting_your_own_data_leaves_the_other_user_untouched(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.delete(
            "/api/v1/account/data", headers=rolled_back_api.dev(user_b)
        )
        assert response.status_code == 200, response.text
        counts = response.json()["deleted_counts"]
        assert sum(counts.values()) == 0, f"B's deletion removed rows it does not own: {counts}"
        assert await _snapshot(session, owned, user_a) == before

    async def test_the_owner_can_still_do_all_of_it(
        self, rolled_back_api: ApiHarness, owned: OwnedData, user_a: str
    ) -> None:
        """The counterpart again: the 404s are about ownership, not plumbing."""
        headers = rolled_back_api.dev(user_a)
        ticked = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{owned.item_id}",
            json={"is_complete": True},
            headers=headers,
        )
        assert ticked.status_code == 200, ticked.text
        assert ticked.json()["is_complete"] is True

        reevaluated = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(owned.run_id), "changes": {"household_size": 9}},
            headers=headers,
        )
        assert reevaluated.status_code == 202, reevaluated.text
        assert reevaluated.json()["status"] == WorkflowStatus.COMPLETED.value

        receipt = await rolled_back_api.client.delete("/api/v1/account/data", headers=headers)
        assert receipt.status_code == 200
        assert receipt.json()["deleted_counts"]["workflow_runs"] >= 1


# ---------------------------------------------------------------------------
# the audit trail
# ---------------------------------------------------------------------------
class TestTheAuditTrail:
    """A rejected attempt must not write an audit row under A's name."""

    async def test_a_rejected_attempt_writes_no_audit_row_for_the_owner(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
    ) -> None:
        before = await session.scalar(
            select(func.count())
            .select_from(models.AuditEvent)
            .where(models.AuditEvent.user_id == user_a)
        )
        for method, path, body in (
            ("GET", f"/api/v1/workflows/{owned.run_id}", None),
            ("GET", f"/api/v1/plans/{owned.plan_id}", None),
            ("PATCH", f"/api/v1/action-items/{owned.item_id}", {"is_complete": True}),
            ("POST", f"/api/v1/workflows/{owned.run_id}/cancel", None),
        ):
            response = await rolled_back_api.client.request(
                method, path, json=body, headers=rolled_back_api.dev(user_b)
            )
            assert response.status_code == 404, f"{method} {path}"
        after = await session.scalar(
            select(func.count())
            .select_from(models.AuditEvent)
            .where(models.AuditEvent.user_id == user_a)
        )
        assert after == before

    async def test_no_audit_row_ever_stores_a_raw_identifier_in_its_metadata(
        self, rolled_back_api: ApiHarness, owned: OwnedData, session: AsyncSession, user_a: str
    ) -> None:
        ticked = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{owned.item_id}",
            json={"is_complete": True},
            headers=rolled_back_api.dev(user_a),
        )
        assert ticked.status_code == 200
        rows = (
            await session.scalars(
                select(models.AuditEvent).where(models.AuditEvent.user_id == user_a)
            )
        ).all()
        assert rows, "an audited action produced no audit row"
        for row in rows:
            assert row.user_id_hash.startswith("u_")
            assert user_a not in repr(row.event_metadata)
            assert NARRATIVE not in repr(row.event_metadata)


# ---------------------------------------------------------------------------
# a sanity check on the fixture
# ---------------------------------------------------------------------------
async def test_the_two_users_are_different(user_a: str, user_b: str) -> None:
    assert user_a != user_b
    assert uuid.UUID(user_a) != uuid.UUID(user_b)


async def test_the_fixture_really_created_data(
    owned: OwnedData, session: AsyncSession, user_a: str
) -> None:
    snapshot = await _snapshot(session, owned, user_a)
    assert snapshot["counts"] == {
        "workflow_runs": 1,
        "profiles": 1,
        "program_matches": 1,
        "action_plans": 1,
        "action_items": 2,
    }
    assert snapshot["status"] == WorkflowStatus.COMPLETED.value
    assert snapshot["phase"] == WorkflowPhase.DONE.value
    assert snapshot["match_statuses"] == [MatchStatus.POTENTIAL_MATCH.value]


# ---------------------------------------------------------------------------
# the blanket rule, enumerated from the app itself
# ---------------------------------------------------------------------------
#: Every operation that takes a user-owned resource id in its path, with the
#: body it needs. Checked against the live OpenAPI document below, so a new
#: route cannot be added without either covering it here or failing the suite.
OWNED_ID_OPERATIONS: tuple[tuple[str, str, dict[str, Any] | None], ...] = (
    ("GET", "/api/v1/workflows/{run_id}", None),
    ("POST", "/api/v1/workflows/{run_id}/resume", {"answers": []}),
    ("POST", "/api/v1/workflows/{run_id}/cancel", None),
    ("GET", "/api/v1/plans/{plan_id}", None),
    ("PATCH", "/api/v1/action-items/{item_id}", {"is_complete": True}),
)

#: Path parameters that name something a user owns. ``program_id`` is
#: deliberately absent: the catalog is public and shared.
OWNED_ID_PARAMS: frozenset[str] = frozenset({"run_id", "plan_id", "item_id"})


def _declared_owned_id_operations(app: Any) -> set[tuple[str, str]]:
    """Read the operations that take a user-owned id out of the OpenAPI spec."""
    found: set[tuple[str, str]] = set()
    for path, operations in app.openapi()["paths"].items():
        if not any(f"{{{name}}}" in path for name in OWNED_ID_PARAMS):
            continue
        for method in operations:
            if method.upper() in {"GET", "POST", "PATCH", "PUT", "DELETE"}:
                found.add((method.upper(), path))
    return found


class TestEveryOwnedIdRouteAnswers404:
    """The rule as a property of the whole API, not of the routes we remembered."""

    async def test_the_enumeration_matches_the_application(
        self, rolled_back_api: ApiHarness
    ) -> None:
        """A route added later must be covered here or this test fails.

        Without this, the 404-not-403 guarantee quietly becomes "the five
        routes somebody thought of in October".
        """
        declared = _declared_owned_id_operations(rolled_back_api.app)
        covered = {(method, path) for method, path, _body in OWNED_ID_OPERATIONS}
        assert covered == declared, (
            f"uncovered owned-id routes: {sorted(declared - covered)}; "
            f"routes that no longer exist: {sorted(covered - declared)}"
        )

    @pytest.mark.parametrize(
        ("method", "template", "body"),
        OWNED_ID_OPERATIONS,
        ids=[f"{method}-{path}" for method, path, _body in OWNED_ID_OPERATIONS],
    )
    async def test_anothers_id_is_not_found_and_never_forbidden(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        session: AsyncSession,
        user_a: str,
        user_b: str,
        method: str,
        template: str,
        body: dict[str, Any] | None,
    ) -> None:
        ids = {
            "run_id": str(owned.run_id),
            "plan_id": str(owned.plan_id),
            "item_id": str(owned.item_id),
        }
        path = template
        for name, value in ids.items():
            path = path.replace(f"{{{name}}}", value)
        assert "{" not in path, f"unsubstituted path parameter in {path}"

        before = await _snapshot(session, owned, user_a)
        response = await rolled_back_api.client.request(
            method, path, json=body, headers=rolled_back_api.dev(user_b)
        )

        assert response.status_code != 403, (
            f"{method} {template} answered 403, which confirms the id exists"
        )
        assert response.status_code == 404, f"{method} {template} answered {response.status_code}"
        assert response.json()["error"]["code"] == "not_found"
        for value in ids.values():
            assert value not in response.text, "the response echoed an id B does not own"
        assert await _snapshot(session, owned, user_a) == before, (
            f"{method} {template} changed A's data before refusing"
        )

    @pytest.mark.parametrize(
        ("method", "template", "body"),
        OWNED_ID_OPERATIONS,
        ids=[f"{method}-{path}" for method, path, _body in OWNED_ID_OPERATIONS],
    )
    async def test_an_id_nobody_owns_looks_exactly_the_same(
        self,
        rolled_back_api: ApiHarness,
        owned: OwnedData,
        user_b: str,
        method: str,
        template: str,
        body: dict[str, Any] | None,
    ) -> None:
        """The indistinguishability half: A's id and a random id agree."""
        unknown = str(uuid.uuid4())
        path = template
        for name in OWNED_ID_PARAMS:
            path = path.replace(f"{{{name}}}", unknown)

        response = await rolled_back_api.client.request(
            method, path, json=body, headers=rolled_back_api.dev(user_b)
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"
