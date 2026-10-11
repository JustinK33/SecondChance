"""``/api/v1/workflows`` and ``/api/v1/matches`` as an HTTP contract.

The assertions here are about the contract rather than about the workflow: the
runner is a recording stub, so what is under test is the 202-then-poll shape,
the idempotency guarantee, the pagination caps, the 404-not-403 rule and the
error envelope. The workflow itself is exercised by
``tests/integration/test_e2e_journey.py`` and ``test_workflow_*.py``.

Response shapes are checked by validating the JSON back through the declared
response model. Every one of those models sets ``extra="forbid"``, so a route
that added, renamed or dropped a field fails here rather than in a frontend
six weeks later.

The idempotency section is the one worth reading. ``POST /workflows`` promises
that a replayed key returns the original run and creates exactly one row, and
it has to hold when the two requests arrive at the same moment - which is the
common case, because a double tap on a slow connection is exactly how a user
produces one. That test needs two real connections, so it runs on the
committing harness and cleans up after itself.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models
from app.db.repositories import MatchRepository, ProfileRepository, WorkflowRunRepository
from app.schemas import (
    Page,
    WorkflowCreateResponse,
    WorkflowPhase,
    WorkflowRunDetail,
    WorkflowRunSummary,
    WorkflowStatus,
)
from tests.conftest import make_match, make_profile, make_program, make_ruleset
from tests.security.harness import (
    ApiHarness,
    committing_api,
    rolled_back_api,
    seed_catalog,
)

__all__ = ["committing_api", "rolled_back_api"]

pytestmark = pytest.mark.db

NARRATIVE = "I need help with groceries and my electric bill this month, things are tight."
OTHER_NARRATIVE = "I am about to lose my apartment and I do not know who to call about it."


def error_of(response) -> dict:
    body = response.json()
    assert set(body) == {"error"}, body
    return body["error"]


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------
class TestCreatingARun:
    async def test_the_happy_path_matches_the_declared_response_model(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 202, response.text
        created = WorkflowCreateResponse.model_validate(response.json())
        assert created.status is WorkflowStatus.QUEUED
        assert created.phase is WorkflowPhase.PENDING
        assert created.poll_url == f"/api/v1/workflows/{created.run_id}"
        assert created.deduplicated is False
        assert created.progress_message
        assert response.headers["Location"] == created.poll_url
        assert response.headers["X-Request-ID"]

    async def test_the_row_is_committed_before_the_task_is_scheduled(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        """Otherwise the id handed to the client could 404 on its first poll."""
        response = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        run_id = uuid.UUID(response.json()["run_id"])
        assert await WorkflowRunRepository(session).get(user_a, run_id) is not None
        assert [call["run_id"] for call in rolled_back_api.runner.scheduled] == [run_id]
        assert rolled_back_api.runner.scheduled[0]["user_id"] == user_a

    @pytest.mark.parametrize(
        ("body", "field"),
        [
            ({}, "narrative"),
            ({"narrative": "too short"}, "narrative"),
            ({"narrative": NARRATIVE, "idempotency_key": "short"}, "idempotency_key"),
            ({"narrative": NARRATIVE, "prefilled": {"state": "NEWJERSEY"}}, "prefilled.state"),
            (
                {"narrative": NARRATIVE, "prefilled": {"household_size": 0}},
                "prefilled.household_size",
            ),
            ({"narrative": NARRATIVE, "nonsense": 1}, "nonsense"),
        ],
    )
    async def test_validation_failures_use_the_standard_envelope(
        self, rolled_back_api: ApiHarness, user_a: str, body: dict, field: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/workflows", json=body, headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 422, response.text
        error = error_of(response)
        assert error["code"] == "validation_error"
        assert error["request_id"]
        assert field in error["detail"]["fields"], error["detail"]
        assert rolled_back_api.runner.scheduled == []

    async def test_an_income_period_is_never_guessed(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        """A ``ProfileDraft`` is a patch, so an amount with no period is a valid
        *request*: the period can come from the narrative or from a stored
        profile. What must never happen is one being invented, because an
        income compared against the wrong period's limit is a wrong answer
        delivered confidently.
        """
        response = await rolled_back_api.client.post(
            "/api/v1/intake",
            json={"narrative": NARRATIVE, "prefilled": {"income_min_cents": 200000}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 200, response.text
        profile = response.json()["profile"]
        assert profile["income_period"] == "MONTHLY", "the period came from the narrative"
        assert profile["income_min_cents"] == 200000, "what the user typed wins"
        assert profile["income_min_cents"] <= profile["income_max_cents"]

    async def test_a_typed_amount_that_contradicts_the_narrative_drops_the_income(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        """A minimum above the narrative's maximum cannot be reconciled.

        Taking either half would invent a range nobody stated, so the whole
        income is dropped and the field stays unknown - which keeps the
        programs that read it at ``MORE_INFO_NEEDED`` and produces a question
        rather than a wrong threshold.
        """
        response = await rolled_back_api.client.post(
            "/api/v1/intake",
            json={"narrative": NARRATIVE, "prefilled": {"income_min_cents": 9_000_00}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 200, response.text
        profile = response.json()["profile"]
        assert profile["income_min_cents"] is None
        assert profile["income_max_cents"] is None
        assert profile["income_period"] is None

    async def test_a_complete_typed_income_wins_over_the_narrative(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        """The counterpart: what the user typed beats what we inferred."""
        response = await rolled_back_api.client.post(
            "/api/v1/intake",
            json={
                "narrative": NARRATIVE,
                "prefilled": {
                    "income_min_cents": 250000,
                    "income_max_cents": 250000,
                    "income_period": "MONTHLY",
                },
            },
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 200, response.text
        profile = response.json()["profile"]
        assert profile["income_min_cents"] == 250000
        assert profile["income_max_cents"] == 250000

    async def test_an_inverted_income_range_is_refused(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={
                "narrative": NARRATIVE,
                "prefilled": {
                    "income_min_cents": 400000,
                    "income_max_cents": 100000,
                    "income_period": "MONTHLY",
                },
            },
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------
class TestIdempotency:
    KEY = "idempotency-key-for-the-tests"

    async def test_the_same_key_twice_returns_the_same_run(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        headers = {**rolled_back_api.dev(user_a), "Idempotency-Key": self.KEY}
        first = await rolled_back_api.client.post(
            "/api/v1/workflows", json={"narrative": NARRATIVE}, headers=headers
        )
        second = await rolled_back_api.client.post(
            "/api/v1/workflows", json={"narrative": NARRATIVE}, headers=headers
        )
        assert first.status_code == 202
        assert second.status_code == 200, "a replay accepted nothing, so it is not a 202"
        assert first.json()["run_id"] == second.json()["run_id"]
        assert first.json()["deduplicated"] is False
        assert second.json()["deduplicated"] is True

        rows = await session.scalar(
            select(func.count())
            .select_from(models.WorkflowRun)
            .where(models.WorkflowRun.user_id == user_a)
        )
        assert rows == 1, f"the replay created a second run ({rows} rows)"
        assert len(rolled_back_api.runner.scheduled) == 1, "the replay scheduled a second run"

    async def test_the_key_can_come_from_the_body_instead_of_the_header(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        headers = rolled_back_api.dev(user_a)
        body = {"narrative": NARRATIVE, "idempotency_key": self.KEY}
        first = await rolled_back_api.client.post("/api/v1/workflows", json=body, headers=headers)
        second = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers={**headers, "Idempotency-Key": self.KEY},
        )
        assert first.status_code == 202
        assert second.status_code == 200
        assert first.json()["run_id"] == second.json()["run_id"]

    async def test_reusing_a_key_for_a_different_narrative_is_a_conflict(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        """Returning the first run would answer a question nobody asked."""
        headers = {**rolled_back_api.dev(user_a), "Idempotency-Key": self.KEY}
        first = await rolled_back_api.client.post(
            "/api/v1/workflows", json={"narrative": NARRATIVE}, headers=headers
        )
        assert first.status_code == 202
        clash = await rolled_back_api.client.post(
            "/api/v1/workflows", json={"narrative": OTHER_NARRATIVE}, headers=headers
        )
        assert clash.status_code == 409
        error = error_of(clash)
        assert error["code"] == "conflict"
        assert error["detail"] == {"field": "idempotency_key"}
        assert OTHER_NARRATIVE not in clash.text, "the narrative must not be echoed back"

    async def test_the_same_key_for_two_users_is_two_runs(
        self, rolled_back_api: ApiHarness, user_a: str, user_b: str
    ) -> None:
        """The uniqueness is per user; a shared key space would be a leak."""
        mine = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers={**rolled_back_api.dev(user_a), "Idempotency-Key": self.KEY},
        )
        theirs = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers={**rolled_back_api.dev(user_b), "Idempotency-Key": self.KEY},
        )
        assert mine.status_code == theirs.status_code == 202
        assert mine.json()["run_id"] != theirs.json()["run_id"]

    async def test_two_concurrent_requests_with_one_key_create_one_run(
        self, committing_api: ApiHarness, user_a: str
    ) -> None:
        """The race, run for real on two connections.

        A read-then-write in Python would let both requests see "no existing
        run" and both insert. The guarantee comes from
        ``UNIQUE (user_id, idempotency_key)`` plus ``ON CONFLICT DO NOTHING``,
        and this is the only test that can tell the difference.
        """
        headers = {**committing_api.dev(user_a), "Idempotency-Key": "concurrent-double-tap"}
        try:
            responses = await asyncio.gather(
                *(
                    committing_api.client.post(
                        "/api/v1/workflows", json={"narrative": NARRATIVE}, headers=headers
                    )
                    for _ in range(4)
                )
            )
            assert {response.status_code for response in responses} <= {200, 202}
            run_ids = {response.json()["run_id"] for response in responses}
            assert len(run_ids) == 1, f"the race produced {len(run_ids)} runs: {run_ids}"
            accepted = [r for r in responses if r.status_code == 202]
            assert len(accepted) == 1, "more than one request believed it had created the run"
            assert len(committing_api.runner.scheduled) == 1, "the graph was scheduled twice"

            assert committing_api.session_factory is not None
            async with committing_api.session_factory() as session:
                rows = await session.scalar(
                    select(func.count())
                    .select_from(models.WorkflowRun)
                    .where(models.WorkflowRun.user_id == user_a)
                )
            assert rows == 1, f"{rows} workflow_runs rows for one key"
        finally:
            await committing_api.cleanup_users([user_a])


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
class TestReadingRuns:
    async def test_the_detail_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        created = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        run_id = created.json()["run_id"]
        response = await rolled_back_api.client.get(
            f"/api/v1/workflows/{run_id}", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        detail = WorkflowRunDetail.model_validate(response.json())
        assert str(detail.id) == run_id
        assert detail.matches == []
        assert detail.plan is None
        assert detail.error is None
        assert 0.0 <= detail.progress_fraction <= 1.0
        assert detail.progress_message

    async def test_an_unknown_run_is_not_found(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        missing = uuid.uuid4()
        response = await rolled_back_api.client.get(
            f"/api/v1/workflows/{missing}", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 404
        assert error_of(response)["code"] == "not_found"

    async def test_a_malformed_run_id_is_a_validation_error(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.get(
            "/api/v1/workflows/not-a-uuid", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 422
        assert error_of(response)["code"] == "validation_error"

    async def test_the_list_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        for index in range(3):
            response = await rolled_back_api.client.post(
                "/api/v1/workflows",
                json={"narrative": f"{NARRATIVE} Attempt number {index}."},
                headers=rolled_back_api.dev(user_a),
            )
            assert response.status_code == 202
        response = await rolled_back_api.client.get(
            "/api/v1/workflows", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200
        page = Page[WorkflowRunSummary].model_validate(response.json())
        assert page.total == 3
        assert len(page.items) == 3
        assert page.limit == 20
        assert page.offset == 0
        created = [item.created_at for item in page.items]
        assert created == sorted(created, reverse=True), "newest first"


class TestPagination:
    @pytest.mark.parametrize("path", ["/api/v1/workflows", "/api/v1/plans"])
    @pytest.mark.parametrize(
        "query", ["limit=0", "limit=201", "limit=-1", "offset=-1", "limit=abc"]
    )
    async def test_an_out_of_range_window_is_refused(
        self, rolled_back_api: ApiHarness, user_a: str, path: str, query: str
    ) -> None:
        response = await rolled_back_api.client.get(
            f"{path}?{query}", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 422, f"{path}?{query} -> {response.status_code}"
        assert error_of(response)["code"] == "validation_error"

    @pytest.mark.parametrize("path", ["/api/v1/workflows", "/api/v1/plans"])
    async def test_the_cap_is_the_documented_number(
        self, rolled_back_api: ApiHarness, user_a: str, path: str
    ) -> None:
        ok = await rolled_back_api.client.get(
            f"{path}?limit=200", headers=rolled_back_api.dev(user_a)
        )
        assert ok.status_code == 200
        assert ok.json()["limit"] == 200

    async def test_the_window_is_applied(self, rolled_back_api: ApiHarness, user_a: str) -> None:
        for index in range(3):
            await rolled_back_api.client.post(
                "/api/v1/workflows",
                json={"narrative": f"{NARRATIVE} Attempt number {index}."},
                headers=rolled_back_api.dev(user_a),
            )
        first = await rolled_back_api.client.get(
            "/api/v1/workflows?limit=2&offset=0", headers=rolled_back_api.dev(user_a)
        )
        second = await rolled_back_api.client.get(
            "/api/v1/workflows?limit=2&offset=2", headers=rolled_back_api.dev(user_a)
        )
        assert [len(first.json()["items"]), len(second.json()["items"])] == [2, 1]
        assert first.json()["total"] == second.json()["total"] == 3
        overlap = {item["id"] for item in first.json()["items"]} & {
            item["id"] for item in second.json()["items"]
        }
        assert overlap == set(), "the pages overlap, so a row can be shown twice"


# ---------------------------------------------------------------------------
# resume and cancel
# ---------------------------------------------------------------------------
class TestResumeAndCancel:
    async def test_resuming_a_run_that_is_not_waiting_is_a_conflict(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        created = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{created.json()['run_id']}/resume",
            json={"answers": []},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 409, response.text
        error = error_of(response)
        assert error["code"] == "conflict"
        assert error["detail"]["status"] == WorkflowStatus.QUEUED.value
        assert rolled_back_api.runner.resumed == []

    async def test_resuming_an_unknown_run_is_not_found(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{uuid.uuid4()}/resume",
            json={"answers": []},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 404

    async def test_answering_one_question_twice_is_refused(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        answer = {"question_id": "q-1", "field": "state", "value": "NJ"}
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{uuid.uuid4()}/resume",
            json={"answers": [answer, answer]},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422
        assert error_of(response)["code"] == "validation_error"

    async def test_more_answers_than_questions_were_ever_asked_is_refused(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        answers = [
            {"question_id": f"q-{index}", "field": "state", "value": "NJ"} for index in range(11)
        ]
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{uuid.uuid4()}/resume",
            json={"answers": answers},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422

    async def test_cancelling_a_queued_run_records_it_and_cancels_the_task(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        created = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        run_id = uuid.UUID(created.json()["run_id"])
        response = await rolled_back_api.client.post(
            f"/api/v1/workflows/{run_id}/cancel", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        detail = WorkflowRunDetail.model_validate(response.json())
        assert detail.status is WorkflowStatus.CANCELLED
        assert detail.completed_at is not None
        assert rolled_back_api.runner.cancelled == [run_id]

    async def test_cancelling_a_finished_run_is_a_conflict(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        created = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        run_id = created.json()["run_id"]
        headers = rolled_back_api.dev(user_a)
        assert (
            await rolled_back_api.client.post(f"/api/v1/workflows/{run_id}/cancel", headers=headers)
        ).status_code == 200
        again = await rolled_back_api.client.post(
            f"/api/v1/workflows/{run_id}/cancel", headers=headers
        )
        assert again.status_code == 409
        assert error_of(again)["code"] == "conflict"


# ---------------------------------------------------------------------------
# re-evaluation
# ---------------------------------------------------------------------------
class TestReevaluation:
    async def test_an_unknown_source_run_is_not_found(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(uuid.uuid4()), "changes": {"household_size": 4}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 404
        assert error_of(response)["code"] == "not_found"

    async def test_an_empty_change_set_is_refused(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(uuid.uuid4()), "changes": {}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422
        assert error_of(response)["code"] == "validation_error"

    async def test_a_run_with_no_results_yet_is_a_conflict(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        """Honest: finish that check first rather than hand back an empty one."""
        runs = WorkflowRunRepository(session)
        row, _ = await runs.create(user_a, narrative=NARRATIVE)
        profile_row = await ProfileRepository(session).save_new_version(
            user_a, make_profile(), changed_fields=["state"]
        )
        await runs.attach_profile(user_a, row.id, profile_row.id)
        await session.commit()

        response = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(row.id), "changes": {"household_size": 4}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 409
        assert error_of(response)["code"] == "conflict"

    async def test_a_change_that_changes_nothing_is_a_conflict(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        program = make_program()
        await seed_catalog(session, [program], [make_ruleset(program.slug)])
        runs = WorkflowRunRepository(session)
        row, _ = await runs.create(user_a, narrative=NARRATIVE)
        profile = make_profile()
        profile_row = await ProfileRepository(session).save_new_version(
            user_a, profile, changed_fields=["state"]
        )
        await runs.attach_profile(user_a, row.id, profile_row.id)
        await MatchRepository(session).upsert_many(user_a, row.id, [make_match(program)])
        await session.commit()

        response = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(row.id), "changes": {"household_size": profile.household_size}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 409
        assert error_of(response)["detail"] == {"field": "changes"}

    async def test_a_real_edit_completes_synchronously_with_no_model_call(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        program = make_program()
        await seed_catalog(session, [program], [make_ruleset(program.slug)])
        runs = WorkflowRunRepository(session)
        row, _ = await runs.create(user_a, narrative=NARRATIVE)
        profile_row = await ProfileRepository(session).save_new_version(
            user_a, make_profile(), changed_fields=["state"]
        )
        await runs.attach_profile(user_a, row.id, profile_row.id)
        await MatchRepository(session).upsert_many(user_a, row.id, [make_match(program)])
        await session.commit()

        before = len(rolled_back_api.gemini.calls)
        response = await rolled_back_api.client.post(
            "/api/v1/matches",
            json={"run_id": str(row.id), "changes": {"household_size": 6}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 202, response.text
        created = WorkflowCreateResponse.model_validate(response.json())
        assert created.status is WorkflowStatus.COMPLETED
        assert len(rolled_back_api.gemini.calls) == before, "a model was called"
        assert rolled_back_api.runner.scheduled == [], "nothing should be scheduled"
        assert response.headers["Location"] == created.poll_url
