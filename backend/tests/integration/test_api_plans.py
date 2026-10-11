"""``/api/v1/plans`` and ``/api/v1/action-items`` - the output a user acts on.

The property that matters most here is the dullest-sounding one: **ticking a
step off has to stick, including across a regeneration**. An action item's id
is derived from the run and its deduplication key, so a new plan version for
the same run reuses the same rows, and the repository deliberately leaves
completion out of its conflict update. A generator has no opinion about
whether somebody has been to the housing office, and a plan that silently
un-ticks their errands is a plan they stop trusting.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import MatchRepository, PlanRepository, WorkflowRunRepository
from app.schemas import ActionItem, ActionItemKind, ActionPlan, Page, make_dedup_key
from tests.conftest import make_match, make_plan, make_program, make_ruleset
from tests.security.harness import ApiHarness, rolled_back_api, seed_catalog

__all__ = ["rolled_back_api"]

pytestmark = pytest.mark.db

NARRATIVE = "I need help with groceries and my electric bill this month, things are tight."


def error_of(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    assert set(body) == {"error"}, body
    error: dict[str, Any] = body["error"]
    return error


@pytest.fixture
async def planned(session: AsyncSession, user_a: str) -> dict[str, Any]:
    """A completed run for user A with one match and a two-step plan."""
    program = make_program()
    await seed_catalog(session, [program], [make_ruleset(program.slug)])
    runs = WorkflowRunRepository(session)
    row, _ = await runs.create(user_a, narrative=NARRATIVE)
    await MatchRepository(session).upsert_many(user_a, row.id, [make_match(program)])
    plan = make_plan(row.id, program)
    await PlanRepository(session).save(user_a, plan)
    await runs.mark_completed(user_a, row.id)
    await session.commit()
    return {"program": program, "run_id": row.id, "plan": plan}


class TestReading:
    async def test_the_list_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, planned: dict[str, Any], user_a: str
    ) -> None:
        response = await rolled_back_api.client.get(
            "/api/v1/plans", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        page = Page[ActionPlan].model_validate(response.json())
        assert page.total == 1
        assert page.items[0].id == planned["plan"].id
        assert len(page.items[0].items) == 2

    async def test_the_detail_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, planned: dict[str, Any], user_a: str
    ) -> None:
        response = await rolled_back_api.client.get(
            f"/api/v1/plans/{planned['plan'].id}", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        plan = ActionPlan.model_validate(response.json())
        assert plan.version == 1
        assert [item.sort_order for item in plan.items] == sorted(
            item.sort_order for item in plan.items
        )
        for item in plan.items:
            assert item.program_name, "a step must say which program it is for"
            assert item.dedup_key.startswith("document:")

    async def test_an_unknown_plan_is_not_found(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.get(
            f"/api/v1/plans/{uuid.uuid4()}", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 404
        assert error_of(response)["code"] == "not_found"

    async def test_a_malformed_plan_id_is_a_validation_error(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.get(
            "/api/v1/plans/nope", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 422
        assert error_of(response)["code"] == "validation_error"


class TestTickingAStep:
    async def test_ticking_and_unticking_both_work(
        self, rolled_back_api: ApiHarness, planned: dict[str, Any], user_a: str
    ) -> None:
        item_id = planned["plan"].items[0].id
        headers = rolled_back_api.dev(user_a)

        ticked = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{item_id}", json={"is_complete": True}, headers=headers
        )
        assert ticked.status_code == 200, ticked.text
        done = ActionItem.model_validate(ticked.json())
        assert done.is_complete is True
        assert done.completed_at is not None

        unticked = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{item_id}", json={"is_complete": False}, headers=headers
        )
        assert unticked.status_code == 200
        undone = ActionItem.model_validate(unticked.json())
        assert undone.is_complete is False
        assert undone.completed_at is None

    async def test_ticking_twice_leaves_the_same_state(
        self, rolled_back_api: ApiHarness, planned: dict[str, Any], user_a: str
    ) -> None:
        """A double tap - the normal consequence of a slow connection.

        The state is idempotent. ``completed_at`` is not: the repository
        rewrites it on every patch, so a second identical tick moves the
        timestamp. Nothing in the product reads it as "when they first did
        this", so it is recorded as a handoff note rather than asserted here.
        """
        item_id = planned["plan"].items[0].id
        headers = rolled_back_api.dev(user_a)
        first = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{item_id}", json={"is_complete": True}, headers=headers
        )
        second = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{item_id}", json={"is_complete": True}, headers=headers
        )
        assert first.status_code == second.status_code == 200
        assert first.json()["is_complete"] is second.json()["is_complete"] is True
        assert second.json()["completed_at"] is not None
        assert first.json()["id"] == second.json()["id"]

    async def test_only_completion_is_editable(
        self, rolled_back_api: ApiHarness, planned: dict[str, Any], user_a: str
    ) -> None:
        """A user must not be able to rewrite a step into a link elsewhere."""
        item = planned["plan"].items[0]
        response = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{item.id}",
            json={
                "is_complete": True,
                "url": "https://not-the-catalog.example.net/phishing",
                "title": "Send your bank details here",
            },
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422
        assert error_of(response)["code"] == "validation_error"
        assert "not-the-catalog" not in response.text

    @pytest.mark.parametrize("body", [{}, {"is_complete": "yes please"}, {"is_complete": None}])
    async def test_a_bad_body_is_a_validation_error(
        self,
        rolled_back_api: ApiHarness,
        planned: dict[str, Any],
        user_a: str,
        body: dict[str, Any],
    ) -> None:
        response = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{planned['plan'].items[0].id}",
            json=body,
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422
        assert "is_complete" in error_of(response)["detail"]["fields"]

    async def test_an_unknown_item_is_not_found(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.patch(
            f"/api/v1/action-items/{uuid.uuid4()}",
            json={"is_complete": True},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 404
        assert error_of(response)["code"] == "not_found"


class TestRegeneration:
    async def test_a_new_version_does_not_untick_a_completed_step(
        self,
        rolled_back_api: ApiHarness,
        session: AsyncSession,
        planned: dict[str, Any],
        user_a: str,
    ) -> None:
        """The whole reason action item ids are derived rather than random."""
        item_id = planned["plan"].items[0].id
        headers = rolled_back_api.dev(user_a)
        assert (
            await rolled_back_api.client.patch(
                f"/api/v1/action-items/{item_id}", json={"is_complete": True}, headers=headers
            )
        ).status_code == 200

        # Regenerate: same run, same deduplication keys, version 2.
        regenerated = make_plan(planned["run_id"], planned["program"], version=2)
        assert regenerated.items[0].id == item_id, "a regenerated step keeps its identity"
        await PlanRepository(session).save(user_a, regenerated)
        await session.commit()

        response = await rolled_back_api.client.get(
            f"/api/v1/plans/{regenerated.id}", headers=headers
        )
        assert response.status_code == 200, response.text
        plan = ActionPlan.model_validate(response.json())
        assert plan.version == 2
        kept = next(item for item in plan.items if item.id == item_id)
        assert kept.is_complete is True, "regeneration un-ticked a completed step"
        assert kept.completed_at is not None

    async def test_both_versions_remain_readable(
        self,
        rolled_back_api: ApiHarness,
        session: AsyncSession,
        planned: dict[str, Any],
        user_a: str,
    ) -> None:
        """ "What did you tell me last Tuesday" has to stay answerable."""
        regenerated = make_plan(planned["run_id"], planned["program"], version=2)
        await PlanRepository(session).save(user_a, regenerated)
        await session.commit()
        response = await rolled_back_api.client.get(
            "/api/v1/plans", headers=rolled_back_api.dev(user_a)
        )
        page = Page[ActionPlan].model_validate(response.json())
        assert page.total == 2
        assert {plan.version for plan in page.items} == {1, 2}
        assert {plan.run_id for plan in page.items} == {planned["run_id"]}


class TestSharedDocuments:
    async def test_a_document_two_programs_ask_for_is_listed_once_and_shared(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        """The point of the plan: one errand, not one errand per program."""
        from app.schemas import action_item_uuid, group_shared_documents

        grocery = make_program("example-grocery-help")
        utility = make_program("example-utility-help")
        await seed_catalog(
            session,
            [grocery, utility],
            [make_ruleset(grocery.slug), make_ruleset(utility.slug)],
        )
        runs = WorkflowRunRepository(session)
        row, _ = await runs.create(user_a, narrative=NARRATIVE)
        await MatchRepository(session).upsert_many(
            user_a, row.id, [make_match(grocery), make_match(utility)]
        )

        key = make_dedup_key(ActionItemKind.DOCUMENT, "proof of address")
        items = [
            ActionItem(
                id=action_item_uuid(row.id, key),
                program_id=program.id,
                program_name=program.official_name,
                kind=ActionItemKind.DOCUMENT,
                title="Find something that shows your address",
                detail="A recent utility bill, lease or piece of official mail works.",
                url=None,
                sort_order=index,
                dedup_key=key,
            )
            for index, program in enumerate((grocery, utility))
        ]
        shared = group_shared_documents(items)
        assert len(shared) == 1, "the grouping is what the plan stores"
        plan = make_plan(row.id, grocery).model_copy(
            update={"items": items[:1], "shared_documents": shared}
        )
        await PlanRepository(session).save(user_a, plan)
        await session.commit()

        response = await rolled_back_api.client.get(
            f"/api/v1/plans/{plan.id}", headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        stored = ActionPlan.model_validate(response.json())
        assert len(stored.items) == 1, "the collapsed step appears once"
        assert len(stored.shared_documents) == 1
        document = stored.shared_documents[0]
        assert document.dedup_key == key
        assert set(document.program_ids) == {grocery.id, utility.id}
        assert len(document.program_names) == 2
