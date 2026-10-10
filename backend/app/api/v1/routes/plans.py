"""Action plans and the steps inside them.

A plan is the output a user actually acts on, so two properties matter more
than anything else here.

**Ownership is enforced on every call**, in SQL rather than in Python, and a
miss is a 404. A plan or a step belonging to somebody else is indistinguishable
from one that does not exist.

**Ticking a step is durable and never undone by us.** An item's identity is
derived from the run and its deduplication key, so a regenerated plan reuses
the same row and the repository deliberately leaves completion out of its
conflict update. A regeneration cannot un-tick something a person has done.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Path

from app.api.errors import error_responses
from app.api.v1.deps import CurrentUser, DbSession, Pagination, RequestId
from app.schemas import ActionItem, ActionItemPatch, ActionPlan, Page
from app.services import plan_service

__all__ = ["router"]

router = APIRouter(tags=["plans"])


@router.get(
    "/plans",
    response_model=Page[ActionPlan],
    summary="List your action plans",
    description="Your plans, newest first, each with its steps and shared documents.",
    responses=error_responses(401, 422),
)
async def list_plans(user: CurrentUser, session: DbSession, page: Pagination) -> Page[ActionPlan]:
    return await plan_service.list_plans(
        session, user.user_id, limit=page.limit, offset=page.offset
    )


@router.get(
    "/plans/{plan_id}",
    response_model=ActionPlan,
    summary="One action plan",
    description=(
        "A plan's steps in display order, plus the documents more than one "
        "program asks for - gathered once rather than listed per program."
    ),
    responses=error_responses(401, 404),
)
async def get_plan(
    user: CurrentUser,
    session: DbSession,
    plan_id: Annotated[uuid.UUID, Path(description="The plan id.")],
) -> ActionPlan:
    return await plan_service.get_plan(session, user.user_id, plan_id)


@router.patch(
    "/action-items/{item_id}",
    response_model=ActionItem,
    summary="Tick or untick one step",
    description=(
        "Marks a step complete or incomplete. Ownership is enforced: a step "
        "belonging to another user is a 404.\n\n"
        "Completion survives a plan being regenerated, because the generator "
        "has no opinion about whether a person has done something."
    ),
    responses=error_responses(401, 404, 422),
)
async def patch_action_item(
    body: ActionItemPatch,
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    item_id: Annotated[uuid.UUID, Path(description="The action item id.")],
) -> ActionItem:
    return await plan_service.set_item_complete(
        session, user.user_id, item_id, body, request_id=request_id
    )
