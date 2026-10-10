"""``DELETE /api/v1/account/data`` - erase everything we hold about a user.

Hard deletes, in one transaction, with a receipt. The receipt matters: a
privacy promise nobody can check is a marketing claim, so the user gets the
per-table counts of what was actually removed and a plain-language statement of
what deliberately survives - aggregate model-usage rows whose reference to them
has been severed, and one audit row recording that the deletion happened.

The LangGraph checkpoint tables are included. They hold the narrative and the
extracted profile for every run, they have no ``user_id`` column of their own,
and the only link from a user to those rows is the run's ``thread_id``. A
deletion that skipped them would leave the most personal content we store.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.errors import error_responses
from app.api.v1.deps import CurrentUser, DbSession
from app.schemas import AccountDeletionReceipt
from app.services import account_service

__all__ = ["router"]

router = APIRouter(tags=["account"])


@router.delete(
    "/account/data",
    response_model=AccountDeletionReceipt,
    status_code=status.HTTP_200_OK,
    summary="Delete all of your data",
    description=(
        "Permanently removes every row belonging to the authenticated user: "
        "profiles, runs, matches, plans, steps, audit events and the workflow "
        "checkpoints that hold what they wrote.\n\n"
        "One transaction, so either all of it is gone or none of it is. The "
        "response is a receipt with the row counts that were deleted and a "
        "statement of what was kept and why."
    ),
    responses=error_responses(401),
)
async def delete_account_data(user: CurrentUser, session: DbSession) -> AccountDeletionReceipt:
    return await account_service.delete_user_data(session, user.user_id)
