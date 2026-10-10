"""Account data deletion.

One function, one transaction, and a receipt the user can read. The repository
does the deleting; this module owns the transaction boundary, which is what
makes the promise real: either every row is gone or none of them are, and there
is no intermediate state in which a user's plans have been removed but their
narratives have not.

Deletion is hard, not soft. ``SoftDeleteMixin`` exists for the catalog, where
"we stopped publishing this program" is useful history. A privacy promise that
resolves to ``deleted_at IS NOT NULL`` is not a privacy promise.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger, hash_user_id
from app.db.repositories import AccountDataRepository
from app.schemas import AccountDeletionReceipt

__all__ = ["delete_user_data"]

_log = get_logger(__name__)


async def delete_user_data(session: AsyncSession, user_id: str) -> AccountDeletionReceipt:
    """Delete everything ``user_id`` owns and return the receipt.

    Commits, because the caller is an HTTP handler and a deletion that is
    reported as done must actually be done before the response is written.
    """
    receipt = await AccountDataRepository(session).delete_user_data(user_id)
    await session.commit()
    _log.info(
        "account.deleted",
        user=hash_user_id(user_id),
        row_count=sum(receipt.deleted_counts.values()),
    )
    return receipt
