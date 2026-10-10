"""Audit events: what happened to a user's data, with nothing personal in it.

Every write goes through :meth:`AuditRepository.record`, and every metadata
payload is passed through :func:`app.core.logging.redact_event_dict` on the way
in. Reusing the log redactor rather than writing a second one is deliberate:
there is exactly one denylist of field names in this codebase, exactly one set
of value patterns for tokens, emails and money, and adding to it protects the
audit table and the logs at the same time. Two independent redactors would
inevitably drift, and the one that drifted would be the one nobody was looking
at.

Both identities are stored. ``user_id`` is the raw id, because row-level
security compares it to ``auth.uid()`` and because a user exercising a data
request has to be able to read their own audit rows and nobody else's.
``user_id_hash`` is :func:`app.core.logging.hash_user_id`, so an audit row and
a log line can be correlated without the log line ever carrying the raw id.

The table is append-only by convention: nothing here updates or deletes, and
the only deletion anywhere is the account-deletion flow, which removes a user's
audit rows along with the rest of their data.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger, hash_user_id, redact_event_dict
from app.db import models
from app.db.repositories.base import UserScopedRepository, normalise_page

__all__ = ["AuditAction", "AuditRepository"]

_log = get_logger(__name__)


class AuditAction(StrEnum):
    """The closed vocabulary of auditable actions.

    An enum rather than free text so that "what happened to my data" is a
    groupable, chartable question, and so no caller can write a sentence - and
    therefore personal content - into the ``action`` column.
    """

    PROFILE_SAVED = "profile.saved"
    RUN_CREATED = "run.created"
    RUN_RESUMED = "run.resumed"
    RUN_CANCELLED = "run.cancelled"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    MATCHES_PERSISTED = "matches.persisted"
    PLAN_CREATED = "plan.created"
    PLAN_VIEWED = "plan.viewed"
    ACTION_ITEM_UPDATED = "action_item.updated"
    ASSISTANT_ANSWERED = "assistant.answered"
    ACCOUNT_DATA_DELETED = "account.data_deleted"


class AuditRepository(UserScopedRepository):
    """Append audit events, and read back one user's own trail."""

    __slots__ = ("_settings",)

    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        super().__init__(session)
        self._settings = settings or get_settings()

    def _redact(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        """Redact a metadata payload with the same rules the logger uses."""
        if not metadata:
            return {}
        cleaned = redact_event_dict(
            dict(metadata), max_text_chars=self._settings.log_max_text_chars
        )
        return dict(cleaned)

    async def record(
        self,
        user_id: str,
        action: AuditAction,
        *,
        resource_type: str | None = None,
        resource_id: str | uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> models.AuditEvent:
        """Append one audit event. Never raises on a redactable payload.

        Does not commit: an audit event describes a change, so it belongs in
        the same transaction as the change it describes. An audit row that
        survived a rolled-back operation would be a record of something that
        did not happen.
        """
        row = models.AuditEvent(
            user_id=user_id,
            user_id_hash=hash_user_id(user_id),
            action=action.value,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id is not None else None,
            run_id=run_id,
            request_id=request_id,
            event_metadata=self._redact(metadata),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_for_user(
        self, user_id: str, *, limit: int = 50, offset: int = 0
    ) -> list[models.AuditEvent]:
        """One user's audit trail, newest first."""
        limit, offset = normalise_page(limit, offset)
        statement = (
            select(models.AuditEvent)
            .where(models.AuditEvent.user_id == user_id)
            .order_by(models.AuditEvent.created_at.desc(), models.AuditEvent.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.scalars(statement)).all())
