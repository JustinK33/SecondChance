"""Append-only operational tables: ``audit_events`` and ``model_calls``.

Both exist so that privacy and cost claims can be checked rather than
asserted, and both are deliberately free of personal content.

``audit_events`` records *that* something happened to a user's data - a run was
created, a plan was read, an account was deleted - with the ids needed to
reconstruct the sequence. It stores two identities on purpose:

* ``user_id`` - the raw id, because row-level security compares it to
  ``auth.uid()`` and because a user asking "what did you do with my data"
  must be able to get their own audit rows and nobody else's.
* ``user_id_hash`` - :func:`app.core.logging.hash_user_id`, so an audit row can
  be correlated with a log line without the log line ever carrying the raw id.

``metadata`` is for redacted, structured, non-personal detail: counts, enum
values, durations, error codes. Never a narrative, an income figure, an email
or a token. Nothing in this table is written from user-supplied text.

Append-only is a convention, not a trigger: the repositories only ever insert,
and the only deletion is the account-deletion flow, which removes a user's
audit rows along with everything else - a privacy promise has to be real even
when it is inconvenient for the audit trail.

``model_calls`` is token and latency telemetry per model call. It contains no
user content at all, which is why it is the one table safe to report on
freely, and its ``run_id`` is ``ON DELETE SET NULL`` so aggregate cost history
survives a run being deleted.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UuidPk
from app.db.models.columns import EnumText, UserId

__all__ = ["AuditEvent", "ModelCall"]


class AuditEvent(Base):
    """One recorded action against one user's data. Insert only."""

    __tablename__ = "audit_events"

    id: Mapped[UuidPk]
    user_id: Mapped[UserId]
    user_id_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc='The "u_<16 hex>" identity from app.core.logging.hash_user_id.',
    )
    action: Mapped[EnumText]
    resource_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    resource_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        nullable=True,
        doc="Deliberately not a foreign key: an audit row must outlive the run "
        "it describes, including the run's deletion.",
    )
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Mapped to the column name ``metadata`` but exposed in Python as
    #: ``event_metadata``: ``metadata`` is taken on a declarative class by
    #: ``Base.metadata`` and shadowing it would break the whole ORM.
    event_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        doc="Redacted, structured, non-personal detail only.",
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_audit_events_user_id_created_at", "user_id", text("created_at DESC")),
        Index("ix_audit_events_action", "action"),
    )


class ModelCall(Base):
    """Cost and latency of one logical model call, including its retries."""

    __tablename__ = "model_calls"

    id: Mapped[UuidPk]
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="SET NULL"), nullable=True
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[EnumText]

    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    thought_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, server_default=text("0")
    )

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_model_calls_run_id", "run_id"),
        Index("ix_model_calls_created_at", text("created_at DESC")),
        CheckConstraint("attempts >= 1", name="attempts_is_positive"),
        CheckConstraint(
            "prompt_tokens >= 0 AND output_tokens >= 0 AND thought_tokens >= 0 "
            "AND total_tokens >= 0 AND latency_ms >= 0 AND estimated_cost_usd >= 0",
            name="telemetry_is_non_negative",
        ),
    )
