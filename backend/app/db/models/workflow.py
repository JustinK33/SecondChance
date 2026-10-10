"""The ``workflow_runs`` table: one durable row per analysis a user asked for.

A model-backed workflow takes tens of seconds, so no HTTP request is held open
for one. ``POST /api/v1/workflows`` writes a row here and returns its id; the
client polls. That makes this table the run's only durable identity and the
place status, phase and failure live.

**The narrative is not stored here.** Only ``narrative_hash``. The free-text
description of someone's situation is the most sensitive thing this product
handles, and per ``docs/INTERFACES.md`` section 8 it stays inside the LangGraph
checkpoint row for the run and nowhere else. The hash exists so that a retried
submission can be recognised as the same request without the text being
readable in a table that reporting, support tooling and future joins will all
touch.

``idempotency_key`` is unique per user. A double-tapped submit button or a
retried request therefore cannot produce two runs and two plans. The constraint
is a plain ``UNIQUE``, which in Postgres treats NULLs as distinct - exactly the
wanted behaviour, since the key is optional.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UuidPk
from app.db.models.columns import EmptyJsonList, EmptyTextArray, EnumText, UserId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.db.models.match import ProgramMatch
    from app.db.models.plan import ActionPlan
    from app.db.models.profile import Profile

__all__ = ["WorkflowRun"]


class WorkflowRun(TimestampMixin, Base):
    """One workflow execution: its state, its inputs by reference, its result."""

    __tablename__ = "workflow_runs"

    id: Mapped[UuidPk]
    user_id: Mapped[UserId]
    status: Mapped[EnumText]
    phase: Mapped[EnumText]

    idempotency_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="LangGraph checkpoint thread id, always ``str(run_id)``. Stored "
        "explicitly so the account-deletion flow can find a user's "
        "checkpoint rows without reconstructing the convention, and so a "
        "future change to it does not orphan old threads.",
    )
    narrative_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="SHA-256 of the submitted narrative. Never the narrative itself.",
    )
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("profiles.id", ondelete="SET NULL"),
        nullable=True,
        doc="The profile version this run was evaluated against.",
    )

    clarification_payload: Mapped[dict[str, Any] | None] = mapped_column(
        nullable=True,
        doc="The open clarification question set while status is "
        "AWAITING_INPUT, so a poll can render the questions without "
        "reading the checkpoint. Cleared when the run resumes.",
    )
    limitations: Mapped[EmptyTextArray]
    error: Mapped[dict[str, Any] | None] = mapped_column(
        nullable=True,
        doc="A serialised UserFacingError. Never a driver message or a stack "
        "frame; see app.core.errors.",
    )
    model_usage: Mapped[EmptyJsonList]

    started_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)

    profile: Mapped[Profile | None] = relationship(back_populates="runs", lazy="raise_on_sql")
    matches: Mapped[list[ProgramMatch]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise_on_sql",
    )
    plans: Mapped[list[ActionPlan]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise_on_sql",
    )

    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key"),
        UniqueConstraint("thread_id"),
        Index("ix_workflow_runs_user_id_created_at", "user_id", text("created_at DESC")),
        Index("ix_workflow_runs_status", "status"),
        CheckConstraint("narrative_hash ~ '^[0-9a-f]{64}$'", name="narrative_hash_is_sha256_hex"),
    )
