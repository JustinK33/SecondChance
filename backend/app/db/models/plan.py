"""The ``action_plans`` and ``action_items`` tables: the product's real output.

A user told they might qualify for five programs does not need five
instruction sets that each begin "gather proof of address". They need one list
of documents and five applications. That is what these two tables, plus the
``dedup_key``, exist to produce.

**Plans are versioned, not mutated.** ``UNIQUE (run_id, version)`` means
regenerating a plan writes version N+1. A step the user already ticked off is
never silently rewritten underneath them, and "what changed when I corrected my
income" is answerable.

**Items deduplicate on ``dedup_key``.** The key is
``"{kind}:{canonical_subject}"``, built server-side by
:func:`app.schemas.plan.make_dedup_key` from the canonical requirement rather
than from the model-written title. ``UNIQUE (plan_id, dedup_key)`` makes that
collapse a database guarantee instead of a hope about the generator's
consistency.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UuidPk
from app.db.models.columns import EmptyJsonList, EnumText, UserId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.db.models.workflow import WorkflowRun

__all__ = ["ActionItem", "ActionPlan"]


class ActionPlan(TimestampMixin, Base):
    """One version of the action plan for one run."""

    __tablename__ = "action_plans"

    id: Mapped[UuidPk]
    user_id: Mapped[UserId]
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    shared_documents: Mapped[EmptyJsonList]

    run: Mapped[WorkflowRun] = relationship(back_populates="plans", lazy="raise_on_sql")
    items: Mapped[list[ActionItem]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
        order_by="ActionItem.sort_order",
    )

    __table_args__ = (
        # ``(run_id, version)`` rather than ``(user_id, run_id, version)``: a
        # run belongs to exactly one user, so the two-column constraint already
        # rejects every duplicate the three-column one would, and a second
        # overlapping unique index would cost a write on every insert for no
        # additional guarantee. User-scoped reads use the index below.
        UniqueConstraint("run_id", "version"),
        Index("ix_action_plans_user_id_created_at", "user_id", text("created_at DESC")),
        CheckConstraint("version >= 1", name="version_is_positive"),
    )


class ActionItem(TimestampMixin, Base):
    """One concrete step in a plan.

    The primary key is ``uuid5(run_id, dedup_key)`` (see
    :func:`app.schemas.common.action_item_uuid`), so regenerating a plan for the
    same run yields the same item ids. That is what lets the frontend keep
    completion state across a regeneration and what makes plan creation
    idempotent under a retried request.
    """

    __tablename__ = "action_items"

    id: Mapped[UuidPk]
    plan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("action_plans.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[UserId]
    program_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("assistance_programs.id", ondelete="SET NULL"),
        nullable=True,
        doc="Nullable so a shared, program-independent step can exist. In "
        "practice every generated item names a program.",
    )
    program_name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Denormalised for display so a plan reads correctly even after a "
        "program is retired. Never used as a link target.",
    )
    kind: Mapped[EnumText]
    title: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Validated against the catalog before persistence. A URL the model "
        "invented is dropped, never stored and never followed.",
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    is_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    completed_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    dedup_key: Mapped[str] = mapped_column(Text, nullable=False)

    plan: Mapped[ActionPlan] = relationship(back_populates="items", lazy="raise_on_sql")

    __table_args__ = (
        UniqueConstraint("plan_id", "dedup_key"),
        Index("ix_action_items_plan_id_sort_order", "plan_id", "sort_order"),
        Index("ix_action_items_user_id_is_complete", "user_id", "is_complete"),
        CheckConstraint("sort_order >= 0", name="sort_order_is_non_negative"),
        CheckConstraint(
            "(completed_at IS NOT NULL) = is_complete",
            name="completion_timestamp_matches_flag",
        ),
    )
