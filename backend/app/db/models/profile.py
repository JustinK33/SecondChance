"""The ``profiles`` table: what we know about a person, versioned.

Two modelling decisions carried over from :mod:`app.schemas.profile`:

* **Income is a range, never a point.** ``income_min_cents`` /
  ``income_max_cents`` / ``income_period``, with a check constraint that an
  amount can never be stored without the period it refers to. A bare amount
  would eventually be read as monthly by somebody, and a wrong period turns a
  "we should ask" into a confident "do not bother applying".
* **Unknown is explicit.** ``unknown_fields`` records what the user was given
  the chance to answer and declined, so the clarification step never asks the
  same question twice.

Profiles are versioned rather than overwritten. The "edit and re-run" flow
inserts version N+1 and leaves N alone, which is what keeps a three-week-old
evaluation explainable against the profile it was actually made from instead of
appearing to have been made from the current one.

The table is user-owned: every row carries ``user_id``, row-level security
restricts it to ``auth.uid()``, and the account-deletion flow hard-deletes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, CheckConstraint, Index, Integer, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JsonDict, TimestampMixin, UuidPk
from app.db.models.columns import (
    Cents,
    EmptyJsonObject,
    EmptyTextArray,
    NullableEnumText,
    SmallCount,
    UserId,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.db.models.workflow import WorkflowRun

__all__ = ["MAX_AGE", "MAX_HOUSEHOLD_SIZE", "Profile"]

#: Kept in step with ``app.schemas.profile.MAX_HOUSEHOLD_SIZE`` / ``Age``.
MAX_HOUSEHOLD_SIZE = 20
MAX_AGE = 120


class Profile(TimestampMixin, Base):
    """One version of one user's profile snapshot."""

    __tablename__ = "profiles"

    id: Mapped[UuidPk]
    user_id: Mapped[UserId]
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
        doc="Monotonic per user, starting at 1. Insert a new version; never "
        "update an existing one.",
    )

    state: Mapped[NullableEnumText]
    county: Mapped[NullableEnumText]
    student_status: Mapped[NullableEnumText]
    household_size: Mapped[SmallCount]
    household_composition: Mapped[EmptyJsonObject]

    income_min_cents: Mapped[Cents]
    income_max_cents: Mapped[Cents]
    income_period: Mapped[NullableEnumText]

    employment_status: Mapped[NullableEnumText]
    housing_situation: Mapped[NullableEnumText]
    needs: Mapped[EmptyTextArray]
    age: Mapped[SmallCount]

    flags: Mapped[EmptyJsonObject]
    unknown_fields: Mapped[EmptyTextArray]
    changed_fields: Mapped[EmptyTextArray] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
        server_default=text("'{}'::text[]"),
        doc="Which fields this version changed relative to the previous one. "
        "Carries ProfileSnapshot.changed_fields, so an old evaluation can be "
        "explained in terms of what the user had told us at the time.",
    )
    extras: Mapped[JsonDict] = mapped_column(
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        doc="Non-eligibility context captured for wording only, e.g. a "
        "preferred language. Never read by the rule engine.",
    )

    runs: Mapped[list[WorkflowRun]] = relationship(
        back_populates="profile", lazy="raise_on_sql", viewonly=True
    )

    __table_args__ = (
        UniqueConstraint("user_id", "version"),
        Index("ix_profiles_user_id_created_at", "user_id", text("created_at DESC")),
        CheckConstraint("version >= 1", name="version_is_positive"),
        CheckConstraint(
            f"household_size IS NULL OR (household_size BETWEEN 1 AND {MAX_HOUSEHOLD_SIZE})",
            name="household_size_in_range",
        ),
        CheckConstraint(f"age IS NULL OR (age BETWEEN 0 AND {MAX_AGE})", name="age_in_range"),
        CheckConstraint(
            "income_min_cents IS NULL OR income_max_cents IS NULL "
            "OR income_min_cents <= income_max_cents",
            name="income_range_is_ordered",
        ),
        CheckConstraint(
            "(income_min_cents IS NULL AND income_max_cents IS NULL) OR income_period IS NOT NULL",
            name="income_amount_requires_a_period",
        ),
    )
