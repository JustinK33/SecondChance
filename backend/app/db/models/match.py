"""The ``program_matches`` table: one persisted verdict per program per run.

This is where the deterministic engine's output is stored, and the column
layout keeps the trust boundary visible:

* ``status``, ``outcome``, ``evidence`` and ``missing_fields`` come from the
  rule engine and from nowhere else.
* ``explanation`` is model-written prose. It is a separate JSONB column with no
  bearing on ``status``, so there is no write path by which language generation
  could move a verdict.
* ``rule_version`` and ``rule_version_id`` pin the exact rule set that produced
  the row. A match stored today stays explainable after the program's rules are
  superseded, because the version it was judged under is still readable.

``score`` is a ranking aid in ``[0, 1]``. It is not a probability of
eligibility and never affects ``status``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UuidPk
from app.db.models.columns import (
    ConfidenceNumeric,
    EmptyJsonList,
    EmptyTextArray,
    EnumText,
    UserId,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.db.models.program import AssistanceProgram
    from app.db.models.workflow import WorkflowRun

__all__ = ["ProgramMatch"]


class ProgramMatch(TimestampMixin, Base):
    """One program as presented to one user, with its verdict and reasoning."""

    __tablename__ = "program_matches"

    id: Mapped[UuidPk]
    user_id: Mapped[UserId]
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False
    )
    program_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assistance_programs.id", ondelete="RESTRICT"),
        nullable=False,
        doc="RESTRICT, not CASCADE: a catalog program is soft-deleted, never "
        "removed, so a saved result can always still name the program it "
        "was about.",
    )
    rule_version: Mapped[str] = mapped_column(
        nullable=False,
        doc='The version string the engine judged under, e.g. "2026.1". Never "latest".',
    )
    rule_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("program_rule_versions.id", ondelete="SET NULL"),
        nullable=True,
        doc="Direct reference to the rule row, for auditing. Nullable because "
        "rule sets are append-only but may predate this column.",
    )

    status: Mapped[EnumText]
    outcome: Mapped[EnumText]
    score: Mapped[ConfidenceNumeric]
    missing_fields: Mapped[EmptyTextArray]
    evidence: Mapped[EmptyJsonList]
    explanation: Mapped[dict[str, Any] | None] = mapped_column(
        nullable=True,
        doc="A serialised MatchExplanation, or NULL while explanations are "
        "still being generated or if generation failed. Prose only.",
    )
    limitations: Mapped[EmptyTextArray]
    evaluated_at: Mapped[dt.datetime] = mapped_column(nullable=False)

    run: Mapped[WorkflowRun] = relationship(back_populates="matches", lazy="raise_on_sql")
    program: Mapped[AssistanceProgram] = relationship(
        back_populates="matches", lazy="raise_on_sql", viewonly=True
    )

    __table_args__ = (
        UniqueConstraint("run_id", "program_id"),
        Index("ix_program_matches_user_id_status", "user_id", "status"),
        Index("ix_program_matches_run_id_status", "run_id", "status"),
        Index("ix_program_matches_program_id", "program_id"),
        CheckConstraint("score >= 0 AND score <= 1", name="score_in_unit_interval"),
        CheckConstraint(
            "status <> 'MORE_INFO_NEEDED' OR coalesce(array_length(missing_fields, 1), 0) >= 1",
            name="more_info_needed_names_a_missing_field",
        ),
    )
