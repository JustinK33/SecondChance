"""Catalog tables: programs, the sources that justify them, and their rules.

These three tables are the product's evidence base, and their shape encodes the
rule that nothing is ever fabricated:

* ``assistance_programs.verification_status`` and ``last_verified_at`` record
  whether a human has checked the record. Only a ``VERIFIED`` row may support
  an eligibility claim.
* ``program_sources`` is the receipt. Every row carries the URL, the quoted
  excerpt, when it was fetched and a hash of what was fetched, so "last
  verified" is a fact about a retrieval rather than a claim.
* ``program_rule_versions`` is append-only. Superseding a rule set writes a new
  version and closes the old one with ``effective_to``, which is what keeps an
  evaluation recorded last month explainable against the rules that were
  actually in force then.

The catalog is public-read, so it is the only part of the schema not scoped to
a user. It uses :class:`~app.db.base.SoftDeleteMixin` instead of hard deletes:
a retired program must stop appearing in search results while staying
resolvable from an action plan a user saved last month.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import TYPE_CHECKING, Final

from sqlalchemy import (
    CheckConstraint,
    Computed,
    ForeignKey,
    Index,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JsonDict, SoftDeleteMixin, TimestampMixin, UuidPk
from app.db.models.columns import (
    EmptyTextArray,
    EnumText,
    OptionalDate,
    RequiredDate,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.db.models.match import ProgramMatch

__all__ = [
    "SEARCH_VECTOR_EXPRESSION",
    "TEXT_ARRAY_JOIN_FUNCTION_DDL",
    "AssistanceProgram",
    "ProgramRuleVersion",
    "ProgramSource",
]

#: An ``IMMUTABLE`` wrapper around ``array_to_string``.
#:
#: Postgres marks ``array_to_string`` ``STABLE`` - not because joining a
#: ``text[]`` is non-deterministic, but because the generic signature has to go
#: through the element type's output function, which for an arbitrary type may
#: depend on settings. A generated column may only call ``IMMUTABLE``
#: functions, so the ``text[]`` case is wrapped and labelled honestly here.
#: ``array_to_string(text[], ' ')`` genuinely is immutable.
#:
#: Created by the first migration, before ``assistance_programs``, because the
#: generated column below depends on it.
TEXT_ARRAY_JOIN_FUNCTION_DDL: Final[str] = """
CREATE OR REPLACE FUNCTION sc_text_array_to_text(value text[])
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
STRICT
AS $$ SELECT array_to_string(value, ' ') $$
"""

#: The ``search_vector`` generated-column expression.
#:
#: Weighted so that a query term in the official name outranks the same term in
#: the body text: ``A`` name, ``B`` plain-language summary and keywords, ``C``
#: description. Written once here and referenced by both the ORM model and the
#: migration so the two cannot drift - a mismatch would make every autogenerate
#: run want to rewrite the column.
SEARCH_VECTOR_EXPRESSION: Final[str] = (
    "setweight(to_tsvector('english', coalesce(official_name, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(plain_language_summary, '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(sc_text_array_to_text(keywords), '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(description, '')), 'C')"
)


class AssistanceProgram(TimestampMixin, SoftDeleteMixin, Base):
    """One curated real assistance program.

    Mirrors :class:`app.schemas.program.AssistanceProgram`. The primary key is
    ``uuid5(PROGRAM_NAMESPACE, slug)`` (see
    :func:`app.schemas.common.program_uuid`) rather than a random UUID, so the
    seed script, the fixtures and a saved action plan all agree on a program's
    id without a lookup table and re-seeding is idempotent.
    """

    __tablename__ = "assistance_programs"

    id: Mapped[UuidPk]
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    official_name: Mapped[str] = mapped_column(Text, nullable=False)
    administering_org: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Who actually runs the program. Required so no screen can imply "
        "that SecondChance is the provider or an agency.",
    )
    category: Mapped[EnumText]
    description: Mapped[str] = mapped_column(Text, nullable=False)
    plain_language_summary: Mapped[str] = mapped_column(Text, nullable=False)
    who_it_helps: Mapped[EmptyTextArray]
    geo_scope: Mapped[JsonDict]
    official_url: Mapped[str] = mapped_column(Text, nullable=False)
    application_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    phone: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[EmptyTextArray]
    verification_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'UNVERIFIED'")
    )
    last_verified_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)
    active_rule_version: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Version string of the approved rule set in force. NULL means the "
        "program is listed as information only and cannot be matched against.",
    )

    #: Generated, stored ``tsvector`` for lexical retrieval, with a GIN index.
    #:
    #: Generated rather than trigger-maintained: a trigger can be bypassed by a
    #: ``COPY`` or forgotten by a migration, whereas a generated column cannot
    #: be out of date by construction.
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(SEARCH_VECTOR_EXPRESSION, persisted=True),
        nullable=False,
    )

    sources: Mapped[list[ProgramSource]] = relationship(
        back_populates="program",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )
    rule_versions: Mapped[list[ProgramRuleVersion]] = relationship(
        back_populates="program",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise_on_sql",
    )
    matches: Mapped[list[ProgramMatch]] = relationship(
        back_populates="program",
        lazy="raise_on_sql",
        viewonly=True,
    )

    __table_args__ = (
        UniqueConstraint("slug"),
        Index("ix_assistance_programs_category", "category"),
        Index("ix_assistance_programs_verification_status", "verification_status"),
        Index(
            "ix_assistance_programs_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        CheckConstraint(
            "verification_status <> 'VERIFIED' OR last_verified_at IS NOT NULL",
            name="verified_records_last_verified_at",
        ),
    )


class ProgramSource(TimestampMixin, Base):
    """An official document that justifies what the catalog says.

    ``excerpt`` is retrieved third-party content and is therefore untrusted:
    instructions found inside it must never influence a prompt. The database
    stores it as data and nothing more.

    The primary key is the human-authored ``SourceId`` slug, not a UUID,
    because these ids appear verbatim in citations and in curated rule sets and
    have to stay readable.
    """

    __tablename__ = "program_sources"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    program_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assistance_programs.id", ondelete="CASCADE"), nullable=False
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    publisher: Mapped[str] = mapped_column(Text, nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_at: Mapped[dt.datetime] = mapped_column(nullable=False)
    content_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="SHA-256 of the fetched body. A changed hash on re-fetch is what "
        "moves a rule set back to PENDING_REVIEW instead of silently "
        "changing people's results.",
    )
    freshness: Mapped[EnumText]

    program: Mapped[AssistanceProgram] = relationship(back_populates="sources")

    __table_args__ = (
        UniqueConstraint("program_id", "url"),
        Index("ix_program_sources_program_id", "program_id"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash_is_sha256_hex"),
    )


class ProgramRuleVersion(TimestampMixin, Base):
    """One version of one program's requirement tree. Append-only.

    ``root`` is a serialised :class:`app.schemas.eligibility.Condition` tree -
    the same object the rule engine evaluates and the same object the program
    detail page renders its requirement bullets from, so what a user reads and
    what the engine checks cannot drift.

    Two uniqueness rules matter:

    * ``(program_id, version)`` - a version string identifies exactly one rule
      set, which is what makes a stored ``rule_version`` on a match a usable
      audit trail.
    * at most one *open* version per program, enforced by a partial unique
      index on ``effective_to IS NULL``. Without it, a botched supersede could
      leave two current rule sets and the answer to "what are this program's
      rules today" would depend on row order.

    Old versions are never deleted or rewritten.
    """

    __tablename__ = "program_rule_versions"

    id: Mapped[UuidPk]
    program_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("assistance_programs.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc='Monotonic version string, e.g. "2026.1". Text, not an integer: '
        "RuleSet.version is a string and may carry a point release.",
    )
    effective_from: Mapped[RequiredDate]
    effective_to: Mapped[OptionalDate]
    root: Mapped[JsonDict]
    source_ids: Mapped[EmptyTextArray]
    review_status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'DRAFT'"),
        doc="Only APPROVED rule sets may influence a user's results. This gate "
        "is what stops a newly ingested requirement from changing anyone's "
        "status before a human has read it.",
    )
    limitations: Mapped[EmptyTextArray]
    reviewed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[dt.datetime | None] = mapped_column(nullable=True)

    program: Mapped[AssistanceProgram] = relationship(back_populates="rule_versions")

    __table_args__ = (
        UniqueConstraint("program_id", "version"),
        Index(
            "uq_program_rule_versions_program_id_active",
            "program_id",
            unique=True,
            postgresql_where=text("effective_to IS NULL"),
        ),
        Index(
            "ix_program_rule_versions_active_lookup",
            "program_id",
            "review_status",
            "effective_from",
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_to > effective_from",
            name="effective_to_after_effective_from",
        ),
        CheckConstraint(
            "review_status <> 'APPROVED' OR reviewed_at IS NOT NULL",
            name="approved_rules_record_reviewed_at",
        ),
    )
