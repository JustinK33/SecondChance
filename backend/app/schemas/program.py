"""Assistance program catalog schemas.

Every program in SecondChance is a real program with a real official URL. The
schema is built so that a fabricated one is hard to represent and impossible to
present as fact:

* A program carries :class:`~app.schemas.enums.VerificationStatus`. Only
  ``VERIFIED`` records may support an eligibility claim; anything else is shown
  with an explicit "we have not verified this yet" label.
* Requirements are only ever rendered from author-written ``description`` text
  on an approved :class:`~app.schemas.eligibility.RuleSet`, so there is no code
  path that turns model prose into a requirement.
* Sources carry ``retrieved_at`` and ``content_hash``, so "last verified" is a
  fact about a fetch rather than a claim.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.common import (
    ConfidenceScore,
    CountyName,
    HttpUrlStr,
    NonEmptyStr,
    Page,
    PlainText,
    Slug,
    SourceId,
    StateCode,
    UtcDatetime,
    utcnow,
)
from app.schemas.eligibility import RequirementBullet, RuleSet
from app.schemas.enums import ProgramCategory, SourceFreshness, VerificationStatus

__all__ = [
    "AGING_AFTER",
    "STALE_AFTER",
    "AssistanceProgram",
    "GeoScope",
    "ProgramDetail",
    "ProgramPage",
    "ProgramSource",
    "ProgramSummary",
    "RetrievedProgram",
    "compute_freshness",
]

#: A source retrieved within this window is ``FRESH``.
AGING_AFTER: Final[timedelta] = timedelta(days=90)
#: Past this window a source is ``STALE`` and excluded from eligibility claims.
STALE_AFTER: Final[timedelta] = timedelta(days=365)

#: Free-form US phone string as the program publishes it, e.g. "1-800-687-9512".
PhoneNumber = Annotated[str, StringConstraints(strip_whitespace=True, min_length=7, max_length=40)]


def compute_freshness(retrieved_at: UtcDatetime, now: UtcDatetime | None = None) -> SourceFreshness:
    """Bucket a source by age. Deterministic; the only definition of freshness."""
    age = (now or utcnow()) - retrieved_at
    if age < AGING_AFTER:
        return SourceFreshness.FRESH
    if age < STALE_AFTER:
        return SourceFreshness.AGING
    return SourceFreshness.STALE


class ProgramSource(BaseModel):
    """An official document that justifies what the catalog says about a program.

    ``excerpt`` is the quoted passage a citation points at. It is retrieved
    third-party content and is therefore **untrusted**: instructions found
    inside it must never influence prompts or behaviour.

    ``content_hash`` is the hash of the fetched body. Comparing it on a re-fetch
    is how a policy change is detected, which is what moves a rule set back to
    ``PENDING_REVIEW`` instead of silently changing people's results.
    """

    model_config = ConfigDict(extra="forbid")

    id: SourceId = Field(description='Human-authored stable id, e.g. "nj-snap-income-limits-2026".')
    url: HttpUrlStr
    title: NonEmptyStr
    publisher: NonEmptyStr = Field(
        description='The issuing body, e.g. "NJ Department of Human Services".'
    )
    excerpt: PlainText = Field(description="Quoted passage this source is cited for. Untrusted.")
    retrieved_at: UtcDatetime
    content_hash: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] = Field(
        description="SHA-256 hex digest of the fetched document body."
    )
    freshness: SourceFreshness = Field(
        description="Freshness recorded at ingestion. Use current_freshness() when displaying."
    )

    def current_freshness(self, now: UtcDatetime | None = None) -> SourceFreshness:
        """Freshness recomputed from ``retrieved_at``.

        Authoritative at read time: the stored ``freshness`` field is a snapshot
        from ingestion and goes out of date simply by the passage of time.
        """
        return compute_freshness(self.retrieved_at, now)

    def supports_eligibility_claim(self, now: UtcDatetime | None = None) -> bool:
        return self.current_freshness(now) is not SourceFreshness.STALE


class GeoScope(BaseModel):
    """Where a program operates."""

    model_config = ConfigDict(extra="forbid")

    nationwide: bool = False
    states: list[StateCode] = Field(default_factory=list)
    counties: list[CountyName] = Field(default_factory=list)
    campuses: list[NonEmptyStr] = Field(
        default_factory=list, description='University scope, e.g. "Rutgers-New Brunswick".'
    )
    label: PlainText = Field(description='Plain-language scope, e.g. "All of New Jersey".')

    @model_validator(mode="after")
    def _require_a_scope(self) -> Self:
        if not (self.nationwide or self.states or self.counties or self.campuses):
            msg = "GeoScope needs nationwide, or at least one state, county or campus"
            raise ValueError(msg)
        return self


class AssistanceProgram(BaseModel):
    """A curated real assistance program."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID = Field(description="uuid5(PROGRAM_NAMESPACE, slug); see app.schemas.common.")
    slug: Slug
    official_name: NonEmptyStr = Field(
        description="The program's name exactly as the administering body writes it."
    )
    administering_org: NonEmptyStr = Field(
        description=(
            "Who actually runs it. Shown next to the name so the user can see "
            "SecondChance is not the provider."
        )
    )
    category: ProgramCategory
    description: PlainText = Field(
        description="Factual description, close to the official wording."
    )
    plain_language_summary: PlainText = Field(
        description="One or two sentences at roughly a 6th-grade reading level."
    )
    who_it_helps: list[PlainText] = Field(
        default_factory=list, description="Short audience bullets, e.g. 'Households with children'."
    )
    geo_scope: GeoScope
    official_url: HttpUrlStr = Field(description="The program's own official information page.")
    application_url: HttpUrlStr | None = Field(
        default=None, description="Direct application entry point, when one is published."
    )
    phone: PhoneNumber | None = None
    keywords: list[NonEmptyStr] = Field(
        default_factory=list, description="Lexical retrieval aids. Not shown to users."
    )
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    last_verified_at: UtcDatetime | None = Field(
        default=None, description="When a human last checked this record against the sources."
    )
    active_rule_version: NonEmptyStr | None = Field(
        default=None, description="Version of the approved rule set in force. ``None`` = unrated."
    )
    sources: list[ProgramSource] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verified_needs_evidence(self) -> Self:
        """``VERIFIED`` is a claim about human work, so it requires the receipts."""
        if self.verification_status is VerificationStatus.VERIFIED:
            if not self.sources:
                msg = "a VERIFIED program must cite at least one source"
                raise ValueError(msg)
            if self.last_verified_at is None:
                msg = "a VERIFIED program must record last_verified_at"
                raise ValueError(msg)
        return self

    def supports_eligibility_claims(self) -> bool:
        """Whether this record may be used to state a preliminary match.

        Requires human verification and an approved rule set. Unverified
        programs are still surfaced - they are useful leads - but they are
        presented as information, never as an assessment.
        """
        return (
            self.verification_status is VerificationStatus.VERIFIED
            and self.active_rule_version is not None
        )

    def to_summary(self) -> ProgramSummary:
        return ProgramSummary(
            id=self.id,
            slug=self.slug,
            official_name=self.official_name,
            administering_org=self.administering_org,
            category=self.category,
            plain_language_summary=self.plain_language_summary,
            geo_scope_label=self.geo_scope.label,
            official_url=self.official_url,
            application_url=self.application_url,
            verification_status=self.verification_status,
            last_verified_at=self.last_verified_at,
        )


class ProgramSummary(BaseModel):
    """List-view shape. Deliberately small: a results page renders many of these."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    slug: Slug
    official_name: NonEmptyStr
    administering_org: NonEmptyStr
    category: ProgramCategory
    plain_language_summary: PlainText
    geo_scope_label: PlainText
    official_url: HttpUrlStr
    application_url: HttpUrlStr | None = None
    verification_status: VerificationStatus
    last_verified_at: UtcDatetime | None = None


class ProgramDetail(BaseModel):
    """Full program view: the record, its sources and its requirements as prose.

    ``requirements`` is produced by
    :func:`app.schemas.eligibility.render_requirements` from the approved rule
    set, so the bullets a user reads and the conditions the engine evaluates
    come from the same object and cannot drift.
    """

    model_config = ConfigDict(extra="forbid")

    program: AssistanceProgram
    sources: list[ProgramSource] = Field(default_factory=list)
    requirements: list[RequirementBullet] = Field(default_factory=list)
    rule_version: NonEmptyStr | None = None
    rule_effective_from: UtcDatetime | None = None
    limitations: list[PlainText] = Field(
        default_factory=list,
        description="What this record does not cover. Shown, never hidden.",
    )

    @model_validator(mode="after")
    def _requirements_need_a_rule_version(self) -> Self:
        if self.requirements and self.rule_version is None:
            msg = "requirements were rendered without recording the rule_version they came from"
            raise ValueError(msg)
        return self


class RetrievedProgram(BaseModel):
    """A retrieval candidate: the program plus everything the engine needs next.

    Retrieval hands this to the eligibility engine so the engine performs no
    I/O of its own and stays a pure function - which is what makes its results
    reproducible and its tests meaningful.
    """

    model_config = ConfigDict(extra="forbid")

    program: AssistanceProgram
    ruleset: RuleSet | None = Field(
        default=None,
        description="The approved rule set in force, or ``None`` if the program has no rules yet.",
    )
    sources: list[ProgramSource] = Field(default_factory=list)
    retrieval_score: ConfidenceScore = Field(
        default=0.0,
        description="Relevance score in [0,1]. Ranking only; never an eligibility signal.",
    )
    matched_terms: list[str] = Field(
        default_factory=list, description="Query terms that matched, for debugging retrieval."
    )
    retrieval_reasons: list[PlainText] = Field(
        default_factory=list,
        description='Why this candidate surfaced, e.g. "you mentioned groceries".',
    )


#: Paginated program list, the body of ``GET /api/v1/programs``.
ProgramPage = Page[ProgramSummary]
