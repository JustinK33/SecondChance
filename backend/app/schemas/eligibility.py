"""The typed eligibility rule DSL and its resolution semantics.

This module is the heart of the product's trust model. Three invariants:

1. **The rule engine decides the status. The model only writes prose.** A
   status is a pure function of a profile and a rule set. Nothing in this
   module takes model output as an input.
2. **Unknown is not false.** Every condition resolves to :class:`Trilean`, and
   a missing profile field yields ``UNKNOWN``, which becomes
   ``MORE_INFO_NEEDED``. Treating missing data as disqualifying would tell real
   people not to apply for help they could get.
3. **Every condition explains itself.** ``description`` is required and written
   in plain language, and ``source_ids`` points at the official document that
   justifies the rule. A requirement with no source cannot support an
   eligibility claim.

Resolution semantics
--------------------
``AllOf``  -> ``FALSE`` if any child is ``FALSE``; otherwise ``UNKNOWN`` if any
              child is ``UNKNOWN``; otherwise ``TRUE``.
``AnyOf``  -> ``TRUE`` if any child is ``TRUE``; otherwise ``UNKNOWN`` if any
              child is ``UNKNOWN``; otherwise ``FALSE``.
``Not``    -> swaps ``TRUE`` and ``FALSE``, preserves ``UNKNOWN``.

A definite result short-circuits the unknowns, which is what makes the engine
honest in both directions: one failed hard requirement is enough for
``UNLIKELY_MATCH`` even with other fields missing, and one satisfied
alternative is enough for ``TRUE`` without probing the rest.

Mapping to the user-facing verdict::

    TRUE    -> MatchStatus.POTENTIAL_MATCH
    UNKNOWN -> MatchStatus.MORE_INFO_NEEDED
    FALSE   -> MatchStatus.UNLIKELY_MATCH
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import (
    Cents,
    ConfidenceScore,
    CountyName,
    NonEmptyStr,
    PlainText,
    Slug,
    SourceId,
    StateCode,
    UtcDatetime,
)
from app.schemas.enums import (
    CategoricalProfileField,
    MatchStatus,
    ProfileFlag,
    ReviewStatus,
    StudentStatus,
    Trilean,
)

__all__ = [
    "STATUS_BY_OUTCOME",
    "AgeCondition",
    "AllOf",
    "AnyOf",
    "CategoricalCondition",
    "Condition",
    "ConditionEvidence",
    "ConditionKind",
    "EligibilityEvaluation",
    "FlagCondition",
    "GeographyCondition",
    "HouseholdSizeCondition",
    "IncomeAbsoluteCondition",
    "IncomeFplCondition",
    "Not",
    "RequirementBullet",
    "RuleSet",
    "StudentStatusCondition",
    "render_requirements",
    "resolve_all_of",
    "resolve_any_of",
    "resolve_not",
    "status_for_outcome",
]

type ConditionKind = Literal[
    "geography",
    "student_status",
    "household_size",
    "income_fpl",
    "income_absolute",
    "categorical",
    "age",
    "flag",
    "all_of",
    "any_of",
    "not",
]


# ---------------------------------------------------------------------------
# three-valued logic
# ---------------------------------------------------------------------------
def resolve_all_of(outcomes: Iterable[Trilean]) -> Trilean:
    """Conjunction: ``FALSE`` dominates, then ``UNKNOWN``, else ``TRUE``.

    An empty conjunction is ``TRUE`` (nothing is required), but ``AllOf``
    rejects an empty child list so that case cannot arise from a rule set.
    """
    seen_unknown = False
    for outcome in outcomes:
        if outcome is Trilean.FALSE:
            return Trilean.FALSE
        if outcome is Trilean.UNKNOWN:
            seen_unknown = True
    return Trilean.UNKNOWN if seen_unknown else Trilean.TRUE


def resolve_any_of(outcomes: Iterable[Trilean]) -> Trilean:
    """Disjunction: ``TRUE`` dominates, then ``UNKNOWN``, else ``FALSE``.

    An empty disjunction is ``FALSE``, but ``AnyOf`` rejects an empty child
    list so that case cannot arise from a rule set.
    """
    seen_unknown = False
    for outcome in outcomes:
        if outcome is Trilean.TRUE:
            return Trilean.TRUE
        if outcome is Trilean.UNKNOWN:
            seen_unknown = True
    return Trilean.UNKNOWN if seen_unknown else Trilean.FALSE


def resolve_not(outcome: Trilean) -> Trilean:
    """Negation: swap ``TRUE``/``FALSE``, preserve ``UNKNOWN``.

    ``UNKNOWN`` is preserved rather than flipped: not knowing whether a
    condition holds is not the same as knowing it does not.
    """
    if outcome is Trilean.TRUE:
        return Trilean.FALSE
    if outcome is Trilean.FALSE:
        return Trilean.TRUE
    return Trilean.UNKNOWN


#: Canonical outcome-to-verdict mapping. The only place this translation exists.
STATUS_BY_OUTCOME: dict[Trilean, MatchStatus] = {
    Trilean.TRUE: MatchStatus.POTENTIAL_MATCH,
    Trilean.UNKNOWN: MatchStatus.MORE_INFO_NEEDED,
    Trilean.FALSE: MatchStatus.UNLIKELY_MATCH,
}


def status_for_outcome(outcome: Trilean) -> MatchStatus:
    """Translate a root condition outcome into the user-facing verdict."""
    return STATUS_BY_OUTCOME[outcome]


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------
class ConditionBase(BaseModel):
    """Fields every condition carries.

    ``description`` is required and is the text the user eventually reads, so it
    is written for a person ("You live in New Jersey"), not for an engineer
    ("state in ['NJ']").
    """

    model_config = ConfigDict(extra="forbid")

    description: PlainText = Field(
        description="Plain-language statement of the requirement, shown to users verbatim."
    )
    source_ids: list[SourceId] = Field(
        default_factory=list,
        description=(
            "Official sources that justify this requirement. A condition with "
            "no source may be evaluated but must not be cited as evidence."
        ),
    )


class GeographyCondition(ConditionBase):
    """Where the person must live.

    ``counties`` is compared case-insensitively against ``profile.county`` with
    a trailing "County" stripped, so "Middlesex" and "Middlesex County" match.
    An empty list means that level is unconstrained.
    """

    kind: Literal["geography"] = "geography"
    states: list[StateCode] = Field(default_factory=list)
    counties: list[CountyName] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_a_constraint(self) -> Self:
        if not self.states and not self.counties:
            msg = "GeographyCondition needs at least one state or county"
            raise ValueError(msg)
        return self


class StudentStatusCondition(ConditionBase):
    """Which student statuses qualify.

    ``StudentStatus.UNKNOWN`` must not appear in ``allowed``: a rule cannot
    require that we do not know something.
    """

    kind: Literal["student_status"] = "student_status"
    allowed: list[StudentStatus] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_unknown(self) -> Self:
        if StudentStatus.UNKNOWN in self.allowed:
            msg = "StudentStatus.UNKNOWN cannot be an allowed value"
            raise ValueError(msg)
        return self


class HouseholdSizeCondition(ConditionBase):
    """Inclusive bounds on household size."""

    kind: Literal["household_size"] = "household_size"
    min: Annotated[int, Field(ge=1)] | None = None
    max: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.min is None and self.max is None:
            msg = "HouseholdSizeCondition needs a min or a max"
            raise ValueError(msg)
        if self.min is not None and self.max is not None and self.min > self.max:
            msg = "min must not exceed max"
            raise ValueError(msg)
        return self


class IncomeFplCondition(ConditionBase):
    """Income at or below a percentage of the Federal Poverty Level.

    The FPL table is indexed by household size and year, so this condition
    needs ``household_size`` as well as income; if either is missing it
    resolves to ``UNKNOWN``.

    ``fpl_year`` is pinned rather than taken as "current" so an evaluation
    stays reproducible after the guidelines are updated, and
    ``household_definition`` records whose definition of "household" the
    program applies - they differ materially between programs and the engine
    must not silently substitute its own.
    """

    kind: Literal["income_fpl"] = "income_fpl"
    max_pct_fpl: Annotated[float, Field(gt=0, le=1000)] = Field(
        description="Upper bound as a percentage, e.g. 130.0 for 130% of FPL."
    )
    fpl_year: Annotated[int, Field(ge=2020, le=2100)]
    household_definition: PlainText = Field(
        description="How this program counts a household, quoted from the official source."
    )


class IncomeAbsoluteCondition(ConditionBase):
    """Income at or below a fixed dollar amount.

    Exactly one of the two bounds is set. The engine normalises the profile's
    income range to the same period before comparing, and resolves ``UNKNOWN``
    when the range straddles the limit.
    """

    kind: Literal["income_absolute"] = "income_absolute"
    max_monthly_cents: Cents | None = None
    max_annual_cents: Cents | None = None

    @model_validator(mode="after")
    def _exactly_one_bound(self) -> Self:
        bounds = [self.max_monthly_cents, self.max_annual_cents]
        if sum(bound is not None for bound in bounds) != 1:
            msg = "set exactly one of max_monthly_cents or max_annual_cents"
            raise ValueError(msg)
        return self


class CategoricalCondition(ConditionBase):
    """A closed-vocabulary profile field must be one of ``any_of``.

    ``field`` is restricted to :class:`CategoricalProfileField` so a typo in a
    rule set fails at load time instead of evaluating to ``UNKNOWN`` forever.
    Values are compared as the enum's wire strings.
    """

    kind: Literal["categorical"] = "categorical"
    field: CategoricalProfileField
    any_of: list[NonEmptyStr] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_unknown_values(self) -> Self:
        if any(value.upper() == "UNKNOWN" for value in self.any_of):
            msg = "'UNKNOWN' cannot be an accepted categorical value"
            raise ValueError(msg)
        return self


class AgeCondition(ConditionBase):
    """Inclusive bounds on the applicant's age in years."""

    kind: Literal["age"] = "age"
    min: Annotated[int, Field(ge=0, le=120)] | None = None
    max: Annotated[int, Field(ge=0, le=120)] | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        if self.min is None and self.max is None:
            msg = "AgeCondition needs a min or a max"
            raise ValueError(msg)
        if self.min is not None and self.max is not None and self.min > self.max:
            msg = "min must not exceed max"
            raise ValueError(msg)
        return self


class FlagCondition(ConditionBase):
    """A boolean profile fact must equal ``expected``.

    Resolves to ``UNKNOWN`` when the flag was never stated, because flags are
    only set from explicit user input and absence means "not asked", not
    "false".
    """

    kind: Literal["flag"] = "flag"
    flag: ProfileFlag
    expected: bool = True


class AllOf(ConditionBase):
    """Every child condition must hold. See module docstring for semantics."""

    kind: Literal["all_of"] = "all_of"
    conditions: list[Condition] = Field(min_length=1)


class AnyOf(ConditionBase):
    """At least one child condition must hold. See module docstring."""

    kind: Literal["any_of"] = "any_of"
    conditions: list[Condition] = Field(min_length=1)


class Not(ConditionBase):
    """The child condition must not hold. ``UNKNOWN`` is preserved."""

    kind: Literal["not"] = "not"
    condition: Condition


#: The recursive rule expression. Discriminated on ``kind``, so a malformed
#: rule set produces a precise error instead of a confused union attempt.
type Condition = Annotated[
    GeographyCondition
    | StudentStatusCondition
    | HouseholdSizeCondition
    | IncomeFplCondition
    | IncomeAbsoluteCondition
    | CategoricalCondition
    | AgeCondition
    | FlagCondition
    | AllOf
    | AnyOf
    | Not,
    Field(discriminator="kind"),
]

AllOf.model_rebuild()
AnyOf.model_rebuild()
Not.model_rebuild()


# ---------------------------------------------------------------------------
# evidence and results
# ---------------------------------------------------------------------------
class ConditionEvidence(BaseModel):
    """The audit record of evaluating one condition.

    Stored with the evaluation so that an assessment made months ago can still
    be explained: which rule ran, which profile fields it read, what threshold
    it applied, what it concluded and which official source justified it.
    ``children`` mirrors the condition tree for combinators, so the reasoning
    is inspectable rather than collapsed into one boolean.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ConditionKind
    description: PlainText
    profile_fields_used: list[str] = Field(
        default_factory=list, description="Profile fields actually read, for auditability."
    )
    outcome: Trilean
    threshold_applied: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "The concrete threshold used, in plain language, e.g. "
            '"130% of the 2026 federal poverty level for a household of 4".'
        ),
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields whose absence caused an UNKNOWN outcome.",
    )
    source_ids: list[SourceId] = Field(default_factory=list)
    children: list[ConditionEvidence] = Field(
        default_factory=list, description="Child evidence for all_of / any_of / not."
    )

    @model_validator(mode="after")
    def _unknown_needs_a_reason(self) -> Self:
        """An ``UNKNOWN`` leaf must say which field it was missing.

        Without this, "we need more information" reaches the user with nothing
        to act on, which is the most common way a product like this becomes
        useless.
        """
        is_leaf = not self.children
        if self.outcome is Trilean.UNKNOWN and is_leaf and not self.missing_fields:
            msg = f"UNKNOWN leaf evidence for {self.kind!r} must list missing_fields"
            raise ValueError(msg)
        return self


ConditionEvidence.model_rebuild()


class EligibilityEvaluation(BaseModel):
    """One deterministic assessment of one program for one profile.

    Reproducible by construction: given the same profile, the same
    ``rule_version`` and the same FPL table, re-running the engine must produce
    an identical record apart from ``evaluated_at``.
    """

    model_config = ConfigDict(extra="forbid")

    program_id: uuid.UUID
    program_slug: Slug
    rule_version: NonEmptyStr = Field(
        description="Version of the rule set that produced this result. Never 'latest'."
    )
    status: MatchStatus
    outcome: Trilean = Field(description="Root condition outcome that produced ``status``.")
    score: ConfidenceScore = Field(
        default=0.0,
        description=(
            "Ranking aid only, in [0,1]. Combines how much of the rule set "
            "resolved definitively with retrieval relevance. It never affects "
            "``status`` and is not a probability of eligibility."
        ),
    )
    evidence: list[ConditionEvidence] = Field(default_factory=list)
    missing_fields: list[str] = Field(
        default_factory=list, description="Union of missing_fields across the evidence tree."
    )
    limitations: list[PlainText] = Field(
        default_factory=list,
        description=(
            "Known gaps in this assessment: requirements the engine cannot "
            "evaluate, stale sources, conflicting guidance. Always surfaced to "
            "the user alongside the status."
        ),
    )
    evaluated_at: UtcDatetime

    @model_validator(mode="after")
    def _status_matches_outcome(self) -> Self:
        """Enforce the status mapping structurally.

        This is the invariant that makes "Gemini never changes a status"
        checkable rather than aspirational: any code path that hands back a
        status inconsistent with the deterministic outcome fails validation.
        """
        expected = STATUS_BY_OUTCOME[self.outcome]
        if self.status is not expected:
            msg = (
                f"status {self.status.value} contradicts outcome "
                f"{self.outcome.value} (expected {expected.value})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _more_info_needs_fields(self) -> Self:
        if self.status is MatchStatus.MORE_INFO_NEEDED and not self.missing_fields:
            msg = "MORE_INFO_NEEDED requires at least one entry in missing_fields"
            raise ValueError(msg)
        return self


class RuleSet(BaseModel):
    """A versioned, dated, reviewed set of requirements for one program.

    Rule sets are append-only. Superseding one creates a new version with a new
    ``effective_from`` and sets the old one's ``effective_to``, so an evaluation
    recorded last month can still be explained against the rules that were in
    force then.

    A rule set only influences matching when ``review_status`` is ``APPROVED``.
    That gate is what stops a newly ingested or model-extracted requirement from
    changing anyone's results before a human has read it.
    """

    model_config = ConfigDict(extra="forbid")

    version: NonEmptyStr = Field(
        description='Monotonic version string for this program, e.g. "2026.1".'
    )
    program_slug: Slug
    effective_from: date
    effective_to: date | None = Field(
        default=None, description="Exclusive end of applicability. ``None`` means current."
    )
    root: Condition = Field(description="Root of the condition tree, usually an ``AllOf``.")
    source_ids: list[SourceId] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    limitations: list[PlainText] = Field(
        default_factory=list,
        description="Requirements known to exist that this rule set does not encode.",
    )
    reviewed_by: str | None = Field(
        default=None, max_length=120, description="Who approved it. Not an end user."
    )
    reviewed_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            msg = "effective_to must be after effective_from"
            raise ValueError(msg)
        if self.review_status is ReviewStatus.APPROVED and self.reviewed_at is None:
            msg = "an APPROVED rule set must record reviewed_at"
            raise ValueError(msg)
        return self

    def is_effective_on(self, when: date) -> bool:
        if when < self.effective_from:
            return False
        return self.effective_to is None or when < self.effective_to

    def is_usable_for_matching(self, when: date) -> bool:
        """Whether this rule set may influence a user's results right now."""
        return self.review_status is ReviewStatus.APPROVED and self.is_effective_on(when)


class RequirementBullet(BaseModel):
    """One plain-language requirement line rendered for the program detail view."""

    model_config = ConfigDict(extra="forbid")

    text: PlainText
    source_ids: list[SourceId] = Field(default_factory=list)
    is_evaluable: bool = Field(
        default=True,
        description=(
            "False for requirements the engine cannot check automatically; the "
            "UI shows these as 'confirm with the program'."
        ),
    )
    confidence: ConfidenceScore = 1.0


def _flatten_requirements(condition: Condition, depth: int) -> list[RequirementBullet]:
    """Render a condition tree as flat bullets, one level of nesting deep.

    ``AllOf`` contributes its children directly, because "all of these must be
    true" is exactly a bullet list. ``AnyOf`` and ``Not`` contribute their own
    ``description`` instead, since an author wording a disjunction or negation
    for a reader always does better than a mechanical join.
    """
    if isinstance(condition, AllOf) and depth == 0:
        bullets: list[RequirementBullet] = []
        for child in condition.conditions:
            bullets.extend(_flatten_requirements(child, depth + 1))
        return bullets
    return [
        RequirementBullet(
            text=condition.description,
            source_ids=list(condition.source_ids),
            is_evaluable=True,
        )
    ]


def render_requirements(
    root: Condition, extra_limitations: Sequence[str] = ()
) -> list[RequirementBullet]:
    """Render a rule set's root condition as the detail view's bullet list.

    Deterministic, uses only author-written ``description`` text, and therefore
    cannot fabricate a requirement. ``extra_limitations`` are appended as
    non-evaluable bullets so a requirement the engine cannot check is still
    shown to the user.
    """
    bullets = _flatten_requirements(root, 0)
    bullets.extend(
        RequirementBullet(text=text, source_ids=[], is_evaluable=False)
        for text in extra_limitations
    )
    return bullets
