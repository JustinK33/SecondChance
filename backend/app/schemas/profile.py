"""Profile schemas: what we know about a person, and how we learned it.

The central modelling decision is that **income is a range, never a point**.
People in the target audience rarely know a precise figure, and a question that
demands one either gets a guess or gets abandoned. So the profile carries
``income_min_cents``/``income_max_cents`` and the rule engine resolves a
threshold against the whole range:

* the entire range under the limit  -> ``TRUE``
* the entire range over the limit   -> ``FALSE``
* the range straddles the limit     -> ``UNKNOWN`` (ask, do not guess)

The second decision is that **unknown is explicit**. ``unknown_fields`` lists
what the user was asked or prompted about and did not answer, which is how the
clarification step avoids asking the same question twice.
"""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.ai import ModelCallSource
from app.schemas.common import (
    Cents,
    ConfidenceScore,
    CountyName,
    NonEmptyStr,
    PlainText,
    StateCode,
    UtcDatetime,
)
from app.schemas.enums import (
    ClarificationInputKind,
    EmploymentStatus,
    HousingSituation,
    IncomePeriod,
    NeedCategory,
    ProfileFlag,
    StudentStatus,
)

__all__ = [
    "MAX_HOUSEHOLD_SIZE",
    "MAX_NARRATIVE_CHARS",
    "PROFILE_FIELD_NAMES",
    "ClarificationAnswer",
    "ClarificationOption",
    "ClarificationQuestion",
    "ClarificationQuestionSet",
    "ExtractedFlag",
    "ExtractedProfile",
    "FieldEvidence",
    "HouseholdComposition",
    "IntakeRequest",
    "IntakeResponse",
    "ProfileDraft",
    "ProfileSnapshot",
    "RangeValue",
    "UserProfile",
]

MAX_NARRATIVE_CHARS = 4000
MAX_HOUSEHOLD_SIZE = 20
_MIN_AGE = 0
_MAX_AGE = 120
_MONTHS_PER_YEAR = 12

HouseholdSize = Annotated[int, Field(ge=1, le=MAX_HOUSEHOLD_SIZE)]
Age = Annotated[int, Field(ge=_MIN_AGE, le=_MAX_AGE)]

#: Every field name that may legitimately appear in ``unknown_fields``, in a
#: ``ClarificationQuestion.field``, or in ``ConditionEvidence.missing_fields``.
#: Validated against, so a typo in a rule or a prompt fails loudly.
PROFILE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "state",
        "county",
        "student_status",
        "household_size",
        "household_composition",
        "income_min_cents",
        "income_max_cents",
        "income_period",
        "income",
        "employment_status",
        "housing_situation",
        "needs",
        "age",
        *(f"flags.{flag.value}" for flag in ProfileFlag),
    }
)


class HouseholdComposition(BaseModel):
    """Who is in the household, as counts rather than people.

    Counts only - no names, no dates of birth. Programs define "household"
    differently (SNAP counts people who buy and prepare food together; a tax
    credit counts dependents), so each count is stored separately and each rule
    states which definition it applies in its own ``household_definition``.
    """

    model_config = ConfigDict(extra="forbid")

    adults: Annotated[int, Field(ge=0, le=MAX_HOUSEHOLD_SIZE)] | None = None
    children_under_18: Annotated[int, Field(ge=0, le=MAX_HOUSEHOLD_SIZE)] | None = None
    seniors_65_plus: Annotated[int, Field(ge=0, le=MAX_HOUSEHOLD_SIZE)] | None = None
    full_time_students: Annotated[int, Field(ge=0, le=MAX_HOUSEHOLD_SIZE)] | None = None
    members_with_disability: Annotated[int, Field(ge=0, le=MAX_HOUSEHOLD_SIZE)] | None = None

    def known_minimum(self) -> int | None:
        """Lower bound on household size implied by the counts, if any are set.

        ``adults`` and ``children_under_18`` partition the household; the other
        counts are overlapping attributes, so only the partition is summed.
        """
        parts = [self.adults, self.children_under_18]
        known = [part for part in parts if part is not None]
        if not known:
            return None
        return sum(known)

    def is_empty(self) -> bool:
        return all(value is None for value in self.model_dump().values())


class UserProfile(BaseModel):
    """The sanitised, validated profile the rule engine evaluates against.

    Every field is optional because a profile is built incrementally - from a
    narrative, then from clarification answers - and must be evaluable at any
    point. Absence means "not known", never "not applicable".
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state: StateCode | None = None
    county: CountyName | None = None
    student_status: StudentStatus | None = None
    household_size: HouseholdSize | None = None
    household_composition: HouseholdComposition = Field(default_factory=HouseholdComposition)

    income_min_cents: Cents | None = Field(
        default=None, description="Low end of the reported income range, in cents."
    )
    income_max_cents: Cents | None = Field(
        default=None,
        description=(
            "High end of the reported income range, in cents. Equal to "
            "income_min_cents when the user gave an exact figure."
        ),
    )
    income_period: IncomePeriod | None = Field(
        default=None, description="Period the income figures refer to. Required if either is set."
    )

    employment_status: EmploymentStatus | None = None
    housing_situation: HousingSituation | None = None
    needs: list[NeedCategory] = Field(default_factory=list)
    age: Age | None = None

    flags: dict[ProfileFlag, bool] = Field(
        default_factory=dict,
        description=(
            "Boolean facts set only from explicit user input. A flag that is "
            "absent is unknown, not false."
        ),
    )

    unknown_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Profile fields the user was given the chance to supply and did "
            "not. Prevents the clarification step from asking again."
        ),
    )
    extras: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        description=(
            "Non-eligibility context captured for wording only, for example "
            "a preferred language. Never read by the rule engine."
        ),
    )

    # ------------------------------------------------------------------ rules
    @model_validator(mode="after")
    def _validate_income_range(self) -> Self:
        if (
            self.income_min_cents is not None
            and self.income_max_cents is not None
            and self.income_min_cents > self.income_max_cents
        ):
            msg = (
                "income_min_cents must not exceed income_max_cents "
                f"({self.income_min_cents} > {self.income_max_cents})"
            )
            raise ValueError(msg)
        has_amount = self.income_min_cents is not None or self.income_max_cents is not None
        if has_amount and self.income_period is None:
            msg = (
                "income_period is required when an income amount is set; an "
                "amount with an unknown period cannot be normalised and must "
                "not be assumed to be monthly"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_household(self) -> Self:
        minimum = self.household_composition.known_minimum()
        over = minimum is not None and minimum > (self.household_size or 0)
        if self.household_size is not None and over:
            msg = (
                "household_composition counts "
                f"({minimum}) exceed household_size ({self.household_size})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_unknown_fields(self) -> Self:
        unexpected = sorted(set(self.unknown_fields) - PROFILE_FIELD_NAMES)
        if unexpected:
            msg = f"unknown_fields contains names that are not profile fields: {unexpected}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _dedupe_needs(self) -> Self:
        if len(set(self.needs)) != len(self.needs):
            deduped = list(dict.fromkeys(self.needs))
            object.__setattr__(self, "needs", deduped)
        return self

    # -------------------------------------------------------------- accessors
    def income_monthly_range_cents(self) -> tuple[int | None, int | None]:
        """Income range normalised to cents per month.

        Returns ``(None, None)`` when no income has been reported, and may
        return a half-open range such as ``(250000, None)`` when the user gave
        only a floor.

        Annual figures are divided by 12 and **rounded outward** - the minimum
        floored, the maximum ceilinged - so a rounding artefact can only ever
        widen the range. Widening can turn a ``TRUE`` into an ``UNKNOWN`` (we
        ask a question), never a ``TRUE`` into a ``FALSE`` (we wrongly tell
        someone not to bother applying).
        """
        if self.income_period is None:
            return (None, None)
        if self.income_period is IncomePeriod.MONTHLY:
            return (self.income_min_cents, self.income_max_cents)
        low = (
            self.income_min_cents // _MONTHS_PER_YEAR if self.income_min_cents is not None else None
        )
        high = (
            math.ceil(self.income_max_cents / _MONTHS_PER_YEAR)
            if self.income_max_cents is not None
            else None
        )
        return (low, high)

    def income_annual_range_cents(self) -> tuple[int | None, int | None]:
        """Income range normalised to cents per year, rounded outward."""
        if self.income_period is None:
            return (None, None)
        if self.income_period is IncomePeriod.ANNUAL:
            return (self.income_min_cents, self.income_max_cents)
        low = (
            self.income_min_cents * _MONTHS_PER_YEAR if self.income_min_cents is not None else None
        )
        high = (
            self.income_max_cents * _MONTHS_PER_YEAR if self.income_max_cents is not None else None
        )
        return (low, high)

    def has_income_information(self) -> bool:
        return self.income_period is not None and (
            self.income_min_cents is not None or self.income_max_cents is not None
        )

    def flag(self, name: ProfileFlag) -> bool | None:
        """Return a flag's value, or ``None`` when the user never stated it."""
        return self.flags.get(name)

    def is_unknown(self, field_name: str) -> bool:
        """Whether ``field_name`` is absent or was explicitly declined."""
        if field_name in self.unknown_fields:
            return True
        if field_name.startswith("flags."):
            return self.flags.get(ProfileFlag(field_name.removeprefix("flags."))) is None
        if field_name == "income":
            return not self.has_income_information()
        if field_name == "household_composition":
            return self.household_composition.is_empty()
        if field_name == "needs":
            return not self.needs
        return getattr(self, field_name, None) is None


class ProfileDraft(BaseModel):
    """An all-optional patch over a :class:`UserProfile`.

    Used for pre-filled intake answers and for the "edit my information"
    screen. Validated on its own so the client gets field-level errors, then
    merged by ``app.services.profile_assembly`` which owns the precedence
    policy. ``None`` means "no opinion"; it never clears a known value.
    """

    model_config = ConfigDict(extra="forbid")

    state: StateCode | None = None
    county: CountyName | None = None
    student_status: StudentStatus | None = None
    household_size: HouseholdSize | None = None
    household_composition: HouseholdComposition | None = None
    income_min_cents: Cents | None = None
    income_max_cents: Cents | None = None
    income_period: IncomePeriod | None = None
    employment_status: EmploymentStatus | None = None
    housing_situation: HousingSituation | None = None
    needs: list[NeedCategory] | None = None
    age: Age | None = None
    flags: dict[ProfileFlag, bool] | None = None

    @model_validator(mode="after")
    def _validate_income_range(self) -> Self:
        if (
            self.income_min_cents is not None
            and self.income_max_cents is not None
            and self.income_min_cents > self.income_max_cents
        ):
            msg = "income_min_cents must not exceed income_max_cents"
            raise ValueError(msg)
        return self

    def set_fields(self) -> dict[str, object]:
        """The fields the client actually supplied, excluding ``None``."""
        return self.model_dump(exclude_none=True)


class IntakeRequest(BaseModel):
    """What the user submits on the onboarding screen.

    The narrative is free text in the user's own words. Pre-filled fields come
    from the optional structured questions shown alongside it, and always win
    over the same field extracted from the narrative.
    """

    model_config = ConfigDict(extra="forbid")

    narrative: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=MAX_NARRATIVE_CHARS),
    ] = Field(
        description="The user's description of their situation, in their own words.",
    )
    prefilled: ProfileDraft | None = Field(
        default=None, description="Structured answers the user gave directly."
    )


class FieldEvidence(BaseModel):
    """Why the extraction model believes one profile field has the value it gave.

    ``source_span`` is a verbatim quote from the narrative. It exists so a
    reviewer can check an extraction against what the person actually wrote, and
    so the UI can show "you said: ..." instead of an unexplained value. It is
    personal content: never logged, never sent to a third party.
    """

    model_config = ConfigDict(extra="forbid")

    field: NonEmptyStr = Field(description="Profile field name this evidence is for.")
    confidence: ConfidenceScore = Field(
        description=(
            "The model's self-reported confidence. Advisory only; it never changes a status."
        )
    )
    source_span: str | None = Field(
        default=None,
        max_length=400,
        description="Verbatim excerpt of the narrative the value was read from.",
    )
    reasoning: str | None = Field(
        default=None, max_length=400, description="One short sentence of justification."
    )

    @model_validator(mode="after")
    def _validate_field_name(self) -> Self:
        if self.field not in PROFILE_FIELD_NAMES:
            msg = f"{self.field!r} is not a known profile field"
            raise ValueError(msg)
        return self


class ExtractedFlag(BaseModel):
    """One boolean fact the model read out of the narrative.

    Modelled as a list element rather than a dict entry so the structured
    output schema can constrain ``flag`` to the :class:`ProfileFlag` enum, which
    is what stops the model from inventing a flag name the rule engine would
    silently ignore.
    """

    model_config = ConfigDict(extra="forbid")

    flag: ProfileFlag
    value: bool
    confidence: ConfidenceScore = 1.0
    source_span: str | None = Field(default=None, max_length=400)


class ExtractedProfile(BaseModel):
    """Raw structured output of the extraction model. Never persisted as-is.

    Every field is optional: the model must leave a field unset rather than
    guess, and ``unknown_fields`` records what it could not determine. This is
    converted into a :class:`UserProfile` by
    ``app.services.profile_assembly.assemble_profile``, which applies
    precedence, drops low-confidence values and performs the real validation.
    """

    model_config = ConfigDict(extra="forbid")

    state: StateCode | None = None
    county: CountyName | None = None
    student_status: StudentStatus | None = None
    household_size: HouseholdSize | None = None
    household_composition: HouseholdComposition | None = None
    income_min_cents: Cents | None = None
    income_max_cents: Cents | None = None
    income_period: IncomePeriod | None = None
    employment_status: EmploymentStatus | None = None
    housing_situation: HousingSituation | None = None
    needs: list[NeedCategory] = Field(default_factory=list)
    age: Age | None = None
    flags: list[ExtractedFlag] = Field(default_factory=list)

    field_evidence: list[FieldEvidence] = Field(
        default_factory=list,
        description="Per-field confidence and the narrative span each value came from.",
    )
    unknown_fields: list[str] = Field(
        default_factory=list, description="Fields the narrative did not determine."
    )
    ambiguities: list[PlainText] = Field(
        default_factory=list,
        description=(
            "Statements that could be read more than one way. Drives "
            "clarification questions; never resolved by guessing."
        ),
    )

    def confidence_for(self, field_name: str) -> float | None:
        for evidence in self.field_evidence:
            if evidence.field == field_name:
                return evidence.confidence
        return None

    def source_span_for(self, field_name: str) -> str | None:
        for evidence in self.field_evidence:
            if evidence.field == field_name:
                return evidence.source_span
        return None

    def to_profile_fields(self, *, min_confidence: float = 0.0) -> dict[str, object]:
        """Mechanically project the set fields onto ``UserProfile`` field names.

        Applies no precedence or merge policy - that belongs to
        ``assemble_profile``. A field whose reported confidence is below
        ``min_confidence`` is dropped, and fields with no reported confidence
        are kept (absence of evidence is not low confidence).
        """
        simple = (
            "state",
            "county",
            "student_status",
            "household_size",
            "household_composition",
            "income_min_cents",
            "income_max_cents",
            "income_period",
            "employment_status",
            "housing_situation",
            "age",
        )
        out: dict[str, object] = {}
        for name in simple:
            value = getattr(self, name)
            if value is None:
                continue
            confidence = self.confidence_for(name)
            if confidence is not None and confidence < min_confidence:
                continue
            out[name] = value
        if self.needs:
            out["needs"] = list(dict.fromkeys(self.needs))
        flags = {item.flag: item.value for item in self.flags if item.confidence >= min_confidence}
        if flags:
            out["flags"] = flags
        return out


class ClarificationOption(BaseModel):
    """One choice for a select-style clarification question."""

    model_config = ConfigDict(extra="forbid")

    value: NonEmptyStr = Field(description="Machine value submitted back in the answer.")
    label: NonEmptyStr = Field(description="Plain-language label shown to the user.")
    help_text: str | None = Field(
        default=None, max_length=300, description="Optional one-line clarification of the option."
    )


class ClarificationQuestion(BaseModel):
    """A question asked only because the answer could change the results.

    ``why_it_matters`` is required. If an author cannot say what the answer
    would change, the question should not be asked - that requirement is what
    keeps the clarification step from turning into a government form.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr = Field(
        description="Stable id; see app.schemas.common.clarification_question_id."
    )
    field: NonEmptyStr = Field(description="The profile field this question resolves.")
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=5, max_length=300)
    ] = Field(description="The question, in plain language, at roughly a 6th-grade reading level.")
    input_kind: ClarificationInputKind
    options: list[ClarificationOption] = Field(default_factory=list)
    why_it_matters: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=5, max_length=300)
    ] = Field(description="What this answer would change, in the user's terms.")
    affected_program_ids: list[str] = Field(
        default_factory=list,
        description="Programs whose status could change. Empty means profile-wide.",
    )
    is_required: bool = Field(
        default=False,
        description=(
            "Questions are skippable by default. Skipping records the field as "
            "unknown and leaves affected programs at MORE_INFO_NEEDED."
        ),
    )
    unit: str | None = Field(
        default=None, max_length=40, description='Unit label for numeric input, e.g. "people".'
    )
    min_value: float | None = None
    max_value: float | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if self.field not in PROFILE_FIELD_NAMES:
            msg = f"{self.field!r} is not a known profile field"
            raise ValueError(msg)
        select_kinds = {
            ClarificationInputKind.SINGLE_SELECT,
            ClarificationInputKind.MULTI_SELECT,
        }
        if self.input_kind in select_kinds:
            if len(self.options) < 2:
                msg = f"{self.input_kind.value} needs at least two options"
                raise ValueError(msg)
            values = [option.value for option in self.options]
            if len(set(values)) != len(values):
                msg = "option values must be unique"
                raise ValueError(msg)
        elif self.options:
            msg = f"{self.input_kind.value} questions must not carry options"
            raise ValueError(msg)
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            msg = "min_value must not exceed max_value"
            raise ValueError(msg)
        return self


class ClarificationQuestionSet(BaseModel):
    """The questions produced for one pause in the workflow."""

    model_config = ConfigDict(extra="forbid")

    questions: list[ClarificationQuestion] = Field(default_factory=list, max_length=5)
    rationale: str | None = Field(
        default=None,
        max_length=400,
        description="Internal note on why these questions were chosen. Not shown to the user.",
    )

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [question.id for question in self.questions]
        if len(set(ids)) != len(ids):
            msg = "clarification question ids must be unique within a set"
            raise ValueError(msg)
        return self


class RangeValue(BaseModel):
    """A ``range``-kind answer, for example an income band."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        if self.min is None and self.max is None:
            msg = "a range answer needs at least one bound"
            raise ValueError(msg)
        if self.min is not None and self.max is not None and self.min > self.max:
            msg = "min must not exceed max"
            raise ValueError(msg)
        return self


class ClarificationAnswer(BaseModel):
    """One answer submitted when resuming a paused run.

    ``skipped`` is a real answer, not a missing one: it tells the workflow to
    record the field as unknown and stop asking, which is how a user gets
    results without surrendering information they would rather not share.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: NonEmptyStr
    field: NonEmptyStr
    value: str | bool | int | float | list[str] | RangeValue | None = None
    skipped: bool = False

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if self.field not in PROFILE_FIELD_NAMES:
            msg = f"{self.field!r} is not a known profile field"
            raise ValueError(msg)
        if self.skipped and self.value is not None:
            msg = "a skipped answer must not carry a value"
            raise ValueError(msg)
        if not self.skipped and self.value is None:
            msg = "value is required unless the question was skipped"
            raise ValueError(msg)
        return self


class ProfileSnapshot(BaseModel):
    """A stored profile version, for the "edit and re-run" flow.

    Keeping versions means an old evaluation stays auditable against the
    profile it was actually made from, instead of appearing to have been made
    from the current one.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    profile: UserProfile
    created_at: UtcDatetime
    changed_fields: list[str] = Field(default_factory=list)


class IntakeResponse(BaseModel):
    """Result of ``POST /api/v1/intake``: what we understood, before any run.

    Extraction only - no retrieval, no eligibility, no model-written prose - so
    it returns in roughly one model call. It exists so the onboarding screen can
    show "here is what we understood, is this right?" and let the user correct
    it before committing to a full run. Confirming cheap mistakes up front is
    far better than explaining expensive ones afterwards.
    """

    model_config = ConfigDict(extra="forbid")

    profile: UserProfile = Field(description="The assembled, validated profile.")
    unknown_fields: list[str] = Field(
        default_factory=list, description="Fields the narrative did not determine."
    )
    clarification_questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        max_length=5,
        description="Questions whose answers could change which programs are suggested.",
    )
    ambiguities: list[PlainText] = Field(
        default_factory=list, description="Statements that could be read more than one way."
    )
    extraction_source: ModelCallSource = Field(
        description="Whether the live model, a cached response or the deterministic path ran."
    )
