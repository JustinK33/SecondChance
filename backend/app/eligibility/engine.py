"""The deterministic eligibility engine.

No model participates in anything in this module. A status is a pure function
of a profile, a rule set and an FPL table, and the only way a status can be
produced is :func:`app.schemas.eligibility.status_for_outcome` applied to the
root condition's :class:`~app.schemas.enums.Trilean`.

The three decisions that matter
-------------------------------
**Unknown is not false.** A missing profile field yields ``UNKNOWN``, which
becomes ``MORE_INFO_NEEDED`` and a question. It never yields ``FALSE``, because
telling someone not to apply for help they could get is the worst outcome this
system can produce.

**Income is a range, compared whole.** The profile carries a range, not a
point. Entirely under the threshold is ``TRUE``; entirely over is ``FALSE``; a
range that straddles the threshold is ``UNKNOWN`` and puts
``income_min_cents``/``income_max_cents`` into ``missing_fields``. Normalisation
between monthly and annual rounds outward (see
:meth:`app.schemas.profile.UserProfile.income_monthly_range_cents`), so a
rounding artefact can only ever widen a range and turn a ``TRUE`` into a
question - never into a ``FALSE``.

**Only blocking unknowns are reported.** A condition can be ``UNKNOWN`` without
mattering: inside an ``AnyOf`` that already has a satisfied alternative, or
anywhere in a rule set that has already failed a hard requirement. Those
unknowns are recorded in the evidence tree for audit, but they are deliberately
kept out of ``EligibilityEvaluation.missing_fields``, because every entry there
becomes a question we ask a person. ``missing_fields`` therefore holds exactly
the fields whose answers could change this program's status, ordered by how
many of its conditions each one unblocks.

That last point is a conscious narrowing of the ``missing_fields`` field
description in ``app.schemas.eligibility`` ("union of missing_fields across the
evidence tree"): the union is still available by walking ``evidence``, and
relevance is worth more at the call site than completeness.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, assert_never

from app.eligibility.fpl import FplTable, default_fpl_table
from app.schemas import (
    AgeCondition,
    AllOf,
    AnyOf,
    AssistanceProgram,
    CategoricalCondition,
    CategoricalProfileField,
    ClarificationInputKind,
    ClarificationOption,
    ClarificationQuestion,
    Condition,
    ConditionEvidence,
    ConditionKind,
    EligibilityEvaluation,
    EmploymentStatus,
    FlagCondition,
    GeographyCondition,
    HouseholdSizeCondition,
    HousingSituation,
    IncomeAbsoluteCondition,
    IncomeFplCondition,
    IncomePeriod,
    Not,
    ProfileFlag,
    RetrievedProgram,
    ReviewStatus,
    RuleSet,
    StudentStatus,
    StudentStatusCondition,
    Trilean,
    UserProfile,
    clarification_question_id,
    resolve_all_of,
    resolve_any_of,
    resolve_not,
    status_for_outcome,
    utcnow,
)

__all__ = [
    "FIELD_PRIORITY",
    "FPL_TABLE_UNAVAILABLE",
    "MAX_CLARIFICATION_QUESTIONS",
    "REVIEW_DUE_AFTER",
    "ProgramRuleset",
    "RuleEngine",
    "evaluate_all",
    "evaluate_condition",
    "evaluate_program",
    "fields_read_by",
    "missing_field_questions",
    "reevaluate_affected",
]

_INCOME_MIN: Final[str] = "income_min_cents"
_INCOME_MAX: Final[str] = "income_max_cents"
_INCOME_PERIOD: Final[str] = "income_period"
_INCOME_ALIAS: Final[str] = "income"
_INCOME_FIELDS: Final[tuple[str, ...]] = (_INCOME_MIN, _INCOME_MAX, _INCOME_PERIOD)

#: Marker written into ``ConditionEvidence.threshold_applied`` when the FPL
#: table cannot be indexed at all (unknown year, out-of-table household size).
#: ``evaluate_program`` turns it into a user-visible limitation, because the
#: honest explanation is "we cannot check this requirement", not "tell us your
#: household size again".
FPL_TABLE_UNAVAILABLE: Final[str] = "no federal poverty guideline table available"

#: An approved rule set older than this is flagged as due for re-review.
#: Matches ``app.schemas.program.STALE_AFTER`` so "stale" means the same number
#: of days everywhere in the product.
REVIEW_DUE_AFTER: Final[timedelta] = timedelta(days=365)

#: Tie-break order when two missing fields unblock the same number of
#: conditions. Cheap and broadly useful answers come first: where you live is
#: one tap, an income band is a slider, and a single yes/no flag is last because
#: it usually only moves one program.
FIELD_PRIORITY: Final[tuple[str, ...]] = (
    "state",
    "county",
    "household_size",
    _INCOME_ALIAS,
    _INCOME_MIN,
    _INCOME_MAX,
    _INCOME_PERIOD,
    "student_status",
    "age",
    "employment_status",
    "housing_situation",
    "household_composition",
)

#: Clarification questions are a cost, not a feature: every extra one loses
#: people. Three is the product cap, matching ``GeminiPort.propose_clarifications``.
MAX_CLARIFICATION_QUESTIONS: Final[int] = 3

#: Terms that must never appear in a question we ask. Nothing in
#: ``PROFILE_FIELD_NAMES`` could produce one, so this is a belt-and-braces
#: check that fails loudly in tests if a template is ever reworded badly.
SENSITIVE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "social security number",
        "ssn",
        "bank account",
        "routing number",
        "account number",
        "credit card",
        "password",
        "immigration status",
        "date of birth",
    }
)


# ---------------------------------------------------------------------------
# small formatting helpers (never touch user income amounts)
# ---------------------------------------------------------------------------
def _format_pct(pct: float) -> str:
    return f"{pct:g}"


def _format_dollars(cents: int) -> str:
    whole, remainder = divmod(cents, 100)
    return f"${whole:,}" if remainder == 0 else f"${whole:,}.{remainder:02d}"


def _period_label(period: IncomePeriod) -> str:
    return "a month" if period is IncomePeriod.MONTHLY else "a year"


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _normalize_county(value: str) -> str:
    return _normalize(value).removesuffix(" county").strip()


# ---------------------------------------------------------------------------
# range comparison - the subtle part
# ---------------------------------------------------------------------------
def _compare_range_to_limit(
    low: int | None, high: int | None, limit: int
) -> tuple[Trilean, list[str]]:
    """Resolve "income is at or below ``limit``" over a range.

    Returns the outcome and, when ``UNKNOWN``, the income fields whose values
    would settle it. A straddling range reports both bounds: knowing only one
    end of a range that crosses the threshold still leaves the answer open.
    """
    if low is None and high is None:
        return Trilean.UNKNOWN, [_INCOME_MIN, _INCOME_MAX]
    if high is not None and high <= limit:
        return Trilean.TRUE, []
    if low is not None and low > limit:
        return Trilean.FALSE, []
    missing = [name for name, bound in ((_INCOME_MIN, low), (_INCOME_MAX, high)) if bound is None]
    return Trilean.UNKNOWN, missing or [_INCOME_MIN, _INCOME_MAX]


# ---------------------------------------------------------------------------
# leaf evaluation
# ---------------------------------------------------------------------------
def _leaf(
    condition: Condition,
    kind: ConditionKind,
    outcome: Trilean,
    *,
    fields: Sequence[str] = (),
    missing: Sequence[str] = (),
    threshold: str | None = None,
) -> ConditionEvidence:
    """Build leaf evidence, clearing ``missing_fields`` on a definite outcome."""
    return ConditionEvidence(
        kind=kind,
        description=condition.description,
        profile_fields_used=list(fields),
        outcome=outcome,
        threshold_applied=threshold,
        missing_fields=list(missing) if outcome is Trilean.UNKNOWN else [],
        source_ids=list(condition.source_ids),
    )


def _eval_geography(condition: GeographyCondition, profile: UserProfile) -> ConditionEvidence:
    fields: list[str] = []
    missing: list[str] = []
    outcomes: list[Trilean] = []
    parts: list[str] = []

    if condition.states:
        fields.append("state")
        parts.append("state is one of " + ", ".join(sorted(condition.states)))
        if profile.state is None:
            outcomes.append(Trilean.UNKNOWN)
            missing.append("state")
        else:
            allowed = {_normalize(state) for state in condition.states}
            outcomes.append(Trilean.TRUE if _normalize(profile.state) in allowed else Trilean.FALSE)
    if condition.counties:
        fields.append("county")
        parts.append("county is one of " + ", ".join(sorted(condition.counties)))
        if profile.county is None:
            outcomes.append(Trilean.UNKNOWN)
            missing.append("county")
        else:
            allowed_counties = {_normalize_county(county) for county in condition.counties}
            outcomes.append(
                Trilean.TRUE
                if _normalize_county(profile.county) in allowed_counties
                else Trilean.FALSE
            )

    return _leaf(
        condition,
        "geography",
        resolve_all_of(outcomes),
        fields=fields,
        missing=missing,
        threshold="; ".join(parts),
    )


def _eval_student_status(
    condition: StudentStatusCondition, profile: UserProfile
) -> ConditionEvidence:
    threshold = "student status is one of " + ", ".join(
        sorted(status.value for status in condition.allowed)
    )
    value = profile.student_status
    if value is None or value is StudentStatus.UNKNOWN:
        outcome, missing = Trilean.UNKNOWN, ["student_status"]
    else:
        outcome = Trilean.TRUE if value in condition.allowed else Trilean.FALSE
        missing = []
    return _leaf(
        condition,
        "student_status",
        outcome,
        fields=["student_status"],
        missing=missing,
        threshold=threshold,
    )


def _bounds_label(noun: str, minimum: int | None, maximum: int | None) -> str:
    if minimum is not None and maximum is not None:
        return f"{noun} between {minimum} and {maximum}"
    if minimum is not None:
        return f"{noun} of at least {minimum}"
    return f"{noun} of at most {maximum}"


def _within(value: int, minimum: int | None, maximum: int | None) -> bool:
    if minimum is not None and value < minimum:
        return False
    return not (maximum is not None and value > maximum)


def _eval_household_size(
    condition: HouseholdSizeCondition, profile: UserProfile
) -> ConditionEvidence:
    """Exact size decides; a composition lower bound can still decide some cases.

    ``household_composition`` partitions adults and children, so its sum is a
    sound *lower* bound on household size. A lower bound above the rule's
    maximum is enough for ``FALSE``, and a lower bound that already clears a
    rule with no maximum is enough for ``TRUE``. Anything else stays
    ``UNKNOWN``: the true size could be larger than the counts we were given.
    """
    threshold = _bounds_label("household size", condition.min, condition.max)
    size = profile.household_size
    if size is not None:
        outcome = Trilean.TRUE if _within(size, condition.min, condition.max) else Trilean.FALSE
        return _leaf(
            condition,
            "household_size",
            outcome,
            fields=["household_size"],
            threshold=threshold,
        )

    lower = profile.household_composition.known_minimum()
    fields = ["household_size", "household_composition"]
    if lower is not None:
        if condition.max is not None and lower > condition.max:
            return _leaf(
                condition, "household_size", Trilean.FALSE, fields=fields, threshold=threshold
            )
        if condition.max is None and condition.min is not None and lower >= condition.min:
            return _leaf(
                condition, "household_size", Trilean.TRUE, fields=fields, threshold=threshold
            )
    return _leaf(
        condition,
        "household_size",
        Trilean.UNKNOWN,
        fields=fields,
        missing=["household_size"],
        threshold=threshold,
    )


def _eval_income_fpl(
    condition: IncomeFplCondition, profile: UserProfile, fpl: FplTable
) -> ConditionEvidence:
    """Compare the annualised income range against a percentage of the FPL.

    The comparison is made in annual terms because monthly-to-annual conversion
    is exact (multiply by twelve) while annual-to-monthly is not, so comparing
    annually never widens the range unnecessarily.
    """
    fields = ["household_size", *_INCOME_FIELDS]
    size = profile.household_size
    low, high = profile.income_annual_range_cents()

    if size is None:
        missing = ["household_size"]
        if low is None and high is None:
            missing.extend((_INCOME_MIN, _INCOME_MAX))
        return _leaf(
            condition,
            "income_fpl",
            Trilean.UNKNOWN,
            fields=fields,
            missing=missing,
            threshold=(
                f"{_format_pct(condition.max_pct_fpl)}% of the {condition.fpl_year} "
                "federal poverty guideline for the household's size"
            ),
        )

    limit = fpl.annual_threshold_cents(
        household_size=size, pct=condition.max_pct_fpl, year=condition.fpl_year
    )
    if limit is None:
        return _leaf(
            condition,
            "income_fpl",
            Trilean.UNKNOWN,
            fields=fields,
            missing=["household_size"],
            threshold=(
                f"{FPL_TABLE_UNAVAILABLE} for {condition.fpl_year} and a household of {size}"
            ),
        )

    threshold = (
        f"{_format_pct(condition.max_pct_fpl)}% of the {condition.fpl_year} federal poverty "
        f"guideline for a household of {size}, which is {_format_dollars(limit)} a year"
    )
    outcome, missing = _compare_range_to_limit(low, high, limit)
    return _leaf(
        condition, "income_fpl", outcome, fields=fields, missing=missing, threshold=threshold
    )


def _eval_income_absolute(
    condition: IncomeAbsoluteCondition, profile: UserProfile
) -> ConditionEvidence:
    if condition.max_monthly_cents is not None:
        period = IncomePeriod.MONTHLY
        limit = condition.max_monthly_cents
        low, high = profile.income_monthly_range_cents()
    else:
        period = IncomePeriod.ANNUAL
        # validated by the schema: exactly one bound is set
        limit = condition.max_annual_cents or 0
        low, high = profile.income_annual_range_cents()

    outcome, missing = _compare_range_to_limit(low, high, limit)
    return _leaf(
        condition,
        "income_absolute",
        outcome,
        fields=list(_INCOME_FIELDS),
        missing=missing,
        threshold=f"income of at most {_format_dollars(limit)} {_period_label(period)}",
    )


def _categorical_value(field: CategoricalProfileField, profile: UserProfile) -> str | None:
    raw = getattr(profile, field.value, None)
    if raw is None:
        return None
    value = str(raw)
    return None if value.upper() == "UNKNOWN" else value


def _eval_categorical(condition: CategoricalCondition, profile: UserProfile) -> ConditionEvidence:
    field = condition.field.value
    threshold = f"{field} is one of " + ", ".join(sorted(condition.any_of))
    value = _categorical_value(condition.field, profile)
    if value is None:
        return _leaf(
            condition,
            "categorical",
            Trilean.UNKNOWN,
            fields=[field],
            missing=[field],
            threshold=threshold,
        )
    is_county = condition.field is CategoricalProfileField.COUNTY
    normalise = _normalize_county if is_county else _normalize
    allowed = {normalise(candidate) for candidate in condition.any_of}
    outcome = Trilean.TRUE if normalise(value) in allowed else Trilean.FALSE
    return _leaf(condition, "categorical", outcome, fields=[field], threshold=threshold)


def _eval_age(condition: AgeCondition, profile: UserProfile) -> ConditionEvidence:
    threshold = _bounds_label("age", condition.min, condition.max)
    if profile.age is None:
        return _leaf(
            condition,
            "age",
            Trilean.UNKNOWN,
            fields=["age"],
            missing=["age"],
            threshold=threshold,
        )
    outcome = Trilean.TRUE if _within(profile.age, condition.min, condition.max) else Trilean.FALSE
    return _leaf(condition, "age", outcome, fields=["age"], threshold=threshold)


def _eval_flag(condition: FlagCondition, profile: UserProfile) -> ConditionEvidence:
    field = f"flags.{condition.flag.value}"
    threshold = f"{condition.flag.value} is {str(condition.expected).lower()}"
    value = profile.flag(condition.flag)
    if value is None:
        return _leaf(
            condition,
            "flag",
            Trilean.UNKNOWN,
            fields=[field],
            missing=[field],
            threshold=threshold,
        )
    outcome = Trilean.TRUE if value is condition.expected else Trilean.FALSE
    return _leaf(condition, "flag", outcome, fields=[field], threshold=threshold)


def _blocking_union(children: Sequence[ConditionEvidence]) -> list[str]:
    """Missing fields from the children that are actually holding up the answer."""
    ordered: dict[str, None] = {}
    for child in children:
        if child.outcome is Trilean.UNKNOWN:
            for field in child.missing_fields:
                ordered[field] = None
    return list(ordered)


def evaluate_condition(
    condition: Condition, profile: UserProfile, *, fpl: FplTable | None = None
) -> ConditionEvidence:
    """Evaluate one condition tree and return its full evidence record.

    Pure and total: every condition type resolves, and an unsupported one is a
    type error rather than a silent ``UNKNOWN``.
    """
    table = fpl if fpl is not None else default_fpl_table()

    if isinstance(condition, GeographyCondition):
        return _eval_geography(condition, profile)
    if isinstance(condition, StudentStatusCondition):
        return _eval_student_status(condition, profile)
    if isinstance(condition, HouseholdSizeCondition):
        return _eval_household_size(condition, profile)
    if isinstance(condition, IncomeFplCondition):
        return _eval_income_fpl(condition, profile, table)
    if isinstance(condition, IncomeAbsoluteCondition):
        return _eval_income_absolute(condition, profile)
    if isinstance(condition, CategoricalCondition):
        return _eval_categorical(condition, profile)
    if isinstance(condition, AgeCondition):
        return _eval_age(condition, profile)
    if isinstance(condition, FlagCondition):
        return _eval_flag(condition, profile)
    if isinstance(condition, AllOf | AnyOf):
        children = [evaluate_condition(child, profile, fpl=table) for child in condition.conditions]
        outcomes = [child.outcome for child in children]
        is_all = isinstance(condition, AllOf)
        outcome = resolve_all_of(outcomes) if is_all else resolve_any_of(outcomes)
        return ConditionEvidence(
            kind="all_of" if is_all else "any_of",
            description=condition.description,
            outcome=outcome,
            missing_fields=_blocking_union(children) if outcome is Trilean.UNKNOWN else [],
            source_ids=list(condition.source_ids),
            children=children,
        )
    if isinstance(condition, Not):
        child = evaluate_condition(condition.condition, profile, fpl=table)
        outcome = resolve_not(child.outcome)
        return ConditionEvidence(
            kind="not",
            description=condition.description,
            outcome=outcome,
            missing_fields=_blocking_union([child]) if outcome is Trilean.UNKNOWN else [],
            source_ids=list(condition.source_ids),
            children=[child],
        )
    assert_never(condition)


# ---------------------------------------------------------------------------
# evidence tree inspection
# ---------------------------------------------------------------------------
def walk_evidence(evidence: ConditionEvidence) -> Iterator[ConditionEvidence]:
    """Pre-order traversal of an evidence tree, root first."""
    yield evidence
    for child in evidence.children:
        yield from walk_evidence(child)


def leaf_evidence(evidence: ConditionEvidence) -> Iterator[ConditionEvidence]:
    """Every node that evaluated a profile field rather than combining others."""
    for node in walk_evidence(evidence):
        if not node.children:
            yield node


def _blocking_counts(evidence: ConditionEvidence) -> Counter[str]:
    """How many still-undecided conditions each missing field would unblock."""
    if evidence.outcome is not Trilean.UNKNOWN:
        return Counter()
    if not evidence.children:
        return Counter(dict.fromkeys(evidence.missing_fields, 1))
    counts: Counter[str] = Counter()
    for child in evidence.children:
        counts += _blocking_counts(child)
    return counts


def _rank_missing_fields(counts: Mapping[str, int]) -> list[str]:
    def sort_key(field: str) -> tuple[int, int, str]:
        priority = FIELD_PRIORITY.index(field) if field in FIELD_PRIORITY else len(FIELD_PRIORITY)
        return (-counts[field], priority, field)

    return sorted(counts, key=sort_key)


def fields_read_by(condition: Condition) -> frozenset[str]:
    """Profile fields a condition tree *could* read, regardless of this profile.

    Static, so it answers "would a change to this field ever matter to this
    program" without evaluating anything. Used by
    :meth:`RuleEngine.affected_programs` to keep an edit from re-running every
    program in the catalog.
    """
    if isinstance(condition, AllOf | AnyOf):
        return frozenset().union(*(fields_read_by(child) for child in condition.conditions))
    if isinstance(condition, Not):
        return fields_read_by(condition.condition)
    if isinstance(condition, GeographyCondition):
        fields = {"state"} if condition.states else set()
        if condition.counties:
            fields.add("county")
        return frozenset(fields)
    if isinstance(condition, StudentStatusCondition):
        return frozenset({"student_status"})
    if isinstance(condition, HouseholdSizeCondition):
        return frozenset({"household_size", "household_composition"})
    if isinstance(condition, IncomeFplCondition):
        return frozenset({"household_size", *_INCOME_FIELDS})
    if isinstance(condition, IncomeAbsoluteCondition):
        return frozenset(_INCOME_FIELDS)
    if isinstance(condition, CategoricalCondition):
        return frozenset({condition.field.value})
    if isinstance(condition, AgeCondition):
        return frozenset({"age"})
    if isinstance(condition, FlagCondition):
        return frozenset({f"flags.{condition.flag.value}"})
    assert_never(condition)


def _expand_field(name: str) -> frozenset[str]:
    """Aliases that make a changed-field name comparable with evidence names.

    ``"income"`` is the name the clarification layer uses for the whole range,
    while the engine records the three concrete fields. Treating them as
    unrelated would silently skip re-evaluating income rules after the one
    answer most likely to change a result.
    """
    if name == _INCOME_ALIAS:
        return frozenset({_INCOME_ALIAS, *_INCOME_FIELDS})
    if name in _INCOME_FIELDS:
        return frozenset({_INCOME_ALIAS, name})
    return frozenset({name})


def _expand_fields(names: Sequence[str]) -> frozenset[str]:
    if not names:
        return frozenset()
    return frozenset().union(*(_expand_field(name) for name in names))


# ---------------------------------------------------------------------------
# program evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ProgramRuleset:
    """A program's identity plus the rules to judge it by.

    The engine needs nothing else about a program, so this is the smallest
    input that keeps it a pure function. ``from_candidate`` adapts the
    retrieval layer's :class:`~app.schemas.program.RetrievedProgram`.
    """

    program_id: uuid.UUID
    program_slug: str
    ruleset: RuleSet
    retrieval_score: float | None = None

    @classmethod
    def from_candidate(cls, candidate: RetrievedProgram) -> ProgramRuleset | None:
        """Adapt a retrieval candidate, or ``None`` if it has no usable rules.

        A candidate without an approved rule set is not a non-match: it is a
        program we cannot assess. Returning ``None`` makes the caller decide
        what to tell the user instead of letting a missing rule set look like a
        verdict.
        """
        ruleset = candidate.ruleset
        if ruleset is None or ruleset.review_status is not ReviewStatus.APPROVED:
            return None
        return cls(
            program_id=candidate.program.id,
            program_slug=candidate.program.slug,
            ruleset=ruleset,
            retrieval_score=candidate.retrieval_score,
        )


def _score(outcome: Trilean, root: ConditionEvidence, retrieval_score: float | None) -> float:
    """Ranking aid in ``[0, 1]``. Never an input to ``status``.

    Two ingredients, in order of weight: the verdict (a potential match ranks
    above a program that needs more information, which ranks above an unlikely
    one), and how much of the rule set resolved definitively, so that between
    two ``MORE_INFO_NEEDED`` programs the one that is nearly decided comes
    first. Retrieval relevance contributes a fifth, as a tiebreaker only.
    """
    leaves = list(leaf_evidence(root))
    decided = sum(1 for leaf in leaves if leaf.outcome is not Trilean.UNKNOWN)
    resolved = decided / len(leaves) if leaves else 1.0

    base, span = {
        Trilean.TRUE: (0.70, 0.30),
        Trilean.UNKNOWN: (0.20, 0.30),
        Trilean.FALSE: (0.00, 0.10),
    }[outcome]
    rule_score = base + span * resolved
    blended = 0.8 * rule_score + 0.2 * (retrieval_score or 0.0)
    combined = rule_score if retrieval_score is None else blended
    return round(min(max(combined, 0.0), 1.0), 6)


def _geography_conditions(condition: Condition) -> list[GeographyCondition]:
    """Positive geography constraints in a rule tree.

    ``Not`` subtrees are deliberately not descended into: "you do not live in
    New York" is a geography condition that says nothing about which states the
    program serves, and reading it as one would tell a New Jersey user the rules
    do not cover them.
    """
    if isinstance(condition, GeographyCondition):
        return [condition]
    if isinstance(condition, AllOf | AnyOf):
        found: list[GeographyCondition] = []
        for child in condition.conditions:
            found.extend(_geography_conditions(child))
        return found
    return []


def _limitations(
    ruleset: RuleSet, profile: UserProfile, root: ConditionEvidence, now: datetime
) -> list[str]:
    """Everything the user must be told alongside the status.

    A limitation is not a hedge for its own sake; each one here corresponds to
    a concrete reason this assessment could be wrong.
    """
    found: dict[str, None] = dict.fromkeys(ruleset.limitations)
    today = now.date()

    if ruleset.review_status is not ReviewStatus.APPROVED:
        found[
            "These requirements have not finished human review yet, so treat this as a "
            "rough indication rather than an assessment."
        ] = None
    elif ruleset.reviewed_at is not None and now - ruleset.reviewed_at > REVIEW_DUE_AFTER:
        found[
            "These requirements were last checked more than a year ago, so confirm them "
            f"with the program before you apply (last checked {ruleset.reviewed_at.date()})."
        ] = None

    if not ruleset.is_effective_on(today):
        found[
            "The requirements we have on file were not in force on the date of this "
            "check, so the program may have changed them."
        ] = None

    if not ruleset.source_ids:
        found[
            "We have not yet linked these requirements to an official document, so they "
            "are unverified and should be confirmed with the program."
        ] = None

    geographies = _geography_conditions(ruleset.root)
    if profile.state is not None and geographies:
        states = {_normalize(state) for condition in geographies for state in condition.states}
        if states and _normalize(profile.state) not in states:
            found[
                "The rules we have for this program only cover certain states, and the "
                "state you gave is not one of them, so there may be nothing here for you."
            ] = None

    for node in leaf_evidence(root):
        if node.threshold_applied and node.threshold_applied.startswith(FPL_TABLE_UNAVAILABLE):
            found[
                "We do not have the federal poverty guidelines this requirement refers to, "
                "so we could not check the income limit at all."
            ] = None

    return list(found)


def evaluate_program(
    program_id: uuid.UUID,
    program_slug: str,
    ruleset: RuleSet,
    profile: UserProfile,
    *,
    fpl: FplTable | None = None,
    now: datetime | None = None,
    retrieval_score: float | None = None,
) -> EligibilityEvaluation:
    """Assess one program for one profile.

    Deterministic: the same inputs produce an identical record apart from
    ``evaluated_at``. The status comes only from the root outcome, and
    ``missing_fields`` holds only the fields that could still change it,
    ordered by how many of this program's conditions each one unblocks.
    """
    table = fpl if fpl is not None else default_fpl_table()
    moment = now if now is not None else utcnow()
    root = evaluate_condition(ruleset.root, profile, fpl=table)
    counts = _blocking_counts(root)

    return EligibilityEvaluation(
        program_id=program_id,
        program_slug=program_slug,
        rule_version=ruleset.version,
        status=status_for_outcome(root.outcome),
        outcome=root.outcome,
        score=_score(root.outcome, root, retrieval_score),
        evidence=[root],
        missing_fields=_rank_missing_fields(counts),
        limitations=_limitations(ruleset, profile, root, moment),
        evaluated_at=moment,
    )


def _as_program_ruleset(item: ProgramRuleset | RetrievedProgram) -> ProgramRuleset | None:
    if isinstance(item, ProgramRuleset):
        return item
    return ProgramRuleset.from_candidate(item)


def evaluate_all(
    programs_with_rulesets: Sequence[ProgramRuleset | RetrievedProgram],
    profile: UserProfile,
    *,
    fpl: FplTable | None = None,
    now: datetime | None = None,
) -> list[EligibilityEvaluation]:
    """Assess every program that has usable rules, in the order given.

    Programs without an approved rule set are skipped rather than reported as
    non-matches. Input order is preserved so ranking stays the caller's job.
    """
    table = fpl if fpl is not None else default_fpl_table()
    moment = now if now is not None else utcnow()
    evaluations: list[EligibilityEvaluation] = []
    for item in programs_with_rulesets:
        entry = _as_program_ruleset(item)
        if entry is None:
            continue
        evaluations.append(
            evaluate_program(
                entry.program_id,
                entry.program_slug,
                entry.ruleset,
                profile,
                fpl=table,
                now=moment,
                retrieval_score=entry.retrieval_score,
            )
        )
    return evaluations


def _evidence_fields(evaluation: EligibilityEvaluation) -> frozenset[str]:
    fields: set[str] = set()
    for root in evaluation.evidence:
        for node in walk_evidence(root):
            fields.update(node.profile_fields_used)
            fields.update(node.missing_fields)
    return _expand_fields(sorted(fields))


def reevaluate_affected(
    previous: Sequence[EligibilityEvaluation],
    profile: UserProfile,
    changed_fields: Sequence[str],
    *,
    programs_with_rulesets: Sequence[ProgramRuleset | RetrievedProgram] = (),
    fpl: FplTable | None = None,
    now: datetime | None = None,
) -> list[EligibilityEvaluation]:
    """Recompute only the evaluations a changed field could have moved.

    Pure, order-preserving, and conservative in the right direction: an
    evaluation is recomputed when its recorded evidence *read or missed* one of
    ``changed_fields``, so a program whose rules never looked at household size
    is returned untouched - the same object, not a re-derived copy - when
    household size changes.

    Rule sets must be supplied for the programs that are affected; an affected
    program with no rule set available is a caller bug, not a data condition,
    so it raises rather than quietly returning a stale verdict.
    """
    changed = _expand_fields(changed_fields)
    if not changed:
        return list(previous)

    table = fpl if fpl is not None else default_fpl_table()
    moment = now if now is not None else utcnow()
    by_id: dict[uuid.UUID, ProgramRuleset] = {}
    for item in programs_with_rulesets:
        entry = _as_program_ruleset(item)
        if entry is not None:
            by_id[entry.program_id] = entry

    result: list[EligibilityEvaluation] = []
    for evaluation in previous:
        if not (_evidence_fields(evaluation) & changed):
            result.append(evaluation)
            continue
        entry = by_id.get(evaluation.program_id)
        if entry is None:
            msg = (
                f"no rule set supplied for affected program {evaluation.program_slug!r}; "
                "pass every candidate's rule set to reevaluate_affected"
            )
            raise ValueError(msg)
        result.append(
            evaluate_program(
                entry.program_id,
                entry.program_slug,
                entry.ruleset,
                profile,
                fpl=table,
                now=moment,
                retrieval_score=entry.retrieval_score,
            )
        )
    return result


# ---------------------------------------------------------------------------
# clarification questions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _QuestionTemplate:
    """Author-written copy for one answerable field.

    Questions are templates rather than generated prose for two reasons: a
    question written once and reviewed is better than one assembled per run,
    and it makes "we never ask for something no rule uses" a property of a
    fixed table instead of a hope about a generator.
    """

    question: str
    why_it_matters: str
    input_kind: ClarificationInputKind
    options: tuple[tuple[str, str], ...] = ()
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None


def _enum_options(
    *values: StudentStatus | EmploymentStatus | HousingSituation,
) -> tuple[tuple[str, str], ...]:
    labels = {
        StudentStatus.NOT_A_STUDENT: "I am not a student",
        StudentStatus.UNDERGRAD_FULL_TIME: "Undergraduate, full time",
        StudentStatus.UNDERGRAD_PART_TIME: "Undergraduate, part time",
        StudentStatus.GRADUATE: "Graduate or professional student",
        StudentStatus.OTHER: "Something else",
        EmploymentStatus.EMPLOYED_FULL_TIME: "Working full time",
        EmploymentStatus.EMPLOYED_PART_TIME: "Working part time",
        EmploymentStatus.SELF_EMPLOYED: "Self-employed",
        EmploymentStatus.SEASONAL_OR_GIG: "Seasonal, gig or on-and-off work",
        EmploymentStatus.UNEMPLOYED: "Not working right now",
        EmploymentStatus.UNABLE_TO_WORK: "Unable to work",
        EmploymentStatus.RETIRED: "Retired",
        EmploymentStatus.NOT_IN_LABOR_FORCE: "Not looking for work right now",
        EmploymentStatus.OTHER: "Something else",
        HousingSituation.RENTING: "Renting",
        HousingSituation.OWN_HOME: "I own my home",
        HousingSituation.LIVING_WITH_FAMILY: "Staying with family or friends",
        HousingSituation.UNIVERSITY_HOUSING: "University housing",
        HousingSituation.TEMPORARY_OR_SHELTER: "Temporary housing or a shelter",
        HousingSituation.UNHOUSED: "I do not have housing right now",
        HousingSituation.OTHER: "Something else",
    }
    return tuple((value.value, labels[value]) for value in values)


_YES_NO: Final[tuple[tuple[str, str], ...]] = (("true", "Yes"), ("false", "No"))

#: Plain-language copy for every field a rule can leave unresolved. A field
#: with no entry here is never asked about, which is the structural reason this
#: function cannot ask for a Social Security number: there is no such field in
#: ``PROFILE_FIELD_NAMES`` and no template for one.
_QUESTION_TEMPLATES: Final[dict[str, _QuestionTemplate]] = {
    "state": _QuestionTemplate(
        question="Which state do you live in?",
        why_it_matters="Most of these programs are only open to people living in New Jersey.",
        input_kind=ClarificationInputKind.TEXT,
    ),
    "county": _QuestionTemplate(
        question="Which New Jersey county do you live in?",
        why_it_matters="Some help is run county by county, so this changes what is available.",
        input_kind=ClarificationInputKind.TEXT,
    ),
    "household_size": _QuestionTemplate(
        question="How many people live in your household, counting yourself?",
        why_it_matters="Income limits go up with household size, so this can change the answer.",
        input_kind=ClarificationInputKind.NUMBER,
        unit="people",
        min_value=1,
        max_value=20,
    ),
    _INCOME_ALIAS: _QuestionTemplate(
        question="Roughly how much money does your household bring in each month?",
        why_it_matters=(
            "A rough range is enough. Several programs set a monthly income limit, and "
            "we cannot tell which side of it you are on yet."
        ),
        input_kind=ClarificationInputKind.RANGE,
        unit="dollars per month",
        min_value=0,
    ),
    "age": _QuestionTemplate(
        question="How old are you?",
        why_it_matters="A few programs are only for certain age groups.",
        input_kind=ClarificationInputKind.NUMBER,
        unit="years",
        min_value=0,
        max_value=120,
    ),
    "student_status": _QuestionTemplate(
        question="Are you enrolled as a student right now?",
        why_it_matters="Student rules change which food and emergency aid you can get.",
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=_enum_options(
            StudentStatus.NOT_A_STUDENT,
            StudentStatus.UNDERGRAD_FULL_TIME,
            StudentStatus.UNDERGRAD_PART_TIME,
            StudentStatus.GRADUATE,
            StudentStatus.OTHER,
        ),
    ),
    "employment_status": _QuestionTemplate(
        question="Which of these describes your work right now?",
        why_it_matters="Some programs have a work requirement or a work-based exception.",
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=_enum_options(
            EmploymentStatus.EMPLOYED_FULL_TIME,
            EmploymentStatus.EMPLOYED_PART_TIME,
            EmploymentStatus.SELF_EMPLOYED,
            EmploymentStatus.SEASONAL_OR_GIG,
            EmploymentStatus.UNEMPLOYED,
            EmploymentStatus.UNABLE_TO_WORK,
            EmploymentStatus.RETIRED,
            EmploymentStatus.NOT_IN_LABOR_FORCE,
            EmploymentStatus.OTHER,
        ),
    ),
    "housing_situation": _QuestionTemplate(
        question="Where are you living at the moment?",
        why_it_matters="Housing and utility help depends on your housing situation.",
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=_enum_options(
            HousingSituation.RENTING,
            HousingSituation.OWN_HOME,
            HousingSituation.LIVING_WITH_FAMILY,
            HousingSituation.UNIVERSITY_HOUSING,
            HousingSituation.TEMPORARY_OR_SHELTER,
            HousingSituation.UNHOUSED,
            HousingSituation.OTHER,
        ),
    ),
}

#: Yes/no copy for the flags a rule can test. Only flags that some rule reads
#: can reach this table, and a flag with no entry is skipped rather than asked
#: about awkwardly.
_FLAG_QUESTIONS: Final[dict[ProfileFlag, tuple[str, str]]] = {
    ProfileFlag.RECEIVES_SNAP: (
        "Do you already get SNAP food benefits?",
        "If you already have SNAP, we can skip it and point you at the rest.",
    ),
    ProfileFlag.RECEIVES_WIC: (
        "Do you already get WIC?",
        "WIC changes which other food programs are worth applying for.",
    ),
    ProfileFlag.RECEIVES_MEDICAID: (
        "Do you already have Medicaid or NJ FamilyCare coverage?",
        "If you are already covered, we will not send you through it again.",
    ),
    ProfileFlag.RECEIVES_TANF: (
        "Do you get cash assistance (WorkFirst NJ or TANF)?",
        "Getting cash assistance can qualify you for several other programs automatically.",
    ),
    ProfileFlag.RECEIVES_SSI: (
        "Do you get SSI (Supplemental Security Income) payments?",
        "SSI can qualify you for other programs without a separate income check.",
    ),
    ProfileFlag.RECEIVES_SSDI: (
        "Do you get disability payments (SSDI)?",
        "Disability payments can qualify you for other programs directly.",
    ),
    ProfileFlag.RECEIVES_UNEMPLOYMENT: (
        "Are you getting unemployment payments right now?",
        "Unemployment counts as income for some programs and qualifies you for others.",
    ),
    ProfileFlag.RECEIVES_FREE_SCHOOL_MEALS: (
        "Does a child in your household get free or reduced-price school meals?",
        "That alone can qualify your household for other food help.",
    ),
    ProfileFlag.IS_PREGNANT: (
        "Is anyone in your household pregnant?",
        "Some health and food programs are open to pregnant people at higher incomes.",
    ),
    ProfileFlag.HAS_CHILDREN: (
        "Are there children under 18 in your household?",
        "Programs for families use this, and it can lift a student restriction.",
    ),
    ProfileFlag.IS_VETERAN: (
        "Are you a veteran?",
        "There is help set aside specifically for veterans.",
    ),
    ProfileFlag.IS_DISABLED: (
        "Do you have a disability?",
        "A disability can lift a student work rule and open up other programs.",
    ),
    ProfileFlag.IS_SENIOR: (
        "Are you 60 or older?",
        "Some programs have a separate, easier route for older adults.",
    ),
    ProfileFlag.IS_CAREGIVER: (
        "Do you take care of a child or an adult who needs help?",
        "Caregiving can count instead of a work requirement.",
    ),
    ProfileFlag.IS_RUTGERS_AFFILIATED: (
        "Are you a Rutgers student or staff member?",
        "Rutgers runs its own aid that is only open to its community.",
    ),
    ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE: (
        "Have you had a shut-off notice from a utility company?",
        "A shut-off notice moves you to the front of the queue for utility help.",
    ),
    ProfileFlag.HAS_EVICTION_NOTICE: (
        "Have you been given an eviction notice?",
        "An eviction notice opens up emergency housing help.",
    ),
    ProfileFlag.FILED_TAXES_LAST_YEAR: (
        "Did you file a tax return last year?",
        "Tax credits are claimed through a return, so this decides if they apply.",
    ),
}

_NULL_RUN: Final[uuid.UUID] = uuid.UUID(int=0)


def _canonical_question_field(field: str) -> str | None:
    """Map an engine field name onto the field a question can actually resolve.

    The three income fields collapse to one ``income`` question: asking
    separately for the bottom and the top of a range is two questions for one
    answer, and ``RangeValue`` exists precisely so it can be one.
    """
    if field in _INCOME_FIELDS or field == _INCOME_ALIAS:
        return _INCOME_ALIAS
    if field in _QUESTION_TEMPLATES:
        return field
    if field.startswith("flags."):
        try:
            flag = ProfileFlag(field.removeprefix("flags."))
        except ValueError:
            return None
        return field if flag in _FLAG_QUESTIONS else None
    return None


def _template_for(field: str) -> _QuestionTemplate | None:
    if field in _QUESTION_TEMPLATES:
        return _QUESTION_TEMPLATES[field]
    if field.startswith("flags."):
        flag = ProfileFlag(field.removeprefix("flags."))
        question, why = _FLAG_QUESTIONS[flag]
        return _QuestionTemplate(
            question=question,
            why_it_matters=why,
            input_kind=ClarificationInputKind.SINGLE_SELECT,
            options=_YES_NO,
        )
    return None


def _is_safe(template: _QuestionTemplate) -> bool:
    haystack = f"{template.question} {template.why_it_matters}".casefold()
    return not any(term in haystack for term in SENSITIVE_TERMS)


def missing_field_questions(
    evaluations: Sequence[EligibilityEvaluation],
    *,
    run_id: uuid.UUID | None = None,
    limit: int = MAX_CLARIFICATION_QUESTIONS,
) -> list[ClarificationQuestion]:
    """Derive clarification questions from unresolved conditions, deterministically.

    Ranking is by how many programs an answer would unblock, then by how many
    individual conditions, then by a fixed field order, so the list is stable
    for a given set of evaluations. Only fields that a rule actually left
    ``UNKNOWN`` can appear, so a question is never asked unless an answer could
    change a result, and nothing outside ``PROFILE_FIELD_NAMES`` is askable at
    all - which is what guarantees we never ask for a Social Security number,
    bank details or an identifier no program needs.
    """
    programs: dict[str, set[str]] = {}
    conditions: Counter[str] = Counter()

    for evaluation in evaluations:
        for field in evaluation.missing_fields:
            canonical = _canonical_question_field(field)
            if canonical is None:
                continue
            programs.setdefault(canonical, set()).add(str(evaluation.program_id))
        for root in evaluation.evidence:
            for name, count in _blocking_counts(root).items():
                canonical = _canonical_question_field(name)
                if canonical is not None:
                    conditions[canonical] += count

    def sort_key(field: str) -> tuple[int, int, int, str]:
        priority = FIELD_PRIORITY.index(field) if field in FIELD_PRIORITY else len(FIELD_PRIORITY)
        return (-len(programs[field]), -conditions[field], priority, field)

    questions: list[ClarificationQuestion] = []
    for field in sorted(programs, key=sort_key):
        template = _template_for(field)
        if template is None or not _is_safe(template):
            continue
        questions.append(
            ClarificationQuestion(
                id=clarification_question_id(run_id or _NULL_RUN, field),
                field=field,
                question=template.question,
                input_kind=template.input_kind,
                options=[
                    ClarificationOption(value=value, label=label)
                    for value, label in template.options
                ],
                why_it_matters=template.why_it_matters,
                affected_program_ids=sorted(programs[field]),
                is_required=False,
                unit=template.unit,
                min_value=template.min_value,
                max_value=template.max_value,
            )
        )
        if len(questions) >= limit:
            break
    return questions


# ---------------------------------------------------------------------------
# the contract implementation
# ---------------------------------------------------------------------------
class RuleEngine:
    """``app.schemas.contracts.EligibilityEngine`` over the module functions.

    Holds the FPL table and nothing else: no session, no clock, no cache. Safe
    to share across concurrent tasks because it has no mutable state.
    """

    __slots__ = ("_fpl",)

    def __init__(self, fpl: FplTable | None = None) -> None:
        self._fpl = fpl if fpl is not None else default_fpl_table()

    @property
    def fpl(self) -> FplTable:
        return self._fpl

    def evaluate(
        self,
        profile: UserProfile,
        program: AssistanceProgram,
        ruleset: RuleSet,
        *,
        now: datetime | None = None,
        retrieval_score: float | None = None,
    ) -> EligibilityEvaluation:
        return evaluate_program(
            program.id,
            program.slug,
            ruleset,
            profile,
            fpl=self._fpl,
            now=now,
            retrieval_score=retrieval_score,
        )

    def evaluate_many(
        self,
        profile: UserProfile,
        candidates: Sequence[RetrievedProgram],
        *,
        now: datetime | None = None,
    ) -> list[EligibilityEvaluation]:
        return evaluate_all(list(candidates), profile, fpl=self._fpl, now=now)

    def affected_programs(
        self,
        changed_fields: Sequence[str],
        candidates: Sequence[RetrievedProgram],
    ) -> list[uuid.UUID]:
        """Which candidates' rules could read one of ``changed_fields``.

        Static over the rule trees, so it is usable before anything has been
        evaluated. Order follows ``candidates`` and duplicates are collapsed.
        """
        changed = _expand_fields(changed_fields)
        if not changed:
            return []
        affected: dict[uuid.UUID, None] = {}
        for candidate in candidates:
            entry = ProgramRuleset.from_candidate(candidate)
            if entry is None:
                continue
            if _expand_fields(sorted(fields_read_by(entry.ruleset.root))) & changed:
                affected[entry.program_id] = None
        return list(affected)

    def reevaluate_affected(
        self,
        previous: Sequence[EligibilityEvaluation],
        profile: UserProfile,
        changed_fields: Sequence[str],
        *,
        candidates: Sequence[RetrievedProgram] = (),
        now: datetime | None = None,
    ) -> list[EligibilityEvaluation]:
        return reevaluate_affected(
            previous,
            profile,
            changed_fields,
            programs_with_rulesets=list(candidates),
            fpl=self._fpl,
            now=now,
        )

    def missing_field_questions(
        self,
        evaluations: Sequence[EligibilityEvaluation],
        *,
        run_id: uuid.UUID | None = None,
        limit: int = MAX_CLARIFICATION_QUESTIONS,
    ) -> list[ClarificationQuestion]:
        return missing_field_questions(evaluations, run_id=run_id, limit=limit)
