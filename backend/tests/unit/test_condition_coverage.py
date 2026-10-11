"""The completeness guard for the rule engine's condition vocabulary.

``tests/unit/test_eligibility_rules.py`` tests each condition in depth. This
file asks the narrower question that an in-depth file cannot answer: *is every
condition kind in the contract covered at all, on all three of its outcomes?*

The mechanism is a single table, :data:`MATRIX`, holding one condition per
``ConditionKind`` together with the three profiles that drive it to ``TRUE``,
``FALSE`` and ``UNKNOWN``. Two things follow from building it that way:

* adding a kind to ``app.schemas.eligibility.ConditionKind`` fails
  ``test_every_condition_kind_in_the_contract_is_covered`` until somebody adds
  it here, so the vocabulary cannot grow past its tests;
* every kind is asserted on all three outcomes by a parametrised test whose id
  names the kind and the outcome, so a gap is visible in the test report rather
  than inferred from a coverage percentage.

The second half is about ``missing_fields`` being *exact*. The engine narrows
the schema's "union of missing fields" to the blocking ones, because every
entry becomes a question a person is asked. So the assertions are set equalities
in both directions: nothing that caused an ``UNKNOWN`` is dropped, and nothing
that no rule reads is ever asked for.
"""

from __future__ import annotations

import typing
import uuid
from datetime import UTC, date, datetime

import pytest

from app.eligibility.engine import (
    evaluate_condition,
    evaluate_program,
    fields_read_by,
    leaf_evidence,
    walk_evidence,
)
from app.eligibility.fpl import default_fpl_table
from app.schemas import (
    AgeCondition,
    AllOf,
    AnyOf,
    CategoricalCondition,
    Condition,
    ConditionEvidence,
    EligibilityEvaluation,
    FlagCondition,
    GeographyCondition,
    HouseholdSizeCondition,
    HousingSituation,
    IncomeAbsoluteCondition,
    IncomeFplCondition,
    IncomePeriod,
    MatchStatus,
    Not,
    ProfileFlag,
    ReviewStatus,
    RuleSet,
    StudentStatus,
    StudentStatusCondition,
    Trilean,
    UserProfile,
    program_uuid,
)
from app.schemas.eligibility import ConditionKind
from app.schemas.profile import PROFILE_FIELD_NAMES

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
FPL_YEAR = 2026

TRUE = Trilean.TRUE
FALSE = Trilean.FALSE
UNKNOWN = Trilean.UNKNOWN

#: Every kind the contract declares. Read off the ``Literal`` rather than
#: retyped, so the two cannot drift.
ALL_KINDS: frozenset[str] = frozenset(typing.get_args(ConditionKind.__value__))


def _fpl_profile(income_min: int, income_max: int) -> UserProfile:
    return UserProfile(
        household_size=4,
        income_min_cents=income_min,
        income_max_cents=income_max,
        income_period=IncomePeriod.ANNUAL,
    )


#: Annual FPL for a household of four, as the shipped table has it. Used to
#: build an income range that sits clearly under, clearly over, or across a
#: 130% threshold without hard-coding a dollar figure that a table update
#: would silently invalidate.
_FPL_4 = default_fpl_table().annual_threshold_cents(household_size=4, pct=130.0, year=FPL_YEAR)
assert _FPL_4 is not None, "the shipped FPL table must cover a household of four"

#: kind -> (condition, profile that resolves TRUE, FALSE, UNKNOWN)
MATRIX: dict[str, tuple[Condition, UserProfile, UserProfile, UserProfile]] = {
    "geography": (
        GeographyCondition(description="You live in New Jersey", states=["NJ"]),
        UserProfile(state="NJ"),
        UserProfile(state="NY"),
        UserProfile(),
    ),
    "student_status": (
        StudentStatusCondition(
            description="You are enrolled at least part time",
            allowed=[StudentStatus.UNDERGRAD_FULL_TIME, StudentStatus.UNDERGRAD_PART_TIME],
        ),
        UserProfile(student_status=StudentStatus.UNDERGRAD_FULL_TIME),
        UserProfile(student_status=StudentStatus.NOT_A_STUDENT),
        UserProfile(student_status=StudentStatus.UNKNOWN),
    ),
    "household_size": (
        HouseholdSizeCondition(description="Your household is two to five people", min=2, max=5),
        UserProfile(household_size=3),
        UserProfile(household_size=9),
        UserProfile(),
    ),
    "income_fpl": (
        IncomeFplCondition(
            description="Your income is at or below 130% of the federal poverty level",
            max_pct_fpl=130.0,
            fpl_year=FPL_YEAR,
            household_definition="Everyone who buys and prepares food together.",
        ),
        _fpl_profile(_FPL_4 - 200_000, _FPL_4 - 100_000),
        _fpl_profile(_FPL_4 + 100_000, _FPL_4 + 200_000),
        _fpl_profile(_FPL_4 - 100_000, _FPL_4 + 100_000),
    ),
    "income_absolute": (
        IncomeAbsoluteCondition(
            description="You earn no more than $2,000 a month", max_monthly_cents=200_000
        ),
        UserProfile(
            income_min_cents=100_000, income_max_cents=150_000, income_period=IncomePeriod.MONTHLY
        ),
        UserProfile(
            income_min_cents=300_000, income_max_cents=400_000, income_period=IncomePeriod.MONTHLY
        ),
        UserProfile(
            income_min_cents=150_000, income_max_cents=250_000, income_period=IncomePeriod.MONTHLY
        ),
    ),
    "categorical": (
        CategoricalCondition(
            description="You rent your home",
            field="housing_situation",
            any_of=[HousingSituation.RENTING.value],
        ),
        UserProfile(housing_situation=HousingSituation.RENTING),
        UserProfile(housing_situation=HousingSituation.OWN_HOME),
        UserProfile(housing_situation=HousingSituation.UNKNOWN),
    ),
    "age": (
        AgeCondition(description="You are 60 or older", min=60),
        UserProfile(age=72),
        UserProfile(age=31),
        UserProfile(),
    ),
    "flag": (
        FlagCondition(description="You already receive SNAP", flag=ProfileFlag.RECEIVES_SNAP),
        UserProfile(flags={ProfileFlag.RECEIVES_SNAP: True}),
        UserProfile(flags={ProfileFlag.RECEIVES_SNAP: False}),
        UserProfile(),
    ),
    "all_of": (
        AllOf(
            description="You live in New Jersey and you are 60 or older",
            conditions=[
                GeographyCondition(description="You live in New Jersey", states=["NJ"]),
                AgeCondition(description="You are 60 or older", min=60),
            ],
        ),
        UserProfile(state="NJ", age=72),
        UserProfile(state="NY", age=72),
        UserProfile(state="NJ"),
    ),
    "any_of": (
        AnyOf(
            description="You are 60 or older, or you already receive SNAP",
            conditions=[
                AgeCondition(description="You are 60 or older", min=60),
                FlagCondition(
                    description="You already receive SNAP", flag=ProfileFlag.RECEIVES_SNAP
                ),
            ],
        ),
        UserProfile(age=72, flags={ProfileFlag.RECEIVES_SNAP: False}),
        UserProfile(age=31, flags={ProfileFlag.RECEIVES_SNAP: False}),
        UserProfile(age=31),
    ),
    "not": (
        Not(
            description="You are not already receiving SNAP",
            condition=FlagCondition(
                description="You already receive SNAP", flag=ProfileFlag.RECEIVES_SNAP
            ),
        ),
        UserProfile(flags={ProfileFlag.RECEIVES_SNAP: False}),
        UserProfile(flags={ProfileFlag.RECEIVES_SNAP: True}),
        UserProfile(),
    ),
}

#: Position of each outcome in a :data:`MATRIX` row.
_SLOT: dict[Trilean, int] = {TRUE: 1, FALSE: 2, UNKNOWN: 3}

CASES = [(kind, outcome) for kind in sorted(MATRIX) for outcome in (TRUE, FALSE, UNKNOWN)]


def condition_for(kind: str) -> Condition:
    return MATRIX[kind][0]


def profile_for(kind: str, outcome: Trilean) -> UserProfile:
    """The profile that drives ``kind`` to ``outcome``."""
    row = MATRIX[kind]
    profile = row[_SLOT[outcome]]
    assert isinstance(profile, UserProfile), f"the {kind!r} row is malformed"
    return profile


def resolve(kind: str, outcome: Trilean) -> ConditionEvidence:
    return evaluate_condition(
        condition_for(kind), profile_for(kind, outcome), fpl=default_fpl_table()
    )


def evaluate(condition: Condition, profile: UserProfile) -> EligibilityEvaluation:
    ruleset = RuleSet(
        version="2026.1",
        program_slug="coverage-matrix-program",
        effective_from=date(2026, 1, 1),
        root=condition,
        source_ids=["test-source"],
        review_status=ReviewStatus.APPROVED,
        reviewed_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    return evaluate_program(
        program_uuid("coverage-matrix-program"),
        "coverage-matrix-program",
        ruleset,
        profile,
        now=NOW,
    )


def blocking_unknown_fields(evaluation: EligibilityEvaluation) -> set[str]:
    """The fields that actually caused the root to be ``UNKNOWN``.

    Walked from the root down the ``UNKNOWN`` spine: a branch the engine already
    decided cannot be unblocked by answering a question, so its missing fields
    are not the user's problem.
    """
    found: set[str] = set()

    def walk(node: ConditionEvidence) -> None:
        if node.outcome is not UNKNOWN:
            return
        if not node.children:
            found.update(node.missing_fields)
            return
        for child in node.children:
            walk(child)

    for root in evaluation.evidence:
        walk(root)
    return found


# ---------------------------------------------------------------------------
# the vocabulary is covered
# ---------------------------------------------------------------------------
def test_every_condition_kind_in_the_contract_is_covered() -> None:
    """A new condition kind must arrive with its three outcomes.

    This is the test that makes the rest of the file a guarantee rather than a
    sample. If it fails, the fix is a row in :data:`MATRIX`, not an exclusion.
    """
    assert set(MATRIX) == ALL_KINDS, (
        "MATRIX and ConditionKind disagree; "
        f"uncovered: {sorted(ALL_KINDS - set(MATRIX))}, "
        f"not in the contract: {sorted(set(MATRIX) - ALL_KINDS)}"
    )


def test_the_matrix_rows_name_the_kind_they_hold() -> None:
    for kind in MATRIX:
        held = condition_for(kind)
        assert held.kind == kind, f"the {kind!r} row holds a {held.kind!r} condition"


@pytest.mark.parametrize(("kind", "expected"), CASES, ids=[f"{k}-{o}" for k, o in CASES])
def test_condition_resolves_to_each_trilean(kind: str, expected: Trilean) -> None:
    """Every kind, on every outcome, with the evidence recording what it read."""
    evidence = resolve(kind, expected)
    assert evidence.outcome is expected
    assert evidence.kind == kind
    assert evidence.description, "evidence must carry the author-written requirement text"


@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_a_definite_outcome_reports_no_missing_fields(kind: str) -> None:
    """Deciding a condition and still asking for data would be a double ask."""
    for outcome in (TRUE, FALSE):
        evidence = resolve(kind, outcome)
        for node in walk_evidence(evidence):
            if node.outcome is not UNKNOWN:
                assert node.missing_fields == [], f"{kind}/{outcome.value}: {node.missing_fields}"


@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_an_unknown_leaf_names_a_real_profile_field(kind: str) -> None:
    """An ``UNKNOWN`` that cannot be acted on is worse than no answer."""
    evidence = resolve(kind, UNKNOWN)
    leaves = [node for node in leaf_evidence(evidence) if node.outcome is UNKNOWN]
    assert leaves, f"{kind} resolved UNKNOWN with no UNKNOWN leaf underneath it"
    for leaf in leaves:
        assert leaf.missing_fields, f"{kind}: an UNKNOWN leaf named no field"
        for field in leaf.missing_fields:
            assert field in PROFILE_FIELD_NAMES, f"{kind}: {field!r} is not a profile field"


@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_a_missing_field_never_produces_an_unlikely_match(kind: str) -> None:
    evaluation = evaluate(condition_for(kind), profile_for(kind, UNKNOWN))
    assert evaluation.status is MatchStatus.MORE_INFO_NEEDED, (
        "a field nobody has supplied must never discourage somebody"
    )


@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_the_three_statuses_follow_the_three_outcomes(kind: str) -> None:
    expected = {
        TRUE: MatchStatus.POTENTIAL_MATCH,
        FALSE: MatchStatus.UNLIKELY_MATCH,
        UNKNOWN: MatchStatus.MORE_INFO_NEEDED,
    }
    for outcome, status in expected.items():
        evaluation = evaluate(condition_for(kind), profile_for(kind, outcome))
        assert evaluation.outcome is outcome
        assert evaluation.status is status


# ---------------------------------------------------------------------------
# missing fields are exactly the blocking ones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_missing_fields_is_exactly_the_set_that_caused_the_unknown(kind: str) -> None:
    evaluation = evaluate(condition_for(kind), profile_for(kind, UNKNOWN))
    assert set(evaluation.missing_fields) == blocking_unknown_fields(evaluation)
    assert len(evaluation.missing_fields) == len(set(evaluation.missing_fields)), (
        "a field asked for twice is asked for twice in the UI"
    )


@pytest.mark.parametrize("kind", sorted(MATRIX))
def test_no_field_is_asked_for_that_the_rules_do_not_read(kind: str) -> None:
    readable = set(fields_read_by(condition_for(kind)))
    for outcome in (TRUE, FALSE, UNKNOWN):
        evaluation = evaluate(condition_for(kind), profile_for(kind, outcome))
        assert set(evaluation.missing_fields) <= readable, (
            f"{kind}/{outcome.value} asked for {sorted(set(evaluation.missing_fields) - readable)}"
        )


def test_a_decided_branch_does_not_contribute_its_unknowns() -> None:
    """A satisfied alternative hides the sibling nobody needs to answer."""
    condition = AnyOf(
        description="You are 60 or older, or you already receive SNAP",
        conditions=[
            AgeCondition(description="You are 60 or older", min=60),
            FlagCondition(description="You already receive SNAP", flag=ProfileFlag.RECEIVES_SNAP),
        ],
    )
    evaluation = evaluate(condition, UserProfile(age=72))

    assert evaluation.status is MatchStatus.POTENTIAL_MATCH
    assert evaluation.missing_fields == []
    buried = {
        field
        for node in leaf_evidence(evaluation.evidence[0])
        for field in node.missing_fields
        if node.outcome is UNKNOWN
    }
    assert buried == {"flags.receives_snap"}, (
        "the unknown leaf is still recorded as evidence, just not asked about"
    )


def test_a_failed_requirement_does_not_turn_into_a_question() -> None:
    """Asking for more detail about a rule the person already fails is cruel."""
    condition = AllOf(
        description="You live in New Jersey and you are 60 or older",
        conditions=[
            GeographyCondition(description="You live in New Jersey", states=["NJ"]),
            AgeCondition(description="You are 60 or older", min=60),
        ],
    )
    evaluation = evaluate(condition, UserProfile(state="NY"))

    assert evaluation.status is MatchStatus.UNLIKELY_MATCH
    assert evaluation.missing_fields == []


def test_two_blocking_unknowns_are_both_asked_about() -> None:
    condition = AllOf(
        description="Your household and your income both have to be checked",
        conditions=[
            HouseholdSizeCondition(description="Your household is two or more people", min=2),
            IncomeAbsoluteCondition(
                description="You earn no more than $2,000 a month", max_monthly_cents=200_000
            ),
        ],
    )
    evaluation = evaluate(condition, UserProfile())

    assert evaluation.status is MatchStatus.MORE_INFO_NEEDED
    assert set(evaluation.missing_fields) == {
        "household_size",
        "income_min_cents",
        "income_max_cents",
    }
    assert set(evaluation.missing_fields) == blocking_unknown_fields(evaluation)


def test_the_evaluation_record_is_reproducible() -> None:
    """Same profile, same rules, same table - byte-identical apart from the clock."""
    condition = condition_for("income_fpl")
    unknown_profile = profile_for("income_fpl", UNKNOWN)
    first = evaluate(condition, unknown_profile)
    second = evaluate(condition, unknown_profile)
    assert first.model_dump(exclude={"evaluated_at"}) == second.model_dump(exclude={"evaluated_at"})
    assert isinstance(first.program_id, uuid.UUID)
