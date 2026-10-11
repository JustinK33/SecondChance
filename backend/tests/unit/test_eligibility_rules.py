"""Tests for the deterministic rule engine.

These matter more than the engine's code does. Every assertion here is about a
sentence someone will read on a results page, so they are written against the
documented semantics rather than against the implementation:

* the three-valued truth tables, including nesting, exhaustively;
* every leaf condition on its satisfied, violated and unknown-input paths;
* the income range comparison, especially a range that straddles a threshold,
  in both monthly and annual periods;
* that a missing field never produces ``UNLIKELY_MATCH``;
* that selective re-evaluation leaves unaffected programs byte-identical;
* that every author-written clarification string avoids the banned status
  vocabulary, which model output was already checked against;
* and 22 hand-labelled scenarios from ``tests/fixtures/eligibility_scenarios.json``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from app.ai.schemas import BANNED_STATUS_VOCABULARY
from app.eligibility.engine import (
    _FLAG_QUESTIONS,
    _QUESTION_TEMPLATES,
    _YES_NO,
    FPL_TABLE_UNAVAILABLE,
    ProgramRuleset,
    RuleEngine,
    evaluate_all,
    evaluate_condition,
    evaluate_program,
    fields_read_by,
    reevaluate_affected,
)
from app.eligibility.fpl import FplTable, default_fpl_table, fpl_threshold_cents
from app.schemas import (
    AgeCondition,
    AllOf,
    AnyOf,
    AssistanceProgram,
    CategoricalCondition,
    CategoricalProfileField,
    Condition,
    ConditionEvidence,
    EligibilityEvaluation,
    EmploymentStatus,
    FlagCondition,
    GeographyCondition,
    GeoScope,
    HouseholdComposition,
    HouseholdSizeCondition,
    HousingSituation,
    IncomeAbsoluteCondition,
    IncomeFplCondition,
    IncomePeriod,
    MatchStatus,
    Not,
    ProfileFlag,
    ProgramCategory,
    RetrievedProgram,
    ReviewStatus,
    RuleSet,
    StudentStatus,
    StudentStatusCondition,
    Trilean,
    UserProfile,
    program_uuid,
)
from app.schemas.profile import PROFILE_FIELD_NAMES

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "eligibility_scenarios.json"

TRUE = Trilean.TRUE
FALSE = Trilean.FALSE
UNKNOWN = Trilean.UNKNOWN


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def leaf_for(outcome: Trilean) -> Condition:
    """A flag condition that resolves to ``outcome`` against :data:`LOGIC_PROFILE`.

    Using three different flags rather than three different profiles keeps one
    profile for the whole truth table, so a combinator test cannot accidentally
    pass because two children saw different inputs.
    """
    flag = {
        TRUE: ProfileFlag.IS_VETERAN,
        FALSE: ProfileFlag.IS_DISABLED,
        UNKNOWN: ProfileFlag.IS_SENIOR,
    }[outcome]
    return FlagCondition(description=f"Leaf that resolves {outcome.value}", flag=flag)


#: ``is_veteran`` is true, ``is_disabled`` is false, ``is_senior`` was never stated.
LOGIC_PROFILE = UserProfile(flags={ProfileFlag.IS_VETERAN: True, ProfileFlag.IS_DISABLED: False})


def resolve(condition: Condition, profile: UserProfile | None = None) -> Trilean:
    return evaluate_condition(
        condition, profile if profile is not None else LOGIC_PROFILE, fpl=default_fpl_table()
    ).outcome


def all_of(*outcomes: Trilean) -> Condition:
    return AllOf(
        description="Every one of these must be true",
        conditions=[leaf_for(outcome) for outcome in outcomes],
    )


def any_of(*outcomes: Trilean) -> Condition:
    return AnyOf(
        description="At least one of these must be true",
        conditions=[leaf_for(outcome) for outcome in outcomes],
    )


def negate(outcome: Trilean) -> Condition:
    return Not(description="This must not be true", condition=leaf_for(outcome))


def approved_ruleset(
    root: Condition,
    *,
    slug: str = "test-program",
    version: str = "2026.1",
    effective_from: date = date(2026, 1, 1),
    effective_to: date | None = None,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
    reviewed_at: datetime | None = datetime(2026, 9, 1, tzinfo=UTC),
    source_ids: list[str] | None = None,
    limitations: list[str] | None = None,
) -> RuleSet:
    return RuleSet(
        version=version,
        program_slug=slug,
        effective_from=effective_from,
        effective_to=effective_to,
        root=root,
        source_ids=["test-source"] if source_ids is None else source_ids,
        review_status=review_status,
        reviewed_at=reviewed_at,
        limitations=limitations or [],
    )


def program(slug: str = "test-program", *, states: list[str] | None = None) -> AssistanceProgram:
    return AssistanceProgram(
        id=program_uuid(slug),
        slug=slug,
        official_name=f"Official name for {slug}",
        administering_org="A New Jersey state department",
        category=ProgramCategory.FOOD,
        description="A factual description of a test program.",
        plain_language_summary="Help with something, for testing.",
        geo_scope=GeoScope(states=states or ["NJ"], label="All of New Jersey"),
        official_url="https://example.nj.gov/program",
    )


def candidate(
    slug: str,
    root: Condition,
    *,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
    retrieval_score: float = 0.0,
    with_ruleset: bool = True,
) -> RetrievedProgram:
    ruleset = (
        approved_ruleset(root, slug=slug, review_status=review_status) if with_ruleset else None
    )
    return RetrievedProgram(program=program(slug), ruleset=ruleset, retrieval_score=retrieval_score)


def evaluate(
    root: Condition, profile: UserProfile, *, slug: str = "test-program", **kwargs: object
) -> EligibilityEvaluation:
    ruleset = approved_ruleset(root, slug=slug, **kwargs)  # type: ignore[arg-type]
    return evaluate_program(program_uuid(slug), slug, ruleset, profile, now=NOW)


def missing_everywhere(evidence: ConditionEvidence) -> set[str]:
    found = set(evidence.missing_fields)
    for child in evidence.children:
        found |= missing_everywhere(child)
    return found


# ---------------------------------------------------------------------------
# three-valued logic
# ---------------------------------------------------------------------------
class TestTruthTables:
    """``AllOf``/``AnyOf``/``Not`` against every combination of TRUE/FALSE/UNKNOWN.

    The expected values are written out longhand rather than computed, so this
    is a check of the documented semantics and not a restatement of the code.
    """

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (TRUE, TRUE, TRUE),
            (TRUE, FALSE, FALSE),
            (TRUE, UNKNOWN, UNKNOWN),
            (FALSE, TRUE, FALSE),
            (FALSE, FALSE, FALSE),
            (FALSE, UNKNOWN, FALSE),
            (UNKNOWN, TRUE, UNKNOWN),
            (UNKNOWN, FALSE, FALSE),
            (UNKNOWN, UNKNOWN, UNKNOWN),
        ],
    )
    def test_all_of(self, left: Trilean, right: Trilean, expected: Trilean) -> None:
        assert resolve(all_of(left, right)) is expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (TRUE, TRUE, TRUE),
            (TRUE, FALSE, TRUE),
            (TRUE, UNKNOWN, TRUE),
            (FALSE, TRUE, TRUE),
            (FALSE, FALSE, FALSE),
            (FALSE, UNKNOWN, UNKNOWN),
            (UNKNOWN, TRUE, TRUE),
            (UNKNOWN, FALSE, UNKNOWN),
            (UNKNOWN, UNKNOWN, UNKNOWN),
        ],
    )
    def test_any_of(self, left: Trilean, right: Trilean, expected: Trilean) -> None:
        assert resolve(any_of(left, right)) is expected

    @pytest.mark.parametrize(
        ("child", "expected"),
        [(TRUE, FALSE), (FALSE, TRUE), (UNKNOWN, UNKNOWN)],
    )
    def test_not_preserves_unknown(self, child: Trilean, expected: Trilean) -> None:
        """Not knowing whether something holds is not knowing that it does not."""
        assert resolve(negate(child)) is expected

    def test_single_child_combinators_are_the_identity(self) -> None:
        for outcome in Trilean:
            assert resolve(all_of(outcome)) is outcome
            assert resolve(any_of(outcome)) is outcome

    @pytest.mark.parametrize(
        ("outcomes", "expected"),
        [
            ((TRUE, TRUE, TRUE), TRUE),
            ((TRUE, TRUE, UNKNOWN), UNKNOWN),
            ((TRUE, FALSE, UNKNOWN), FALSE),
            ((UNKNOWN, UNKNOWN, FALSE), FALSE),
            ((UNKNOWN, UNKNOWN, UNKNOWN), UNKNOWN),
        ],
    )
    def test_all_of_three_children(self, outcomes: tuple[Trilean, ...], expected: Trilean) -> None:
        assert resolve(all_of(*outcomes)) is expected

    @pytest.mark.parametrize(
        ("outcomes", "expected"),
        [
            ((FALSE, FALSE, FALSE), FALSE),
            ((FALSE, FALSE, UNKNOWN), UNKNOWN),
            ((FALSE, TRUE, UNKNOWN), TRUE),
            ((UNKNOWN, UNKNOWN, TRUE), TRUE),
            ((UNKNOWN, UNKNOWN, UNKNOWN), UNKNOWN),
        ],
    )
    def test_any_of_three_children(self, outcomes: tuple[Trilean, ...], expected: Trilean) -> None:
        assert resolve(any_of(*outcomes)) is expected

    @pytest.mark.parametrize("first", list(Trilean))
    @pytest.mark.parametrize("second", list(Trilean))
    @pytest.mark.parametrize("third", list(Trilean))
    def test_order_does_not_change_the_outcome(
        self, first: Trilean, second: Trilean, third: Trilean
    ) -> None:
        """A rule set's wording order must not change anyone's result."""
        forward = (first, second, third)
        backward = tuple(reversed(forward))
        assert resolve(all_of(*forward)) is resolve(all_of(*backward))
        assert resolve(any_of(*forward)) is resolve(any_of(*backward))


class TestNesting:
    def test_all_of_containing_any_of(self) -> None:
        condition = AllOf(
            description="You live here and meet one of the exceptions",
            conditions=[leaf_for(TRUE), any_of(FALSE, UNKNOWN)],
        )
        assert resolve(condition) is UNKNOWN

    def test_all_of_containing_a_satisfied_any_of(self) -> None:
        condition = AllOf(
            description="You live here and meet one of the exceptions",
            conditions=[leaf_for(TRUE), any_of(FALSE, TRUE, UNKNOWN)],
        )
        assert resolve(condition) is TRUE

    def test_failed_hard_requirement_beats_nested_unknowns(self) -> None:
        condition = AllOf(
            description="A hard requirement plus a pile of unknowns",
            conditions=[leaf_for(FALSE), any_of(UNKNOWN, UNKNOWN), all_of(UNKNOWN, TRUE)],
        )
        assert resolve(condition) is FALSE

    def test_not_of_an_any_of(self) -> None:
        assert resolve(Not(description="None of these", condition=any_of(FALSE, FALSE))) is TRUE
        assert resolve(Not(description="None of these", condition=any_of(FALSE, TRUE))) is FALSE
        assert (
            resolve(Not(description="None of these", condition=any_of(FALSE, UNKNOWN))) is UNKNOWN
        )

    def test_double_negation_round_trips(self) -> None:
        for outcome in Trilean:
            doubled = Not(description="Not not this", condition=negate(outcome))
            assert resolve(doubled) is outcome

    def test_three_levels_deep(self) -> None:
        condition = AllOf(
            description="Top level",
            conditions=[
                leaf_for(TRUE),
                AnyOf(
                    description="Middle level",
                    conditions=[
                        AllOf(
                            description="Bottom level",
                            conditions=[leaf_for(TRUE), leaf_for(TRUE)],
                        ),
                        leaf_for(FALSE),
                    ],
                ),
            ],
        )
        assert resolve(condition) is TRUE

    def test_evidence_mirrors_the_condition_tree(self) -> None:
        condition = AllOf(
            description="Top level",
            conditions=[leaf_for(TRUE), any_of(FALSE, UNKNOWN)],
        )
        evidence = evaluate_condition(condition, LOGIC_PROFILE, fpl=default_fpl_table())
        assert evidence.kind == "all_of"
        assert [child.kind for child in evidence.children] == ["flag", "any_of"]
        assert len(evidence.children[1].children) == 2


class TestOnlyBlockingUnknownsAreReported:
    def test_satisfied_alternative_hides_its_unknown_sibling(self) -> None:
        evidence = evaluate_condition(any_of(TRUE, UNKNOWN), LOGIC_PROFILE, fpl=default_fpl_table())
        assert evidence.outcome is TRUE
        assert evidence.missing_fields == []
        # the unknown is still in the tree, for audit
        assert missing_everywhere(evidence) == {"flags.is_senior"}

    def test_failed_requirement_hides_unknowns(self) -> None:
        evidence = evaluate_condition(
            all_of(FALSE, UNKNOWN), LOGIC_PROFILE, fpl=default_fpl_table()
        )
        assert evidence.outcome is FALSE
        assert evidence.missing_fields == []

    def test_blocking_unknown_is_reported(self) -> None:
        evidence = evaluate_condition(all_of(TRUE, UNKNOWN), LOGIC_PROFILE, fpl=default_fpl_table())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["flags.is_senior"]

    def test_unknown_leaf_always_names_a_real_profile_field(self) -> None:
        evidence = evaluate_condition(leaf_for(UNKNOWN), LOGIC_PROFILE)
        assert evidence.missing_fields == ["flags.is_senior"]
        assert set(evidence.missing_fields) <= PROFILE_FIELD_NAMES


# ---------------------------------------------------------------------------
# leaf conditions
# ---------------------------------------------------------------------------
class TestGeographyCondition:
    STATE_RULE = GeographyCondition(description="You live in New Jersey", states=["NJ"])
    COUNTY_RULE = GeographyCondition(
        description="You live in Middlesex or Essex County", counties=["Middlesex", "Essex"]
    )
    BOTH_RULE = GeographyCondition(
        description="You live in Middlesex County, New Jersey",
        states=["NJ"],
        counties=["Middlesex"],
    )

    def test_satisfied(self) -> None:
        assert resolve(self.STATE_RULE, UserProfile(state="NJ")) is TRUE

    def test_state_is_compared_case_insensitively(self) -> None:
        assert resolve(self.STATE_RULE, UserProfile(state="nj")) is TRUE

    def test_violated(self) -> None:
        assert resolve(self.STATE_RULE, UserProfile(state="PA")) is FALSE

    def test_unknown_state_is_not_out_of_state(self) -> None:
        evidence = evaluate_condition(self.STATE_RULE, UserProfile())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["state"]
        assert evidence.profile_fields_used == ["state"]

    def test_county_suffix_and_case_are_ignored(self) -> None:
        assert resolve(self.COUNTY_RULE, UserProfile(county="Middlesex County")) is TRUE
        assert resolve(self.COUNTY_RULE, UserProfile(county="middlesex")) is TRUE

    def test_county_violated(self) -> None:
        assert resolve(self.COUNTY_RULE, UserProfile(county="Camden")) is FALSE

    def test_county_unknown(self) -> None:
        evidence = evaluate_condition(self.COUNTY_RULE, UserProfile(state="NJ"))
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["county"]

    def test_both_levels_must_hold(self) -> None:
        assert resolve(self.BOTH_RULE, UserProfile(state="NJ", county="Middlesex")) is TRUE
        assert resolve(self.BOTH_RULE, UserProfile(state="NJ", county="Camden")) is FALSE

    def test_definite_state_failure_short_circuits_an_unknown_county(self) -> None:
        evidence = evaluate_condition(self.BOTH_RULE, UserProfile(state="PA"))
        assert evidence.outcome is FALSE
        assert evidence.missing_fields == []

    def test_partial_knowledge_reports_only_what_is_missing(self) -> None:
        evidence = evaluate_condition(self.BOTH_RULE, UserProfile(state="NJ"))
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["county"]

    def test_threshold_is_recorded(self) -> None:
        evidence = evaluate_condition(self.STATE_RULE, UserProfile(state="NJ"))
        assert evidence.threshold_applied == "state is one of NJ"


class TestStudentStatusCondition:
    RULE = StudentStatusCondition(
        description="You are not a student",
        allowed=[StudentStatus.NOT_A_STUDENT, StudentStatus.OTHER],
    )

    def test_satisfied(self) -> None:
        profile = UserProfile(student_status=StudentStatus.NOT_A_STUDENT)
        assert resolve(self.RULE, profile) is TRUE

    def test_violated(self) -> None:
        profile = UserProfile(student_status=StudentStatus.GRADUATE)
        assert resolve(self.RULE, profile) is FALSE

    def test_absent_is_unknown(self) -> None:
        evidence = evaluate_condition(self.RULE, UserProfile())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["student_status"]

    def test_explicit_unknown_enum_member_is_unknown(self) -> None:
        """``StudentStatus.UNKNOWN`` means "we asked and did not learn", so it
        must behave exactly like an absent value and never like a mismatch."""
        profile = UserProfile(student_status=StudentStatus.UNKNOWN)
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["student_status"]


class TestHouseholdSizeCondition:
    RANGE_RULE = HouseholdSizeCondition(
        description="Your household has two to four people", min=2, max=4
    )
    MIN_RULE = HouseholdSizeCondition(description="At least two people", min=2)
    MAX_RULE = HouseholdSizeCondition(description="At most two people", max=2)

    @pytest.mark.parametrize(("size", "expected"), [(1, FALSE), (2, TRUE), (4, TRUE), (5, FALSE)])
    def test_inclusive_bounds(self, size: int, expected: Trilean) -> None:
        assert resolve(self.RANGE_RULE, UserProfile(household_size=size)) is expected

    def test_absent_is_unknown(self) -> None:
        evidence = evaluate_condition(self.RANGE_RULE, UserProfile())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["household_size"]

    def test_composition_lower_bound_can_satisfy_an_open_ended_minimum(self) -> None:
        """Two adults and a child is at least three people whatever else we learn."""
        profile = UserProfile(
            household_composition=HouseholdComposition(adults=2, children_under_18=1)
        )
        evidence = evaluate_condition(self.MIN_RULE, profile)
        assert evidence.outcome is TRUE
        assert "household_composition" in evidence.profile_fields_used

    def test_composition_lower_bound_can_violate_a_maximum(self) -> None:
        profile = UserProfile(
            household_composition=HouseholdComposition(adults=2, children_under_18=3)
        )
        assert resolve(self.MAX_RULE, profile) is FALSE

    def test_composition_lower_bound_inside_a_bounded_range_stays_unknown(self) -> None:
        """A lower bound of two tells us nothing about a rule with a maximum:
        the real household could be five people."""
        profile = UserProfile(
            household_composition=HouseholdComposition(adults=1, children_under_18=1)
        )
        evidence = evaluate_condition(self.RANGE_RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["household_size"]

    def test_exact_size_wins_over_composition(self) -> None:
        profile = UserProfile(
            household_size=5,
            household_composition=HouseholdComposition(adults=1, children_under_18=1),
        )
        assert resolve(self.RANGE_RULE, profile) is FALSE


class TestIncomeFplCondition:
    RULE = IncomeFplCondition(
        description="Your income is at or below 185% of the federal poverty level",
        max_pct_fpl=185.0,
        fpl_year=2026,
        household_definition="Everyone who buys and prepares food together",
    )
    # 185% of the 2026 guideline for a household of two is $40,034.00
    LIMIT_HH2 = 4_003_400

    def test_threshold_matches_the_fpl_table(self) -> None:
        assert fpl_threshold_cents(2026, 2, 185.0) == self.LIMIT_HH2

    def test_entire_range_below_is_true(self) -> None:
        profile = UserProfile(
            household_size=2,
            income_min_cents=1_000_000,
            income_max_cents=2_000_000,
            income_period=IncomePeriod.ANNUAL,
        )
        assert resolve(self.RULE, profile) is TRUE

    def test_exactly_at_the_limit_is_true(self) -> None:
        """ "At or below" is inclusive, and a borderline person should be
        encouraged to apply rather than discouraged."""
        profile = UserProfile(
            household_size=2,
            income_min_cents=self.LIMIT_HH2,
            income_max_cents=self.LIMIT_HH2,
            income_period=IncomePeriod.ANNUAL,
        )
        assert resolve(self.RULE, profile) is TRUE

    def test_one_cent_over_is_false(self) -> None:
        profile = UserProfile(
            household_size=2,
            income_min_cents=self.LIMIT_HH2 + 1,
            income_max_cents=self.LIMIT_HH2 + 1,
            income_period=IncomePeriod.ANNUAL,
        )
        assert resolve(self.RULE, profile) is FALSE

    def test_straddling_range_is_unknown_and_asks_for_both_bounds(self) -> None:
        profile = UserProfile(
            household_size=2,
            income_min_cents=3_600_000,
            income_max_cents=6_000_000,
            income_period=IncomePeriod.ANNUAL,
        )
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_min_cents", "income_max_cents"]

    def test_monthly_income_is_annualised(self) -> None:
        """$3,000 a month is $36,000 a year, which is under the $40,034 limit."""
        profile = UserProfile(
            household_size=2,
            income_min_cents=300_000,
            income_max_cents=300_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert resolve(self.RULE, profile) is TRUE

    def test_monthly_straddle(self) -> None:
        profile = UserProfile(
            household_size=2,
            income_min_cents=300_000,
            income_max_cents=500_000,
            income_period=IncomePeriod.MONTHLY,
        )
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_min_cents", "income_max_cents"]

    def test_no_income_at_all_is_unknown(self) -> None:
        evidence = evaluate_condition(self.RULE, UserProfile(household_size=2))
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_min_cents", "income_max_cents"]

    def test_missing_household_size_is_unknown(self) -> None:
        profile = UserProfile(
            income_min_cents=100_000,
            income_max_cents=100_000,
            income_period=IncomePeriod.MONTHLY,
        )
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["household_size"]

    def test_missing_household_size_and_income_asks_for_both(self) -> None:
        evidence = evaluate_condition(self.RULE, UserProfile(state="NJ"))
        assert evidence.missing_fields == [
            "household_size",
            "income_min_cents",
            "income_max_cents",
        ]

    def test_floor_only_above_the_limit_is_false(self) -> None:
        profile = UserProfile(
            household_size=2,
            income_min_cents=self.LIMIT_HH2 + 100,
            income_period=IncomePeriod.ANNUAL,
        )
        assert resolve(self.RULE, profile) is FALSE

    def test_floor_only_below_the_limit_asks_for_the_ceiling(self) -> None:
        profile = UserProfile(
            household_size=2, income_min_cents=1_000_000, income_period=IncomePeriod.ANNUAL
        )
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_max_cents"]

    def test_ceiling_only_above_the_limit_asks_for_the_floor(self) -> None:
        profile = UserProfile(
            household_size=2, income_max_cents=9_000_000, income_period=IncomePeriod.ANNUAL
        )
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_min_cents"]

    def test_unavailable_table_is_unknown_not_a_guess(self) -> None:
        rule = IncomeFplCondition(
            description="Your income is at or below 185% of the federal poverty level",
            max_pct_fpl=185.0,
            fpl_year=2099,
            household_definition="Everyone in the home",
        )
        profile = UserProfile(
            household_size=2,
            income_min_cents=1_000,
            income_max_cents=1_000,
            income_period=IncomePeriod.ANNUAL,
        )
        evidence = evaluate_condition(rule, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.threshold_applied is not None
        assert evidence.threshold_applied.startswith(FPL_TABLE_UNAVAILABLE)

    def test_threshold_is_recorded_in_plain_language(self) -> None:
        profile = UserProfile(
            household_size=2,
            income_min_cents=1_000_000,
            income_max_cents=1_000_000,
            income_period=IncomePeriod.ANNUAL,
        )
        evidence = evaluate_condition(self.RULE, profile)
        assert evidence.threshold_applied == (
            "185% of the 2026 federal poverty guideline for a household of 2, "
            "which is $40,034 a year"
        )


class TestIncomeAbsoluteCondition:
    MONTHLY_RULE = IncomeAbsoluteCondition(
        description="Your household brings in $4,000 a month or less",
        max_monthly_cents=400_000,
    )
    ANNUAL_RULE = IncomeAbsoluteCondition(
        description="Your household brings in $48,000 a year or less",
        max_annual_cents=4_800_000,
    )

    def test_monthly_below(self) -> None:
        profile = UserProfile(
            income_min_cents=100_000,
            income_max_cents=200_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert resolve(self.MONTHLY_RULE, profile) is TRUE

    def test_monthly_above(self) -> None:
        profile = UserProfile(
            income_min_cents=500_000,
            income_max_cents=600_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert resolve(self.MONTHLY_RULE, profile) is FALSE

    def test_monthly_straddle_is_unknown(self) -> None:
        profile = UserProfile(
            income_min_cents=300_000,
            income_max_cents=500_000,
            income_period=IncomePeriod.MONTHLY,
        )
        evidence = evaluate_condition(self.MONTHLY_RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_min_cents", "income_max_cents"]

    def test_annual_straddle_is_unknown(self) -> None:
        profile = UserProfile(
            income_min_cents=4_000_000,
            income_max_cents=6_000_000,
            income_period=IncomePeriod.ANNUAL,
        )
        evidence = evaluate_condition(self.ANNUAL_RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["income_min_cents", "income_max_cents"]

    def test_annual_income_against_a_monthly_rule_rounds_outward(self) -> None:
        """$48,000 a year is $4,000 a month exactly, so it is at the limit.

        The conversion floors the minimum and ceilings the maximum, which can
        only widen the range - a rounding artefact must never produce a FALSE.
        """
        profile = UserProfile(
            income_min_cents=4_800_000,
            income_max_cents=4_800_000,
            income_period=IncomePeriod.ANNUAL,
        )
        assert resolve(self.MONTHLY_RULE, profile) is TRUE

        one_dollar_over = UserProfile(
            income_min_cents=4_800_100,
            income_max_cents=4_800_100,
            income_period=IncomePeriod.ANNUAL,
        )
        assert resolve(self.MONTHLY_RULE, one_dollar_over) is FALSE

    def test_monthly_income_against_an_annual_rule(self) -> None:
        profile = UserProfile(
            income_min_cents=300_000,
            income_max_cents=300_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert resolve(self.ANNUAL_RULE, profile) is TRUE

    def test_no_income_information_is_unknown(self) -> None:
        for rule in (self.MONTHLY_RULE, self.ANNUAL_RULE):
            evidence = evaluate_condition(rule, UserProfile(state="NJ"))
            assert evidence.outcome is UNKNOWN
            assert evidence.missing_fields == ["income_min_cents", "income_max_cents"]

    def test_threshold_records_the_period(self) -> None:
        profile = UserProfile(
            income_min_cents=1, income_max_cents=1, income_period=IncomePeriod.MONTHLY
        )
        monthly = evaluate_condition(self.MONTHLY_RULE, profile)
        annual = evaluate_condition(self.ANNUAL_RULE, profile)
        assert monthly.threshold_applied == "income of at most $4,000 a month"
        assert annual.threshold_applied == "income of at most $48,000 a year"


class TestCategoricalCondition:
    EMPLOYMENT_RULE = CategoricalCondition(
        description="You are working",
        field=CategoricalProfileField.EMPLOYMENT_STATUS,
        any_of=[
            EmploymentStatus.EMPLOYED_FULL_TIME.value,
            EmploymentStatus.EMPLOYED_PART_TIME.value,
        ],
    )
    HOUSING_RULE = CategoricalCondition(
        description="You rent your home",
        field=CategoricalProfileField.HOUSING_SITUATION,
        any_of=[HousingSituation.RENTING.value],
    )
    COUNTY_RULE = CategoricalCondition(
        description="You live in Middlesex County",
        field=CategoricalProfileField.COUNTY,
        any_of=["Middlesex"],
    )
    STATE_RULE = CategoricalCondition(
        description="You live in New Jersey",
        field=CategoricalProfileField.STATE,
        any_of=["NJ"],
    )

    def test_satisfied(self) -> None:
        profile = UserProfile(employment_status=EmploymentStatus.EMPLOYED_PART_TIME)
        assert resolve(self.EMPLOYMENT_RULE, profile) is TRUE

    def test_violated(self) -> None:
        profile = UserProfile(employment_status=EmploymentStatus.RETIRED)
        assert resolve(self.EMPLOYMENT_RULE, profile) is FALSE

    def test_absent_is_unknown(self) -> None:
        evidence = evaluate_condition(self.EMPLOYMENT_RULE, UserProfile())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["employment_status"]

    def test_unknown_enum_member_is_unknown(self) -> None:
        profile = UserProfile(employment_status=EmploymentStatus.UNKNOWN)
        evidence = evaluate_condition(self.EMPLOYMENT_RULE, profile)
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["employment_status"]

    def test_housing_field(self) -> None:
        renting = UserProfile(housing_situation=HousingSituation.RENTING)
        unhoused = UserProfile(housing_situation=HousingSituation.UNHOUSED)
        assert resolve(self.HOUSING_RULE, renting) is TRUE
        assert resolve(self.HOUSING_RULE, unhoused) is FALSE

    def test_county_normalisation(self) -> None:
        assert resolve(self.COUNTY_RULE, UserProfile(county="middlesex county")) is TRUE
        assert resolve(self.COUNTY_RULE, UserProfile(county="Essex")) is FALSE

    def test_state_is_case_insensitive(self) -> None:
        assert resolve(self.STATE_RULE, UserProfile(state="nj")) is TRUE


class TestAgeCondition:
    ADULT_RULE = AgeCondition(description="You are 18 or older", min=18)
    SENIOR_RULE = AgeCondition(description="You are 60 or older", min=60)
    CHILD_RULE = AgeCondition(description="You are under 19", max=18)

    @pytest.mark.parametrize(("age", "expected"), [(17, FALSE), (18, TRUE), (90, TRUE)])
    def test_minimum_is_inclusive(self, age: int, expected: Trilean) -> None:
        assert resolve(self.ADULT_RULE, UserProfile(age=age)) is expected

    @pytest.mark.parametrize(("age", "expected"), [(18, TRUE), (19, FALSE)])
    def test_maximum_is_inclusive(self, age: int, expected: Trilean) -> None:
        assert resolve(self.CHILD_RULE, UserProfile(age=age)) is expected

    def test_absent_is_unknown(self) -> None:
        evidence = evaluate_condition(self.SENIOR_RULE, UserProfile())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["age"]

    def test_zero_is_a_real_age_not_a_missing_one(self) -> None:
        assert resolve(self.CHILD_RULE, UserProfile(age=0)) is TRUE


class TestFlagCondition:
    RULE = FlagCondition(description="You have children under 18", flag=ProfileFlag.HAS_CHILDREN)
    NEGATIVE_RULE = FlagCondition(
        description="You do not already get SNAP",
        flag=ProfileFlag.RECEIVES_SNAP,
        expected=False,
    )

    def test_satisfied(self) -> None:
        profile = UserProfile(flags={ProfileFlag.HAS_CHILDREN: True})
        assert resolve(self.RULE, profile) is TRUE

    def test_violated(self) -> None:
        profile = UserProfile(flags={ProfileFlag.HAS_CHILDREN: False})
        assert resolve(self.RULE, profile) is FALSE

    def test_never_stated_is_unknown_not_false(self) -> None:
        evidence = evaluate_condition(self.RULE, UserProfile())
        assert evidence.outcome is UNKNOWN
        assert evidence.missing_fields == ["flags.has_children"]

    def test_expected_false_inverts_the_comparison(self) -> None:
        not_receiving = UserProfile(flags={ProfileFlag.RECEIVES_SNAP: False})
        receiving = UserProfile(flags={ProfileFlag.RECEIVES_SNAP: True})
        assert resolve(self.NEGATIVE_RULE, not_receiving) is TRUE
        assert resolve(self.NEGATIVE_RULE, receiving) is FALSE

    def test_expected_false_with_an_unstated_flag_is_still_unknown(self) -> None:
        assert resolve(self.NEGATIVE_RULE, UserProfile()) is UNKNOWN


# ---------------------------------------------------------------------------
# program evaluation
# ---------------------------------------------------------------------------
class TestEvaluateProgram:
    @pytest.mark.parametrize(
        ("outcome", "status"),
        [
            (TRUE, MatchStatus.POTENTIAL_MATCH),
            (UNKNOWN, MatchStatus.MORE_INFO_NEEDED),
            (FALSE, MatchStatus.UNLIKELY_MATCH),
        ],
    )
    def test_status_comes_only_from_the_outcome(
        self, outcome: Trilean, status: MatchStatus
    ) -> None:
        evaluation = evaluate(leaf_for(outcome), LOGIC_PROFILE)
        assert evaluation.outcome is outcome
        assert evaluation.status is status

    def test_records_the_rule_version_not_latest(self) -> None:
        evaluation = evaluate(leaf_for(TRUE), LOGIC_PROFILE, version="2026.4")
        assert evaluation.rule_version == "2026.4"

    def test_evidence_is_the_whole_tree(self) -> None:
        evaluation = evaluate(all_of(TRUE, UNKNOWN), LOGIC_PROFILE)
        assert len(evaluation.evidence) == 1
        assert len(evaluation.evidence[0].children) == 2

    def test_missing_fields_are_ranked_by_how_many_conditions_they_unblock(self) -> None:
        """``household_size`` unblocks two conditions here and ``age`` one, so
        the question worth asking first comes first."""
        root = AllOf(
            description="Several rules that need overlapping facts",
            conditions=[
                HouseholdSizeCondition(description="At least two people", min=2),
                IncomeFplCondition(
                    description="Income at or below 200% of the federal poverty level",
                    max_pct_fpl=200.0,
                    fpl_year=2026,
                    household_definition="Everyone in the home",
                ),
                AgeCondition(description="You are 18 or older", min=18),
            ],
        )
        profile = UserProfile(
            income_min_cents=100_000,
            income_max_cents=100_000,
            income_period=IncomePeriod.MONTHLY,
        )
        evaluation = evaluate(root, profile)
        assert evaluation.missing_fields == ["household_size", "age"]

    def test_missing_fields_are_deduplicated(self) -> None:
        root = AllOf(
            description="Two rules that need the same fact",
            conditions=[
                AgeCondition(description="You are 18 or older", min=18),
                AgeCondition(description="You are under 65", max=64),
            ],
        )
        evaluation = evaluate(root, UserProfile(state="NJ"))
        assert evaluation.missing_fields == ["age"]

    def test_more_info_needed_always_carries_a_question_to_ask(self) -> None:
        evaluation = evaluate(all_of(TRUE, UNKNOWN, UNKNOWN), LOGIC_PROFILE)
        assert evaluation.status is MatchStatus.MORE_INFO_NEEDED
        assert evaluation.missing_fields

    def test_unlikely_match_does_not_ask_pointless_questions(self) -> None:
        evaluation = evaluate(all_of(FALSE, UNKNOWN), LOGIC_PROFILE)
        assert evaluation.status is MatchStatus.UNLIKELY_MATCH
        assert evaluation.missing_fields == []

    def test_scores_rank_potential_above_more_info_above_unlikely(self) -> None:
        potential = evaluate(leaf_for(TRUE), LOGIC_PROFILE)
        more_info = evaluate(leaf_for(UNKNOWN), LOGIC_PROFILE)
        unlikely = evaluate(leaf_for(FALSE), LOGIC_PROFILE)
        assert potential.score > more_info.score > unlikely.score

    def test_a_nearly_decided_program_outranks_a_wholly_undecided_one(self) -> None:
        nearly = evaluate(all_of(TRUE, TRUE, TRUE, UNKNOWN), LOGIC_PROFILE)
        wholly = evaluate(all_of(UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN), LOGIC_PROFILE)
        assert nearly.status is wholly.status is MatchStatus.MORE_INFO_NEEDED
        assert nearly.score > wholly.score

    def test_retrieval_score_only_moves_the_ranking_aid(self) -> None:
        base = evaluate_program(
            program_uuid("pp"),
            "pp",
            approved_ruleset(leaf_for(TRUE), slug="pp"),
            LOGIC_PROFILE,
            now=NOW,
        )
        relevant = evaluate_program(
            program_uuid("pp"),
            "pp",
            approved_ruleset(leaf_for(TRUE), slug="pp"),
            LOGIC_PROFILE,
            now=NOW,
            retrieval_score=1.0,
        )
        assert base.status is relevant.status
        assert 0.0 <= relevant.score <= 1.0

    def test_identical_inputs_produce_an_identical_record(self) -> None:
        ruleset = approved_ruleset(all_of(TRUE, UNKNOWN))
        profile = UserProfile(flags=dict(LOGIC_PROFILE.flags))
        first = evaluate_program(program_uuid("pp"), "pp", ruleset, profile, now=NOW)
        second = evaluate_program(program_uuid("pp"), "pp", ruleset, profile, now=NOW)
        assert first.model_dump(mode="json") == second.model_dump(mode="json")


class TestLimitations:
    def test_rule_set_limitations_are_carried_through(self) -> None:
        evaluation = evaluate(
            leaf_for(TRUE),
            LOGIC_PROFILE,
            limitations=["This program also requires an interview we cannot check."],
        )
        assert "This program also requires an interview we cannot check." in evaluation.limitations

    def test_unreviewed_rules_are_flagged(self) -> None:
        evaluation = evaluate(
            leaf_for(TRUE),
            LOGIC_PROFILE,
            review_status=ReviewStatus.PENDING_REVIEW,
            reviewed_at=None,
        )
        assert any("human review" in text for text in evaluation.limitations)

    def test_rules_reviewed_over_a_year_ago_are_flagged(self) -> None:
        evaluation = evaluate(
            leaf_for(TRUE),
            LOGIC_PROFILE,
            reviewed_at=NOW - timedelta(days=400),
            effective_from=date(2024, 1, 1),
        )
        assert any("more than a year ago" in text for text in evaluation.limitations)

    def test_expired_rules_are_flagged(self) -> None:
        evaluation = evaluate(
            leaf_for(TRUE),
            LOGIC_PROFILE,
            effective_from=date(2025, 1, 1),
            effective_to=date(2025, 6, 1),
        )
        assert any("not in force" in text for text in evaluation.limitations)

    def test_unsourced_rules_are_flagged_as_unverified(self) -> None:
        evaluation = evaluate(leaf_for(TRUE), LOGIC_PROFILE, source_ids=[])
        assert any("unverified" in text for text in evaluation.limitations)

    def test_out_of_state_profile_is_told_the_rules_do_not_cover_it(self) -> None:
        root = AllOf(
            description="You live in New Jersey",
            conditions=[GeographyCondition(description="You live in New Jersey", states=["NJ"])],
        )
        evaluation = evaluate(root, UserProfile(state="PA"))
        assert evaluation.status is MatchStatus.UNLIKELY_MATCH
        assert any("not one of them" in text for text in evaluation.limitations)

    def test_a_negated_geography_rule_is_not_read_as_a_service_area(self) -> None:
        """ "You do not live in New York" says nothing about which states the
        program serves, so a New Jersey user must not be warned off."""
        root = AllOf(
            description="You live in New Jersey and not in New York",
            conditions=[
                GeographyCondition(description="You live in New Jersey", states=["NJ"]),
                Not(
                    description="You do not live in New York",
                    condition=GeographyCondition(description="You live in New York", states=["NY"]),
                ),
            ],
        )
        evaluation = evaluate(root, UserProfile(state="NJ"))
        assert evaluation.status is MatchStatus.POTENTIAL_MATCH
        assert evaluation.limitations == []

    def test_in_state_profile_gets_no_geography_limitation(self) -> None:
        root = AllOf(
            description="You live in New Jersey",
            conditions=[GeographyCondition(description="You live in New Jersey", states=["NJ"])],
        )
        evaluation = evaluate(root, UserProfile(state="NJ"))
        assert evaluation.limitations == []

    def test_unavailable_fpl_table_is_an_admitted_gap(self) -> None:
        root = IncomeFplCondition(
            description="Income at or below 185% of the federal poverty level",
            max_pct_fpl=185.0,
            fpl_year=2099,
            household_definition="Everyone in the home",
        )
        profile = UserProfile(
            household_size=2,
            income_min_cents=1_000,
            income_max_cents=1_000,
            income_period=IncomePeriod.ANNUAL,
        )
        evaluation = evaluate(root, profile)
        assert evaluation.status is MatchStatus.MORE_INFO_NEEDED
        assert any("federal poverty guidelines" in text for text in evaluation.limitations)

    def test_limitations_are_not_duplicated(self) -> None:
        evaluation = evaluate(
            leaf_for(TRUE),
            LOGIC_PROFILE,
            source_ids=[],
            limitations=[
                "We have not yet linked these requirements to an official document, so they "
                "are unverified and should be confirmed with the program."
            ],
        )
        assert len(evaluation.limitations) == len(set(evaluation.limitations))


class TestEvaluateAll:
    def test_preserves_input_order(self) -> None:
        candidates = [
            candidate("program-a", leaf_for(TRUE)),
            candidate("program-b", leaf_for(FALSE)),
            candidate("program-c", leaf_for(UNKNOWN)),
        ]
        evaluations = evaluate_all(candidates, LOGIC_PROFILE, now=NOW)
        assert [item.program_slug for item in evaluations] == [
            "program-a",
            "program-b",
            "program-c",
        ]

    def test_skips_candidates_with_no_rules_rather_than_failing_them(self) -> None:
        """No rules means "we cannot assess", which must not look like a verdict."""
        candidates = [
            candidate("program-a", leaf_for(TRUE)),
            candidate("program-b", leaf_for(TRUE), with_ruleset=False),
        ]
        evaluations = evaluate_all(candidates, LOGIC_PROFILE, now=NOW)
        assert [item.program_slug for item in evaluations] == ["program-a"]

    def test_skips_candidates_whose_rules_are_not_approved(self) -> None:
        candidates = [
            candidate("program-a", leaf_for(TRUE), review_status=ReviewStatus.DRAFT),
            candidate("program-b", leaf_for(TRUE), review_status=ReviewStatus.PENDING_REVIEW),
            candidate("program-c", leaf_for(TRUE)),
        ]
        evaluations = evaluate_all(candidates, LOGIC_PROFILE, now=NOW)
        assert [item.program_slug for item in evaluations] == ["program-c"]

    def test_engine_class_matches_the_module_function(self) -> None:
        engine = RuleEngine(default_fpl_table())
        candidates = [candidate("program-a", all_of(TRUE, UNKNOWN))]
        from_class = engine.evaluate_many(LOGIC_PROFILE, candidates, now=NOW)
        from_function = evaluate_all(candidates, LOGIC_PROFILE, now=NOW)
        assert [item.model_dump(mode="json") for item in from_class] == [
            item.model_dump(mode="json") for item in from_function
        ]

    def test_engine_evaluate_takes_a_program(self) -> None:
        engine = RuleEngine()
        ruleset = approved_ruleset(leaf_for(TRUE), slug="program-a")
        evaluation = engine.evaluate(LOGIC_PROFILE, program("program-a"), ruleset, now=NOW)
        assert evaluation.program_id == program_uuid("program-a")
        assert evaluation.status is MatchStatus.POTENTIAL_MATCH


# ---------------------------------------------------------------------------
# selective re-evaluation
# ---------------------------------------------------------------------------
class TestFieldsReadBy:
    def test_collects_fields_through_combinators(self) -> None:
        root = AllOf(
            description="A mixed rule set",
            conditions=[
                GeographyCondition(description="In NJ", states=["NJ"], counties=["Essex"]),
                Not(
                    description="Not already on SNAP",
                    condition=FlagCondition(
                        description="Already on SNAP", flag=ProfileFlag.RECEIVES_SNAP
                    ),
                ),
                AnyOf(
                    description="Either of these",
                    conditions=[
                        AgeCondition(description="60 or older", min=60),
                        IncomeAbsoluteCondition(
                            description="Under $1,000 a month", max_monthly_cents=100_000
                        ),
                    ],
                ),
            ],
        )
        assert fields_read_by(root) == {
            "state",
            "county",
            "flags.receives_snap",
            "age",
            "income_min_cents",
            "income_max_cents",
            "income_period",
        }

    def test_geography_with_only_states_does_not_read_county(self) -> None:
        condition = GeographyCondition(description="In NJ", states=["NJ"])
        assert fields_read_by(condition) == {"state"}

    def test_fpl_condition_reads_household_size(self) -> None:
        condition = IncomeFplCondition(
            description="Under 185%",
            max_pct_fpl=185.0,
            fpl_year=2026,
            household_definition="Everyone in the home",
        )
        assert "household_size" in fields_read_by(condition)


class TestAffectedPrograms:
    def test_only_programs_whose_rules_read_the_field(self) -> None:
        engine = RuleEngine()
        candidates = [
            candidate("reads-age", AgeCondition(description="60 or older", min=60)),
            candidate(
                "reads-household",
                HouseholdSizeCondition(description="At least two people", min=2),
            ),
        ]
        assert engine.affected_programs(["age"], candidates) == [program_uuid("reads-age")]
        assert engine.affected_programs(["household_size"], candidates) == [
            program_uuid("reads-household")
        ]

    def test_income_alias_matches_the_concrete_income_fields(self) -> None:
        engine = RuleEngine()
        candidates = [
            candidate(
                "reads-income",
                IncomeAbsoluteCondition(
                    description="Under $1,000 a month", max_monthly_cents=100_000
                ),
            )
        ]
        assert engine.affected_programs(["income"], candidates) == [program_uuid("reads-income")]
        assert engine.affected_programs(["income_max_cents"], candidates) == [
            program_uuid("reads-income")
        ]

    def test_no_changed_fields_affects_nothing(self) -> None:
        engine = RuleEngine()
        candidates = [candidate("reads-age", AgeCondition(description="60 or older", min=60))]
        assert engine.affected_programs([], candidates) == []

    def test_candidates_without_approved_rules_are_never_affected(self) -> None:
        engine = RuleEngine()
        candidates = [
            candidate(
                "draft",
                AgeCondition(description="60 or older", min=60),
                review_status=ReviewStatus.DRAFT,
            )
        ]
        assert engine.affected_programs(["age"], candidates) == []


class TestReevaluateAffected:
    @staticmethod
    def setup_three() -> tuple[list[RetrievedProgram], list[EligibilityEvaluation]]:
        candidates = [
            candidate("reads-age", AgeCondition(description="You are 60 or older", min=60)),
            candidate(
                "reads-household",
                HouseholdSizeCondition(description="At least two people", min=2),
            ),
            candidate(
                "reads-flag",
                FlagCondition(description="You are a veteran", flag=ProfileFlag.IS_VETERAN),
            ),
        ]
        profile = UserProfile(flags={ProfileFlag.IS_VETERAN: True})
        return candidates, evaluate_all(candidates, profile, now=NOW)

    def test_unaffected_evaluations_are_returned_untouched(self) -> None:
        candidates, previous = self.setup_three()
        updated_profile = UserProfile(age=70, flags={ProfileFlag.IS_VETERAN: True})
        result = reevaluate_affected(
            previous,
            updated_profile,
            ["age"],
            programs_with_rulesets=candidates,
            now=NOW + timedelta(hours=1),
        )
        assert len(result) == len(previous)
        # same objects, not re-derived copies
        assert result[1] is previous[1]
        assert result[2] is previous[2]
        assert result[1].model_dump(mode="json") == previous[1].model_dump(mode="json")

    def test_affected_evaluation_is_recomputed(self) -> None:
        candidates, previous = self.setup_three()
        assert previous[0].status is MatchStatus.MORE_INFO_NEEDED
        updated_profile = UserProfile(age=70, flags={ProfileFlag.IS_VETERAN: True})
        result = reevaluate_affected(
            previous, updated_profile, ["age"], programs_with_rulesets=candidates, now=NOW
        )
        assert result[0] is not previous[0]
        assert result[0].status is MatchStatus.POTENTIAL_MATCH

    def test_order_is_preserved(self) -> None:
        candidates, previous = self.setup_three()
        result = reevaluate_affected(
            previous,
            UserProfile(age=70, flags={ProfileFlag.IS_VETERAN: True}),
            ["age"],
            programs_with_rulesets=candidates,
            now=NOW,
        )
        assert [item.program_slug for item in result] == [item.program_slug for item in previous]

    def test_no_changed_fields_returns_everything_untouched(self) -> None:
        candidates, previous = self.setup_three()
        result = reevaluate_affected(
            previous, UserProfile(), [], programs_with_rulesets=candidates, now=NOW
        )
        assert result == list(previous)
        assert all(new is old for new, old in zip(result, previous, strict=True))

    def test_a_field_no_rule_reads_changes_nothing(self) -> None:
        candidates, previous = self.setup_three()
        result = reevaluate_affected(
            previous,
            UserProfile(housing_situation=HousingSituation.RENTING),
            ["housing_situation"],
            programs_with_rulesets=candidates,
            now=NOW,
        )
        assert all(new is old for new, old in zip(result, previous, strict=True))

    def test_evidence_that_merely_missed_a_field_still_counts_as_affected(self) -> None:
        """An ``UNKNOWN`` caused by a missing field is exactly the case that
        answering it should resolve, so it must not be skipped."""
        candidates, previous = self.setup_three()
        assert previous[0].missing_fields == ["age"]
        result = reevaluate_affected(
            previous,
            UserProfile(age=70, flags={ProfileFlag.IS_VETERAN: True}),
            ["age"],
            programs_with_rulesets=candidates,
            now=NOW,
        )
        assert result[0].status is MatchStatus.POTENTIAL_MATCH

    def test_income_alias_reaches_income_rules(self) -> None:
        candidates = [
            candidate(
                "reads-income",
                IncomeAbsoluteCondition(
                    description="Under $4,000 a month", max_monthly_cents=400_000
                ),
            )
        ]
        previous = evaluate_all(candidates, UserProfile(), now=NOW)
        assert previous[0].status is MatchStatus.MORE_INFO_NEEDED
        updated = UserProfile(
            income_min_cents=100_000,
            income_max_cents=100_000,
            income_period=IncomePeriod.MONTHLY,
        )
        result = reevaluate_affected(
            previous, updated, ["income"], programs_with_rulesets=candidates, now=NOW
        )
        assert result[0].status is MatchStatus.POTENTIAL_MATCH

    def test_missing_rule_set_for_an_affected_program_raises(self) -> None:
        _, previous = self.setup_three()
        with pytest.raises(ValueError, match="no rule set supplied"):
            reevaluate_affected(
                previous, UserProfile(age=70), ["age"], programs_with_rulesets=[], now=NOW
            )

    def test_engine_method_matches_the_module_function(self) -> None:
        candidates, previous = self.setup_three()
        updated = UserProfile(age=70, flags={ProfileFlag.IS_VETERAN: True})
        engine = RuleEngine()
        from_class = engine.reevaluate_affected(
            previous, updated, ["age"], candidates=candidates, now=NOW
        )
        from_function = reevaluate_affected(
            previous, updated, ["age"], programs_with_rulesets=candidates, now=NOW
        )
        assert [item.model_dump(mode="json") for item in from_class] == [
            item.model_dump(mode="json") for item in from_function
        ]

    def test_reevaluation_is_idempotent(self) -> None:
        candidates, previous = self.setup_three()
        updated = UserProfile(age=70, flags={ProfileFlag.IS_VETERAN: True})
        once = reevaluate_affected(
            previous, updated, ["age"], programs_with_rulesets=candidates, now=NOW
        )
        twice = reevaluate_affected(
            once, updated, ["age"], programs_with_rulesets=candidates, now=NOW
        )
        assert [item.model_dump(mode="json") for item in once] == [
            item.model_dump(mode="json") for item in twice
        ]


# ---------------------------------------------------------------------------
# the hand-labelled scenario suite
# ---------------------------------------------------------------------------
def load_fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


FIXTURE = load_fixture()
FIXTURE_RULESETS = [RuleSet.model_validate(item) for item in FIXTURE["rulesets"]]
FIXTURE_ENTRIES = [
    ProgramRuleset(program_uuid(ruleset.program_slug), ruleset.program_slug, ruleset)
    for ruleset in FIXTURE_RULESETS
]
FIXTURE_SCENARIOS: list[dict[str, object]] = list(FIXTURE["scenarios"])
FIXTURE_NOW = datetime.fromisoformat(str(FIXTURE["evaluated_on"]))


def evaluate_scenario(scenario: dict[str, object]) -> dict[str, EligibilityEvaluation]:
    profile = UserProfile.model_validate(scenario["profile"])
    return {
        item.program_slug: item for item in evaluate_all(FIXTURE_ENTRIES, profile, now=FIXTURE_NOW)
    }


class TestLabelledScenarios:
    """The file the benchmark phase scores rule correctness against.

    Each label is derivable from the rule sets embedded in the same file plus
    the published 2026 poverty guidelines, so a disagreement here is either an
    engine bug or a label bug - never an unexplained difference.
    """

    def test_fixture_is_big_enough_to_be_meaningful(self) -> None:
        assert len(FIXTURE_SCENARIOS) >= 20
        assert len({str(scenario["name"]) for scenario in FIXTURE_SCENARIOS}) == len(
            FIXTURE_SCENARIOS
        )

    def test_every_rule_set_is_usable_for_matching(self) -> None:
        for ruleset in FIXTURE_RULESETS:
            assert ruleset.is_usable_for_matching(FIXTURE_NOW.date())

    def test_fixture_carries_a_fictional_data_disclaimer(self) -> None:
        assert "FICTIONAL" in str(FIXTURE["disclaimer"])

    @pytest.mark.parametrize(
        "scenario", FIXTURE_SCENARIOS, ids=[str(item["id"]) for item in FIXTURE_SCENARIOS]
    )
    def test_scenario_statuses(self, scenario: dict[str, object]) -> None:
        evaluations = evaluate_scenario(scenario)
        expected: dict[str, str] = dict(cast(Mapping[str, str], scenario["expected"]))
        actual = {slug: evaluations[slug].status.value for slug in expected}
        assert actual == expected

    @pytest.mark.parametrize(
        "scenario", FIXTURE_SCENARIOS, ids=[str(item["id"]) for item in FIXTURE_SCENARIOS]
    )
    def test_every_program_is_labelled(self, scenario: dict[str, object]) -> None:
        expected: dict[str, str] = dict(cast(Mapping[str, str], scenario["expected"]))
        assert set(expected) == {ruleset.program_slug for ruleset in FIXTURE_RULESETS}

    @pytest.mark.parametrize(
        "scenario", FIXTURE_SCENARIOS, ids=[str(item["id"]) for item in FIXTURE_SCENARIOS]
    )
    def test_more_info_needed_always_names_an_answerable_field(
        self, scenario: dict[str, object]
    ) -> None:
        for evaluation in evaluate_scenario(scenario).values():
            if evaluation.status is MatchStatus.MORE_INFO_NEEDED:
                assert evaluation.missing_fields
                assert set(evaluation.missing_fields) <= PROFILE_FIELD_NAMES

    @pytest.mark.parametrize(
        "scenario", FIXTURE_SCENARIOS, ids=[str(item["id"]) for item in FIXTURE_SCENARIOS]
    )
    def test_results_are_reproducible(self, scenario: dict[str, object]) -> None:
        first = evaluate_scenario(scenario)
        second = evaluate_scenario(scenario)
        assert {slug: item.model_dump(mode="json") for slug, item in first.items()} == {
            slug: item.model_dump(mode="json") for slug, item in second.items()
        }

    def test_the_suite_covers_all_three_statuses(self) -> None:
        seen = {
            evaluation.status
            for scenario in FIXTURE_SCENARIOS
            for evaluation in evaluate_scenario(scenario).values()
        }
        assert seen == set(MatchStatus)

    def test_an_out_of_state_profile_can_still_match_a_campus_program(self) -> None:
        """Guards the rule that geography is per-program, not global."""
        scenario = next(
            item
            for item in FIXTURE_SCENARIOS
            if item["name"] == "out-of-state-rutgers-graduate-student"
        )
        evaluations = evaluate_scenario(scenario)
        assert evaluations["rutgers-student-food-pantry"].status is MatchStatus.POTENTIAL_MATCH
        assert evaluations["nj-snap"].status is MatchStatus.UNLIKELY_MATCH

    def test_a_custom_fpl_table_changes_results_but_not_the_mapping(self) -> None:
        """The table is an injected input, so a pinned table is testable."""
        empty_table = FplTable({})
        scenario = next(
            item for item in FIXTURE_SCENARIOS if item["name"] == "single-parent-low-income-renter"
        )
        profile = UserProfile.model_validate(scenario["profile"])
        evaluations = {
            item.program_slug: item
            for item in evaluate_all(FIXTURE_ENTRIES, profile, fpl=empty_table, now=FIXTURE_NOW)
        }
        snap = evaluations["nj-snap"]
        assert snap.status is MatchStatus.MORE_INFO_NEEDED
        assert any("federal poverty guidelines" in text for text in snap.limitations)


def test_rule_engine_satisfies_the_published_contract() -> None:
    """``RuleEngine`` is what the workflow depends on through the protocol, so
    a signature drift here would only surface at integration time."""
    from app.schemas.contracts import EligibilityEngine

    engine: EligibilityEngine = RuleEngine()
    assert isinstance(engine, EligibilityEngine)


def test_uuid_helper_is_stable() -> None:
    """Program ids in the fixture are derived, not stored, so this must hold."""
    assert program_uuid("nj-snap") == uuid.uuid5(
        uuid.UUID("6f1b9e1c-2a4d-5b8e-9c3f-7d0a1b2c3d4e"), "nj-snap"
    )


# ---------------------------------------------------------------------------
# author-written clarification copy
# ---------------------------------------------------------------------------
def author_copy_strings() -> list[tuple[str, str]]:
    """Every author-written string in the engine's clarification copy tables.

    Built by walking the tables rather than by listing sentences, so copy added
    later is covered without touching this test.
    """
    items: list[tuple[str, str]] = []
    for field, template in _QUESTION_TEMPLATES.items():
        where = f"_QUESTION_TEMPLATES[{field!r}]"
        items.append((f"{where}.question", template.question))
        items.append((f"{where}.why_it_matters", template.why_it_matters))
        if template.unit is not None:
            items.append((f"{where}.unit", template.unit))
        for value, label in template.options:
            items.append((f"{where}.options[{value!r}]", label))
    for flag, (question, why_it_matters) in _FLAG_QUESTIONS.items():
        where = f"_FLAG_QUESTIONS[{flag.value!r}]"
        items.append((f"{where}.question", question))
        items.append((f"{where}.why_it_matters", why_it_matters))
    for value, label in _YES_NO:
        items.append((f"_YES_NO[{value!r}]", label))
    return items


AUTHOR_COPY_STRINGS = author_copy_strings()


class TestAuthorWrittenCopyUsesTheProductVocabulary:
    """Product rule 2 applies to copy we wrote, not only to model output.

    ``BANNED_STATUS_VOCABULARY`` was enforced on generated text alone, so an
    author-written question could say "eligible" and only the end-to-end
    journey would notice. These assertions move that check to the source.
    """

    def test_every_copy_string_is_covered(self) -> None:
        """A table-driven test over an empty table passes silently, so the
        collector itself is asserted against the tables it walks."""
        assert AUTHOR_COPY_STRINGS
        labels = [label for label, _ in AUTHOR_COPY_STRINGS]
        assert len(labels) == len(set(labels))
        for field in _QUESTION_TEMPLATES:
            assert f"_QUESTION_TEMPLATES[{field!r}].question" in labels
        for flag in _FLAG_QUESTIONS:
            assert f"_FLAG_QUESTIONS[{flag.value!r}].question" in labels

    @pytest.mark.parametrize(
        ("where", "text"), AUTHOR_COPY_STRINGS, ids=[label for label, _ in AUTHOR_COPY_STRINGS]
    )
    def test_copy_avoids_banned_status_vocabulary(self, where: str, text: str) -> None:
        """Words like "eligible" or "you qualify" state a conclusion this
        system is not entitled to state, wherever the wording came from."""
        haystack = text.casefold()
        found = [term for term in BANNED_STATUS_VOCABULARY if term in haystack]
        assert not found, f"{where} uses banned status wording {found}: {text!r}"

    def test_the_check_would_catch_a_regression(self) -> None:
        """Pins the assertion against the exact wording that escaped once, so a
        future loosening of the check cannot pass unnoticed."""
        regression = "Getting cash assistance can make you automatically eligible elsewhere."
        assert any(term in regression.casefold() for term in BANNED_STATUS_VOCABULARY)
