"""Contract tests for the shared schemas.

These guard the invariants that every other module is built on top of, so a
failure here means a downstream agent's assumptions have silently changed:

* income is a range and normalises outward, never inward;
* ``income_min > income_max`` is rejected;
* every rule condition survives a JSON round trip through the discriminated
  union, including nested combinators;
* "unknown" is representable on every enum where a value can genuinely be
  unknown;
* a match status can only ever come from the rule engine.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas import (
    STATUS_BY_OUTCOME,
    ActionItem,
    ActionItemKind,
    AgeCondition,
    AllOf,
    AnyOf,
    CategoricalCondition,
    CategoricalProfileField,
    Condition,
    ConditionEvidence,
    ConfidenceLevel,
    EligibilityEvaluation,
    EmploymentStatus,
    ExtractedFlag,
    ExtractedProfile,
    FlagCondition,
    GeographyCondition,
    HouseholdComposition,
    HouseholdSizeCondition,
    HousingSituation,
    IncomeAbsoluteCondition,
    IncomeFplCondition,
    IncomePeriod,
    MatchStatus,
    NeedCategory,
    Not,
    ProfileFlag,
    ReviewStatus,
    RuleSet,
    StudentStatus,
    StudentStatusCondition,
    Trilean,
    UserProfile,
    group_shared_documents,
    make_dedup_key,
    render_requirements,
    resolve_all_of,
    resolve_any_of,
    resolve_not,
    status_for_outcome,
)

CONDITION_ADAPTER: TypeAdapter[Condition] = TypeAdapter(Condition)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# income range normalisation
# ---------------------------------------------------------------------------
class TestIncomeRange:
    def test_monthly_income_passes_through_unchanged(self) -> None:
        profile = UserProfile(
            income_min_cents=180_000,
            income_max_cents=220_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert profile.income_monthly_range_cents() == (180_000, 220_000)

    def test_annual_income_is_divided_by_twelve(self) -> None:
        profile = UserProfile(
            income_min_cents=2_400_000,
            income_max_cents=3_600_000,
            income_period=IncomePeriod.ANNUAL,
        )
        assert profile.income_monthly_range_cents() == (200_000, 300_000)

    def test_annual_rounding_widens_the_range_outward(self) -> None:
        """A rounding artefact must never shrink the range.

        Shrinking could turn a straddling (``UNKNOWN``) comparison into a
        definite ``FALSE`` and tell a real person not to bother applying. So the
        minimum floors and the maximum ceilings.
        """
        profile = UserProfile(
            income_min_cents=100_007,
            income_max_cents=100_013,
            income_period=IncomePeriod.ANNUAL,
        )
        low, high = profile.income_monthly_range_cents()
        assert low == 100_007 // 12
        assert high == -(-100_013 // 12)
        assert low is not None and high is not None
        assert low * 12 <= 100_007
        assert high * 12 >= 100_013

    def test_monthly_to_annual_multiplies_by_twelve(self) -> None:
        profile = UserProfile(
            income_min_cents=150_000,
            income_max_cents=150_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert profile.income_annual_range_cents() == (1_800_000, 1_800_000)

    def test_half_open_range_is_supported(self) -> None:
        """Users often know only a floor or only a ceiling."""
        profile = UserProfile(income_max_cents=1_200_000, income_period=IncomePeriod.ANNUAL)
        assert profile.income_monthly_range_cents() == (None, 100_000)
        assert profile.has_income_information() is True

    def test_no_income_reported_yields_no_range(self) -> None:
        profile = UserProfile()
        assert profile.income_monthly_range_cents() == (None, None)
        assert profile.income_annual_range_cents() == (None, None)
        assert profile.has_income_information() is False

    def test_min_greater_than_max_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="income_min_cents must not exceed"):
            UserProfile(
                income_min_cents=500_000,
                income_max_cents=400_000,
                income_period=IncomePeriod.MONTHLY,
            )

    def test_equal_min_and_max_is_an_exact_figure(self) -> None:
        profile = UserProfile(
            income_min_cents=250_000,
            income_max_cents=250_000,
            income_period=IncomePeriod.MONTHLY,
        )
        assert profile.income_monthly_range_cents() == (250_000, 250_000)

    def test_amount_without_a_period_is_rejected(self) -> None:
        """An amount with an unknown period must not be assumed to be monthly."""
        with pytest.raises(ValidationError, match="income_period is required"):
            UserProfile(income_min_cents=250_000)

    def test_negative_income_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            UserProfile(income_min_cents=-1, income_period=IncomePeriod.MONTHLY)


# ---------------------------------------------------------------------------
# other profile invariants
# ---------------------------------------------------------------------------
class TestUserProfile:
    def test_state_is_upper_cased_and_county_suffix_stripped(self) -> None:
        profile = UserProfile(state="nj", county="Middlesex County")
        assert profile.state == "NJ"
        assert profile.county == "Middlesex"

    def test_absent_flag_is_unknown_not_false(self) -> None:
        profile = UserProfile(flags={ProfileFlag.RECEIVES_SNAP: True})
        assert profile.flag(ProfileFlag.RECEIVES_SNAP) is True
        assert profile.flag(ProfileFlag.IS_VETERAN) is None
        assert profile.is_unknown("flags.is_veteran") is True

    def test_household_composition_cannot_exceed_household_size(self) -> None:
        with pytest.raises(ValidationError, match="exceed household_size"):
            UserProfile(
                household_size=2,
                household_composition=HouseholdComposition(adults=2, children_under_18=2),
            )

    def test_unknown_fields_must_name_real_fields(self) -> None:
        with pytest.raises(ValidationError, match="not profile fields"):
            UserProfile(unknown_fields=["anual_incom"])

    def test_unknown_fields_accepts_flag_paths(self) -> None:
        profile = UserProfile(unknown_fields=["flags.is_disabled", "income"])
        assert profile.is_unknown("flags.is_disabled") is True

    def test_needs_are_deduplicated(self) -> None:
        profile = UserProfile(needs=[NeedCategory.FOOD, NeedCategory.FOOD, NeedCategory.UTILITIES])
        assert profile.needs == [NeedCategory.FOOD, NeedCategory.UTILITIES]

    def test_extra_fields_are_rejected(self) -> None:
        """A typo in a caller must fail loudly, not vanish into the profile."""
        with pytest.raises(ValidationError):
            UserProfile.model_validate({"stat": "NJ"})

    def test_round_trips_through_json(self) -> None:
        profile = UserProfile(
            state="NJ",
            county="Middlesex",
            student_status=StudentStatus.UNDERGRAD_FULL_TIME,
            household_size=4,
            household_composition=HouseholdComposition(adults=2, children_under_18=2),
            income_min_cents=2_400_000,
            income_max_cents=3_000_000,
            income_period=IncomePeriod.ANNUAL,
            employment_status=EmploymentStatus.EMPLOYED_PART_TIME,
            housing_situation=HousingSituation.LIVING_WITH_FAMILY,
            needs=[NeedCategory.FOOD, NeedCategory.UTILITIES],
            age=20,
            flags={ProfileFlag.HAS_CHILDREN: True},
        )
        assert UserProfile.model_validate(profile.model_dump(mode="json")) == profile


class TestExtractedProfile:
    def test_projects_only_the_fields_that_were_set(self) -> None:
        extracted = ExtractedProfile(
            state="NJ",
            needs=[NeedCategory.FOOD],
            flags=[ExtractedFlag(flag=ProfileFlag.HAS_CHILDREN, value=True)],
        )
        fields = extracted.to_profile_fields()
        assert fields == {
            "state": "NJ",
            "needs": [NeedCategory.FOOD],
            "flags": {ProfileFlag.HAS_CHILDREN: True},
        }
        assert "income_min_cents" not in fields

    def test_low_confidence_values_can_be_dropped(self) -> None:
        extracted = ExtractedProfile.model_validate(
            {
                "household_size": 5,
                "field_evidence": [
                    {"field": "household_size", "confidence": 0.3, "source_span": "a few of us"}
                ],
            }
        )
        assert extracted.confidence_for("household_size") == pytest.approx(0.3)
        assert extracted.source_span_for("household_size") == "a few of us"
        assert "household_size" not in extracted.to_profile_fields(min_confidence=0.6)
        assert "household_size" in extracted.to_profile_fields(min_confidence=0.2)

    def test_evidence_for_an_unknown_field_name_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not a known profile field"):
            ExtractedProfile.model_validate(
                {"field_evidence": [{"field": "salary", "confidence": 1.0}]}
            )

    def test_projection_feeds_a_valid_user_profile(self) -> None:
        extracted = ExtractedProfile(
            state="NJ",
            income_min_cents=2_400_000,
            income_max_cents=2_400_000,
            income_period=IncomePeriod.ANNUAL,
        )
        profile = UserProfile.model_validate(extracted.to_profile_fields())
        assert profile.income_monthly_range_cents() == (200_000, 200_000)


# ---------------------------------------------------------------------------
# the rule DSL
# ---------------------------------------------------------------------------
def _every_leaf_condition() -> list[Condition]:
    """One instance of every non-combinator condition type."""
    return [
        GeographyCondition(
            description="You live in New Jersey",
            states=["NJ"],
            counties=["Middlesex"],
            source_ids=["nj-snap-overview-2026"],
        ),
        StudentStatusCondition(
            description="You are enrolled at least part time",
            allowed=[StudentStatus.UNDERGRAD_FULL_TIME, StudentStatus.UNDERGRAD_PART_TIME],
        ),
        HouseholdSizeCondition(description="Your household has 1 to 8 people", min=1, max=8),
        IncomeFplCondition(
            description="Your income is at or below 130% of the federal poverty level",
            max_pct_fpl=130.0,
            fpl_year=2026,
            household_definition="Everyone who buys and prepares food together.",
        ),
        IncomeAbsoluteCondition(
            description="Your household earns $4,000 a month or less",
            max_monthly_cents=400_000,
        ),
        CategoricalCondition(
            description="You rent or live with family",
            field=CategoricalProfileField.HOUSING_SITUATION,
            any_of=[HousingSituation.RENTING.value, HousingSituation.LIVING_WITH_FAMILY.value],
        ),
        AgeCondition(description="You are 18 or older", min=18),
        FlagCondition(
            description="You already receive SNAP",
            flag=ProfileFlag.RECEIVES_SNAP,
            expected=True,
        ),
    ]


class TestConditionUnion:
    @pytest.mark.parametrize("condition", _every_leaf_condition(), ids=lambda c: str(c.kind))
    def test_leaf_condition_round_trips(self, condition: Condition) -> None:
        payload = condition.model_dump(mode="json")
        assert payload["kind"] == condition.kind
        assert CONDITION_ADAPTER.validate_python(payload) == condition

    @pytest.mark.parametrize("condition", _every_leaf_condition(), ids=lambda c: str(c.kind))
    def test_leaf_condition_round_trips_through_json_text(self, condition: Condition) -> None:
        raw = CONDITION_ADAPTER.dump_json(condition)
        assert CONDITION_ADAPTER.validate_json(raw) == condition

    def test_combinators_nest_recursively_and_round_trip(self) -> None:
        tree: Condition = AllOf(
            description="All of the following apply",
            conditions=[
                *_every_leaf_condition(),
                AnyOf(
                    description="Either of the following applies",
                    conditions=[
                        FlagCondition(
                            description="You have children",
                            flag=ProfileFlag.HAS_CHILDREN,
                        ),
                        Not(
                            description="You are not a graduate student",
                            condition=StudentStatusCondition(
                                description="You are a graduate student",
                                allowed=[StudentStatus.GRADUATE],
                            ),
                        ),
                    ],
                ),
            ],
        )
        assert CONDITION_ADAPTER.validate_json(CONDITION_ADAPTER.dump_json(tree)) == tree

    def test_every_condition_kind_is_reachable_through_the_union(self) -> None:
        """Guards against a new condition class being left out of ``Condition``."""
        leaves = {str(condition.kind) for condition in _every_leaf_condition()}
        combinators = {"all_of", "any_of", "not"}
        reachable = set()
        for member in (
            GeographyCondition,
            StudentStatusCondition,
            HouseholdSizeCondition,
            IncomeFplCondition,
            IncomeAbsoluteCondition,
            CategoricalCondition,
            AgeCondition,
            FlagCondition,
            AllOf,
            AnyOf,
            Not,
        ):
            reachable.add(member.model_fields["kind"].default)
        assert reachable == leaves | combinators

    def test_unknown_kind_is_rejected_by_the_discriminator(self) -> None:
        with pytest.raises(ValidationError):
            CONDITION_ADAPTER.validate_python({"kind": "vibes", "description": "Feels right"})

    def test_description_is_required_on_every_condition(self) -> None:
        with pytest.raises(ValidationError):
            CONDITION_ADAPTER.validate_python({"kind": "age", "min": 18})

    def test_empty_combinator_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AllOf(description="Nothing at all", conditions=[])

    def test_income_absolute_requires_exactly_one_bound(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            IncomeAbsoluteCondition(description="Under a limit")
        with pytest.raises(ValidationError, match="exactly one"):
            IncomeAbsoluteCondition(
                description="Under a limit", max_monthly_cents=1, max_annual_cents=2
            )

    def test_a_rule_cannot_require_an_unknown_value(self) -> None:
        with pytest.raises(ValidationError, match="UNKNOWN"):
            StudentStatusCondition(description="Unknown status", allowed=[StudentStatus.UNKNOWN])
        with pytest.raises(ValidationError, match="UNKNOWN"):
            CategoricalCondition(
                description="Unknown housing",
                field=CategoricalProfileField.HOUSING_SITUATION,
                any_of=["UNKNOWN"],
            )

    def test_requirements_render_from_author_written_descriptions(self) -> None:
        tree: Condition = AllOf(
            description="All of the following apply",
            conditions=[
                GeographyCondition(description="You live in New Jersey", states=["NJ"]),
                AgeCondition(description="You are 18 or older", min=18),
            ],
        )
        bullets = render_requirements(tree, ["Student eligibility rules are not fully encoded."])
        assert [bullet.text for bullet in bullets] == [
            "You live in New Jersey",
            "You are 18 or older",
            "Student eligibility rules are not fully encoded.",
        ]
        assert bullets[-1].is_evaluable is False


class TestThreeValuedLogic:
    @pytest.mark.parametrize(
        ("outcomes", "expected"),
        [
            ([Trilean.TRUE, Trilean.TRUE], Trilean.TRUE),
            ([Trilean.TRUE, Trilean.UNKNOWN], Trilean.UNKNOWN),
            ([Trilean.TRUE, Trilean.FALSE], Trilean.FALSE),
            ([Trilean.UNKNOWN, Trilean.FALSE], Trilean.FALSE),
            ([Trilean.UNKNOWN, Trilean.UNKNOWN], Trilean.UNKNOWN),
            ([Trilean.FALSE, Trilean.FALSE], Trilean.FALSE),
        ],
    )
    def test_all_of(self, outcomes: list[Trilean], expected: Trilean) -> None:
        assert resolve_all_of(outcomes) is expected

    @pytest.mark.parametrize(
        ("outcomes", "expected"),
        [
            ([Trilean.TRUE, Trilean.TRUE], Trilean.TRUE),
            ([Trilean.TRUE, Trilean.UNKNOWN], Trilean.TRUE),
            ([Trilean.TRUE, Trilean.FALSE], Trilean.TRUE),
            ([Trilean.UNKNOWN, Trilean.FALSE], Trilean.UNKNOWN),
            ([Trilean.UNKNOWN, Trilean.UNKNOWN], Trilean.UNKNOWN),
            ([Trilean.FALSE, Trilean.FALSE], Trilean.FALSE),
        ],
    )
    def test_any_of(self, outcomes: list[Trilean], expected: Trilean) -> None:
        assert resolve_any_of(outcomes) is expected

    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (Trilean.TRUE, Trilean.FALSE),
            (Trilean.FALSE, Trilean.TRUE),
            (Trilean.UNKNOWN, Trilean.UNKNOWN),
        ],
    )
    def test_not_preserves_unknown(self, outcome: Trilean, expected: Trilean) -> None:
        assert resolve_not(outcome) is expected

    def test_a_definite_result_short_circuits_unknowns(self) -> None:
        """One failed hard requirement is enough, even with fields missing."""
        assert resolve_all_of([Trilean.UNKNOWN, Trilean.FALSE, Trilean.UNKNOWN]) is Trilean.FALSE
        assert resolve_any_of([Trilean.UNKNOWN, Trilean.TRUE, Trilean.UNKNOWN]) is Trilean.TRUE

    def test_status_mapping_is_total_and_hedged(self) -> None:
        assert status_for_outcome(Trilean.TRUE) is MatchStatus.POTENTIAL_MATCH
        assert status_for_outcome(Trilean.UNKNOWN) is MatchStatus.MORE_INFO_NEEDED
        assert status_for_outcome(Trilean.FALSE) is MatchStatus.UNLIKELY_MATCH
        assert set(STATUS_BY_OUTCOME) == set(Trilean)
        assert set(STATUS_BY_OUTCOME.values()) == set(MatchStatus)
        assert not any(
            word in status.value.lower()
            for status in MatchStatus
            for word in ("eligible", "ineligible")
        )


class TestEvidenceAndEvaluation:
    def test_unknown_leaf_must_name_the_missing_field(self) -> None:
        with pytest.raises(ValidationError, match="must list missing_fields"):
            ConditionEvidence(
                kind="age", description="You are 18 or older", outcome=Trilean.UNKNOWN
            )

    def test_unknown_combinator_inherits_the_reason_from_its_children(self) -> None:
        evidence = ConditionEvidence(
            kind="all_of",
            description="All of the following apply",
            outcome=Trilean.UNKNOWN,
            children=[
                ConditionEvidence(
                    kind="age",
                    description="You are 18 or older",
                    outcome=Trilean.UNKNOWN,
                    missing_fields=["age"],
                )
            ],
        )
        assert evidence.children[0].missing_fields == ["age"]

    def test_status_cannot_contradict_the_deterministic_outcome(self) -> None:
        with pytest.raises(ValidationError, match="contradicts outcome"):
            EligibilityEvaluation(
                program_id=uuid.uuid4(),
                program_slug="nj-snap",
                rule_version="2026.1",
                status=MatchStatus.POTENTIAL_MATCH,
                outcome=Trilean.FALSE,
                evaluated_at=NOW,
            )

    def test_more_info_needed_must_say_what_is_missing(self) -> None:
        with pytest.raises(ValidationError, match="missing_fields"):
            EligibilityEvaluation(
                program_id=uuid.uuid4(),
                program_slug="nj-snap",
                rule_version="2026.1",
                status=MatchStatus.MORE_INFO_NEEDED,
                outcome=Trilean.UNKNOWN,
                evaluated_at=NOW,
            )

    def test_a_valid_evaluation_round_trips(self) -> None:
        evaluation = EligibilityEvaluation(
            program_id=uuid.uuid4(),
            program_slug="nj-snap",
            rule_version="2026.1",
            status=MatchStatus.MORE_INFO_NEEDED,
            outcome=Trilean.UNKNOWN,
            score=0.5,
            missing_fields=["income"],
            evidence=[
                ConditionEvidence(
                    kind="income_fpl",
                    description="Your income is at or below 130% of the federal poverty level",
                    outcome=Trilean.UNKNOWN,
                    missing_fields=["income"],
                    profile_fields_used=["income_min_cents", "household_size"],
                    threshold_applied="130% of the 2026 federal poverty level, household of 4",
                    source_ids=["nj-snap-income-limits-2026"],
                )
            ],
            evaluated_at=NOW,
        )
        assert (
            EligibilityEvaluation.model_validate(evaluation.model_dump(mode="json")) == evaluation
        )

    def test_naive_datetime_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="naive datetime"):
            EligibilityEvaluation(
                program_id=uuid.uuid4(),
                program_slug="nj-snap",
                rule_version="2026.1",
                status=MatchStatus.POTENTIAL_MATCH,
                outcome=Trilean.TRUE,
                evaluated_at=datetime(2026, 10, 10, 12, 0),
            )


class TestRuleSet:
    def _ruleset(self, **overrides: object) -> RuleSet:
        payload: dict[str, object] = {
            "version": "2026.1",
            "program_slug": "nj-snap",
            "effective_from": date(2026, 1, 1),
            "root": GeographyCondition(description="You live in New Jersey", states=["NJ"]),
            "review_status": ReviewStatus.APPROVED,
            "reviewed_at": NOW,
        }
        payload.update(overrides)
        return RuleSet.model_validate(payload)

    def test_only_approved_rule_sets_may_influence_matching(self) -> None:
        when = date(2026, 6, 1)
        assert self._ruleset().is_usable_for_matching(when) is True
        for status in (ReviewStatus.DRAFT, ReviewStatus.PENDING_REVIEW, ReviewStatus.REJECTED):
            draft = self._ruleset(review_status=status, reviewed_at=None)
            assert draft.is_usable_for_matching(when) is False

    def test_effectivity_window_is_half_open(self) -> None:
        ruleset = self._ruleset(effective_to=date(2027, 1, 1))
        assert ruleset.is_effective_on(date(2026, 1, 1)) is True
        assert ruleset.is_effective_on(date(2026, 12, 31)) is True
        assert ruleset.is_effective_on(date(2027, 1, 1)) is False
        assert ruleset.is_effective_on(date(2025, 12, 31)) is False

    def test_approved_rule_set_must_record_its_review(self) -> None:
        with pytest.raises(ValidationError, match="reviewed_at"):
            self._ruleset(reviewed_at=None)

    def test_round_trips_with_a_nested_tree(self) -> None:
        ruleset = self._ruleset(
            root=AllOf(
                description="All of the following apply",
                conditions=_every_leaf_condition(),
            )
        )
        assert RuleSet.model_validate(ruleset.model_dump(mode="json")) == ruleset


# ---------------------------------------------------------------------------
# unknown must be representable
# ---------------------------------------------------------------------------
class TestUnknownIsRepresentable:
    @pytest.mark.parametrize(
        "enum_type",
        [
            StudentStatus,
            EmploymentStatus,
            HousingSituation,
            NeedCategory,
            ConfidenceLevel,
            Trilean,
        ],
    )
    def test_enum_has_an_unknown_member(self, enum_type: type[StudentStatus]) -> None:
        assert "UNKNOWN" in enum_type.__members__

    def test_income_period_models_unknown_by_absence(self) -> None:
        """A period cannot be 'unknown' - an amount without one is just invalid.

        The unknown case is ``income_period is None`` with no amounts, so there
        is deliberately no ``IncomePeriod.UNKNOWN`` to be mistaken for a
        normalisable value.
        """
        assert "UNKNOWN" not in IncomePeriod.__members__
        assert UserProfile().income_period is None

    def test_match_status_models_unknown_as_more_info_needed(self) -> None:
        assert MatchStatus.MORE_INFO_NEEDED in MatchStatus
        assert "UNKNOWN" not in MatchStatus.__members__

    def test_every_profile_field_is_optional(self) -> None:
        """An empty profile must be valid: a run has to be evaluable at any point."""
        profile = UserProfile()
        for name in ("state", "county", "student_status", "household_size", "age"):
            assert profile.is_unknown(name) is True


# ---------------------------------------------------------------------------
# action plan deduplication
# ---------------------------------------------------------------------------
class TestActionPlanDedup:
    def test_aliases_collapse_differently_worded_requirements(self) -> None:
        assert make_dedup_key(ActionItemKind.DOCUMENT, "Driver's License") == "document:photo-id"
        assert make_dedup_key(ActionItemKind.DOCUMENT, "photo ID") == "document:photo-id"
        assert (
            make_dedup_key(ActionItemKind.DOCUMENT, "Proof of NJ address")
            == "document:proof-of-address"
        )
        assert (
            make_dedup_key(ActionItemKind.DOCUMENT, "recent pay stubs")
            == "document:proof-of-income"
        )

    def test_different_kinds_never_collapse(self) -> None:
        assert make_dedup_key(ActionItemKind.DOCUMENT, "lease") != make_dedup_key(
            ActionItemKind.CONTACT, "lease"
        )

    def _item(self, program: str, subject: str, kind: ActionItemKind) -> ActionItem:
        program_id = uuid.uuid5(uuid.NAMESPACE_URL, program)
        dedup_key = make_dedup_key(kind, subject)
        return ActionItem(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"{program}:{dedup_key}"),
            program_id=program_id,
            program_name=program,
            kind=kind,
            title=f"Find your {subject}",
            detail=f"You will need your {subject} to apply.",
            sort_order=0,
            dedup_key=dedup_key,
        )

    def test_documents_shared_by_two_programs_are_grouped(self) -> None:
        items = [
            self._item("NJ SNAP", "proof of address", ActionItemKind.DOCUMENT),
            self._item("NJ USF", "proof of residency", ActionItemKind.DOCUMENT),
            self._item("NJ SNAP", "photo ID", ActionItemKind.DOCUMENT),
        ]
        shared = group_shared_documents(items)
        assert len(shared) == 1
        assert shared[0].dedup_key == "document:proof-of-address"
        assert shared[0].program_names == ["NJ SNAP", "NJ USF"]

    def test_single_program_documents_are_not_promoted_to_shared(self) -> None:
        items = [self._item("NJ SNAP", "photo ID", ActionItemKind.DOCUMENT)]
        assert group_shared_documents(items) == []

    def test_non_document_items_are_never_grouped(self) -> None:
        items = [
            self._item("NJ SNAP", "apply online", ActionItemKind.APPLY),
            self._item("NJ USF", "apply online", ActionItemKind.APPLY),
        ]
        assert group_shared_documents(items) == []

    def test_completion_timestamp_must_agree_with_the_flag(self) -> None:
        item = self._item("NJ SNAP", "photo ID", ActionItemKind.DOCUMENT)
        payload = item.model_dump(mode="json")
        payload["is_complete"] = True
        with pytest.raises(ValidationError, match="completed_at"):
            ActionItem.model_validate(payload)
        payload["completed_at"] = NOW.isoformat()
        assert ActionItem.model_validate(payload).is_complete is True
