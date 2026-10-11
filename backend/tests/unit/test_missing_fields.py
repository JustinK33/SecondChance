"""Tests for clarification questions and document requirements.

Both of these are the parts of the engine a user feels directly. A bad question
is an abandoned session, and a duplicated document is an extra trip to an
office, so the properties checked here are deliberately strict:

* a question is asked only when an answer could change a result;
* a question is never asked for a field no rule consumes, and never for a
  Social Security number, bank details or any identifier the programs do not
  need;
* the questions are ranked by how many programs they unblock and capped;
* "proof of NJ address" needed by five programs collapses to one errand, while
  genuinely different documents stay separate.
"""

from __future__ import annotations

import uuid

import pytest

from app.eligibility.documents import (
    SUBJECT_BY_CONDITION,
    SUBJECT_BY_FLAG,
    document_action_drafts,
    required_documents,
    subject_for_evidence,
)
from app.eligibility.engine import (
    MAX_CLARIFICATION_QUESTIONS,
    SENSITIVE_TERMS,
    ProgramRuleset,
    RuleEngine,
    evaluate_all,
    evaluate_condition,
    missing_field_questions,
)
from app.schemas import (
    ActionItem,
    ActionItemKind,
    AgeCondition,
    AllOf,
    AnyOf,
    CategoricalCondition,
    CategoricalProfileField,
    ClarificationInputKind,
    Condition,
    EligibilityEvaluation,
    FlagCondition,
    GeographyCondition,
    HouseholdSizeCondition,
    HousingSituation,
    IncomeAbsoluteCondition,
    IncomeFplCondition,
    IncomePeriod,
    MatchStatus,
    ProfileFlag,
    ReviewStatus,
    RuleSet,
    StudentStatus,
    StudentStatusCondition,
    UserProfile,
    action_item_uuid,
    group_shared_documents,
    make_dedup_key,
    program_uuid,
)
from app.schemas.plan import DOCUMENT_SUBJECT_ALIASES
from app.schemas.profile import PROFILE_FIELD_NAMES
from tests.unit.test_eligibility_rules import NOW, approved_ruleset

RUN_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


def entry(slug: str, root: Condition) -> ProgramRuleset:
    return ProgramRuleset(program_uuid(slug), slug, approved_ruleset(root, slug=slug))


def nj_geography() -> Condition:
    return GeographyCondition(description="You live in New Jersey", states=["NJ"])


def income_fpl(pct: float = 185.0) -> Condition:
    return IncomeFplCondition(
        description=f"Your income is at or below {pct:g}% of the federal poverty level",
        max_pct_fpl=pct,
        fpl_year=2026,
        household_definition="Everyone who lives and eats together",
    )


def evaluate(entries: list[ProgramRuleset], profile: UserProfile) -> list[EligibilityEvaluation]:
    return evaluate_all(entries, profile, now=NOW)


# ---------------------------------------------------------------------------
# clarification questions
# ---------------------------------------------------------------------------
class TestQuestionRelevance:
    def test_no_questions_when_nothing_is_unresolved(self) -> None:
        entries = [entry("nj-snap", nj_geography())]
        evaluations = evaluate(entries, UserProfile(state="NJ"))
        assert evaluations[0].status is MatchStatus.POTENTIAL_MATCH
        assert missing_field_questions(evaluations) == []

    def test_no_questions_from_an_unlikely_match(self) -> None:
        """A program that already failed a hard requirement must not generate
        work for the user; answering would not change its status."""
        root = AllOf(
            description="Both of these",
            conditions=[nj_geography(), AgeCondition(description="18 or older", min=18)],
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile(state="PA"))
        assert evaluations[0].status is MatchStatus.UNLIKELY_MATCH
        assert missing_field_questions(evaluations) == []

    def test_no_question_for_a_field_no_rule_uses(self) -> None:
        """The only source of question fields is the evidence, so a field that
        appears in no rule set can never be asked about."""
        root = AllOf(
            description="Both of these",
            conditions=[nj_geography(), AgeCondition(description="18 or older", min=18)],
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile(state="NJ"))
        questions = missing_field_questions(evaluations)
        assert [question.field for question in questions] == ["age"]
        assert all(question.field != "housing_situation" for question in questions)

    def test_every_question_field_is_a_real_profile_field(self) -> None:
        root = AllOf(
            description="Everything",
            conditions=[
                nj_geography(),
                income_fpl(),
                AgeCondition(description="18 or older", min=18),
                StudentStatusCondition(
                    description="You are not a student", allowed=[StudentStatus.NOT_A_STUDENT]
                ),
                FlagCondition(description="You are a veteran", flag=ProfileFlag.IS_VETERAN),
                CategoricalCondition(
                    description="You rent",
                    field=CategoricalProfileField.HOUSING_SITUATION,
                    any_of=[HousingSituation.RENTING.value],
                ),
            ],
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile())
        questions = missing_field_questions(evaluations, limit=20)
        assert questions
        for question in questions:
            assert question.field in PROFILE_FIELD_NAMES

    def test_an_unknown_inside_a_satisfied_disjunction_is_not_asked_about(self) -> None:
        root = AnyOf(
            description="Either of these",
            conditions=[
                FlagCondition(description="You are a veteran", flag=ProfileFlag.IS_VETERAN),
                AgeCondition(description="60 or older", min=60),
            ],
        )
        evaluations = evaluate(
            [entry("nj-snap", root)], UserProfile(flags={ProfileFlag.IS_VETERAN: True})
        )
        assert evaluations[0].status is MatchStatus.POTENTIAL_MATCH
        assert missing_field_questions(evaluations) == []

    def test_empty_input_is_handled(self) -> None:
        assert missing_field_questions([]) == []


class TestQuestionSafety:
    def test_no_template_mentions_a_sensitive_identifier(self) -> None:
        """Nothing in ``PROFILE_FIELD_NAMES`` could produce one, and this
        confirms no template has been reworded into asking for one."""
        root = AllOf(
            description="Everything askable",
            conditions=[
                nj_geography(),
                GeographyCondition(description="In Essex County", counties=["Essex"]),
                income_fpl(),
                AgeCondition(description="18 or older", min=18),
                HouseholdSizeCondition(description="At least two people", min=2),
                StudentStatusCondition(
                    description="You are not a student", allowed=[StudentStatus.NOT_A_STUDENT]
                ),
                CategoricalCondition(
                    description="You rent",
                    field=CategoricalProfileField.HOUSING_SITUATION,
                    any_of=[HousingSituation.RENTING.value],
                ),
                CategoricalCondition(
                    description="You are working",
                    field=CategoricalProfileField.EMPLOYMENT_STATUS,
                    any_of=["EMPLOYED_FULL_TIME"],
                ),
                *[
                    FlagCondition(description=f"Flag {flag.value}", flag=flag)
                    for flag in ProfileFlag
                ],
            ],
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile())
        questions = missing_field_questions(evaluations, limit=100)
        assert len(questions) >= 10
        for question in questions:
            haystack = f"{question.question} {question.why_it_matters}".casefold()
            for term in SENSITIVE_TERMS:
                assert term not in haystack

    def test_every_question_explains_why_it_matters(self) -> None:
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile())
        for question in missing_field_questions(evaluations):
            assert len(question.why_it_matters) >= 5

    def test_questions_are_skippable_by_default(self) -> None:
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile())
        assert all(not question.is_required for question in missing_field_questions(evaluations))


class TestQuestionRanking:
    @staticmethod
    def three_programs() -> list[ProgramRuleset]:
        """Three programs: all three need income, two need age, one needs county."""
        return [
            entry(
                "program-one",
                AllOf(
                    description="Income and age",
                    conditions=[income_fpl(), AgeCondition(description="18 or older", min=18)],
                ),
            ),
            entry(
                "program-two",
                AllOf(
                    description="Income and age",
                    conditions=[income_fpl(200.0), AgeCondition(description="60 or older", min=60)],
                ),
            ),
            entry(
                "program-three",
                AllOf(
                    description="Income and county",
                    conditions=[
                        income_fpl(400.0),
                        GeographyCondition(description="In Essex County", counties=["Essex"]),
                    ],
                ),
            ),
        ]

    def test_ranked_by_how_many_programs_an_answer_unblocks(self) -> None:
        profile = UserProfile(state="NJ", household_size=2)
        evaluations = evaluate(self.three_programs(), profile)
        questions = missing_field_questions(evaluations, limit=10)
        assert [question.field for question in questions] == ["income", "age", "county"]

    def test_capped_at_the_product_limit(self) -> None:
        profile = UserProfile()
        evaluations = evaluate(self.three_programs(), profile)
        assert len(missing_field_questions(evaluations)) <= MAX_CLARIFICATION_QUESTIONS

    def test_limit_is_configurable(self) -> None:
        evaluations = evaluate(self.three_programs(), UserProfile(state="NJ", household_size=2))
        assert len(missing_field_questions(evaluations, limit=1)) == 1
        assert len(missing_field_questions(evaluations, limit=2)) == 2

    def test_affected_programs_are_recorded(self) -> None:
        profile = UserProfile(state="NJ", household_size=2)
        evaluations = evaluate(self.three_programs(), profile)
        questions = {q.field: q for q in missing_field_questions(evaluations, limit=10)}
        assert questions["income"].affected_program_ids == sorted(
            str(program_uuid(slug)) for slug in ("program-one", "program-two", "program-three")
        )
        assert questions["county"].affected_program_ids == [str(program_uuid("program-three"))]

    def test_ordering_is_stable_across_calls(self) -> None:
        evaluations = evaluate(self.three_programs(), UserProfile(state="NJ"))
        first = [q.model_dump(mode="json") for q in missing_field_questions(evaluations, limit=10)]
        second = [q.model_dump(mode="json") for q in missing_field_questions(evaluations, limit=10)]
        assert first == second

    def test_a_tie_is_broken_by_a_fixed_field_order(self) -> None:
        """Two fields that unblock one condition each must still come out in a
        fixed order, or the question list flickers between runs."""
        root = AllOf(
            description="One of each",
            conditions=[
                AgeCondition(description="18 or older", min=18),
                HouseholdSizeCondition(description="At least two people", min=2),
                GeographyCondition(description="In Essex County", counties=["Essex"]),
            ],
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile(state="NJ"))
        questions = missing_field_questions(evaluations, limit=10)
        assert [question.field for question in questions] == [
            "county",
            "household_size",
            "age",
        ]


class TestQuestionShape:
    def test_income_is_one_range_question_not_two_bound_questions(self) -> None:
        """``income_min_cents`` and ``income_max_cents`` are one answer, and
        ``RangeValue`` exists so it can be asked once."""
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile(household_size=2))
        assert evaluations[0].missing_fields == ["income_min_cents", "income_max_cents"]
        questions = missing_field_questions(evaluations)
        assert [question.field for question in questions] == ["income"]
        assert questions[0].input_kind is ClarificationInputKind.RANGE
        assert questions[0].unit == "dollars per month"

    def test_household_size_is_a_bounded_number(self) -> None:
        root = HouseholdSizeCondition(description="At least two people", min=2)
        evaluations = evaluate([entry("nj-snap", root)], UserProfile())
        question = missing_field_questions(evaluations)[0]
        assert question.input_kind is ClarificationInputKind.NUMBER
        assert question.unit == "people"
        assert question.min_value == 1
        assert question.max_value == 20

    def test_a_flag_becomes_a_yes_no_question(self) -> None:
        root = FlagCondition(
            description="You have a utility shut-off notice",
            flag=ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE,
        )
        evaluations = evaluate([entry("nj-usf", root)], UserProfile())
        question = missing_field_questions(evaluations)[0]
        assert question.field == "flags.has_utility_shutoff_notice"
        assert question.input_kind is ClarificationInputKind.SINGLE_SELECT
        assert [option.value for option in question.options] == ["true", "false"]

    def test_student_status_offers_the_real_enum_values(self) -> None:
        root = StudentStatusCondition(
            description="You are not a student", allowed=[StudentStatus.NOT_A_STUDENT]
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile())
        question = missing_field_questions(evaluations)[0]
        values = {option.value for option in question.options}
        assert values <= {status.value for status in StudentStatus}
        assert StudentStatus.UNKNOWN.value not in values

    def test_ids_are_deterministic_per_run_and_field(self) -> None:
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile(household_size=2))
        first = missing_field_questions(evaluations, run_id=RUN_ID)
        second = missing_field_questions(evaluations, run_id=RUN_ID)
        assert [q.id for q in first] == [q.id for q in second]

    def test_ids_differ_between_runs(self) -> None:
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile(household_size=2))
        mine = missing_field_questions(evaluations, run_id=RUN_ID)
        theirs = missing_field_questions(evaluations, run_id=uuid.uuid4())
        assert mine[0].id != theirs[0].id

    def test_ids_are_unique_within_a_set(self) -> None:
        evaluations = evaluate(TestQuestionRanking.three_programs(), UserProfile())
        questions = missing_field_questions(evaluations, run_id=RUN_ID, limit=5)
        assert len({question.id for question in questions}) == len(questions)

    def test_engine_method_matches_the_module_function(self) -> None:
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile(household_size=2))
        engine = RuleEngine()
        assert [
            q.model_dump(mode="json")
            for q in engine.missing_field_questions(evaluations, run_id=RUN_ID)
        ] == [
            q.model_dump(mode="json") for q in missing_field_questions(evaluations, run_id=RUN_ID)
        ]


# ---------------------------------------------------------------------------
# document requirements
# ---------------------------------------------------------------------------
class TestDocumentDerivation:
    def test_geography_implies_proof_of_address(self) -> None:
        evaluations = evaluate([entry("nj-snap", nj_geography())], UserProfile(state="NJ"))
        documents = required_documents(evaluations)
        assert [document.subject for document in documents] == ["proof of address"]
        assert documents[0].dedup_key == "document:proof-of-address"

    def test_income_rules_imply_proof_of_income(self) -> None:
        for root in (
            income_fpl(),
            IncomeAbsoluteCondition(description="Under $4,000 a month", max_monthly_cents=400_000),
        ):
            evaluations = evaluate([entry("nj-usf", root)], UserProfile(household_size=1))
            assert [item.subject for item in required_documents(evaluations)] == ["proof of income"]

    def test_a_flag_with_no_paperwork_implies_no_document(self) -> None:
        """ "Do you have children" is answered on the form, not evidenced at a
        counter, so it must not create an errand."""
        root = FlagCondition(
            description="You have children under 18", flag=ProfileFlag.HAS_CHILDREN
        )
        evaluations = evaluate(
            [entry("nj-snap", root)], UserProfile(flags={ProfileFlag.HAS_CHILDREN: True})
        )
        assert required_documents(evaluations) == []

    def test_a_flag_with_paperwork_implies_the_right_document(self) -> None:
        root = FlagCondition(
            description="You have a utility shut-off notice",
            flag=ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE,
        )
        evaluations = evaluate([entry("nj-usf", root)], UserProfile())
        assert [item.subject for item in required_documents(evaluations)] == [
            "utility shutoff notice"
        ]

    def test_unknown_conditions_still_imply_their_document(self) -> None:
        """The document is exactly what resolves the unknown, so it belongs on
        the list even before we know the answer."""
        evaluations = evaluate([entry("nj-snap", income_fpl())], UserProfile(household_size=1))
        assert evaluations[0].status is MatchStatus.MORE_INFO_NEEDED
        assert [item.subject for item in required_documents(evaluations)] == ["proof of income"]

    def test_a_failed_branch_implies_nothing(self) -> None:
        """A satisfied disjunction with one FALSE branch must not ask for the
        FALSE branch's paperwork."""
        root = AnyOf(
            description="Either of these",
            conditions=[
                FlagCondition(
                    description="You have a utility shut-off notice",
                    flag=ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE,
                ),
                CategoricalCondition(
                    description="You rent or own your home",
                    field=CategoricalProfileField.HOUSING_SITUATION,
                    any_of=[HousingSituation.RENTING.value],
                ),
            ],
        )
        profile = UserProfile(
            housing_situation=HousingSituation.UNHOUSED,
            flags={ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE: True},
        )
        documents = required_documents(evaluate([entry("nj-usf", root)], profile))
        subjects = [item.subject for item in documents]
        assert subjects == ["utility shutoff notice"]
        assert "lease or mortgage" not in subjects

    def test_unlikely_matches_contribute_nothing(self) -> None:
        root = AllOf(
            description="Both of these",
            conditions=[nj_geography(), income_fpl()],
        )
        evaluations = evaluate([entry("nj-snap", root)], UserProfile(state="PA"))
        assert evaluations[0].status is MatchStatus.UNLIKELY_MATCH
        assert required_documents(evaluations) == []

    def test_combinators_imply_nothing_of_their_own(self) -> None:
        evidence = evaluate_condition(
            AllOf(description="Both", conditions=[nj_geography(), income_fpl()]),
            UserProfile(state="NJ", household_size=1),
        )
        assert subject_for_evidence(evidence) is None

    def test_every_derivable_subject_has_author_written_copy(self) -> None:
        subjects = set(SUBJECT_BY_CONDITION.values()) | set(SUBJECT_BY_FLAG.values())
        root = AllOf(
            description="Everything that implies paperwork",
            conditions=[
                nj_geography(),
                income_fpl(),
                AgeCondition(description="18 or older", min=18),
                HouseholdSizeCondition(description="At least two people", min=2),
                StudentStatusCondition(
                    description="You are enrolled", allowed=[StudentStatus.GRADUATE]
                ),
                *[
                    FlagCondition(description=f"Flag {flag.value}", flag=flag)
                    for flag in SUBJECT_BY_FLAG
                ],
                *[
                    CategoricalCondition(
                        description=f"Categorical {field.value}",
                        field=field,
                        any_of=["NJ" if field.value == "state" else "Essex"]
                        if field.value in {"state", "county"}
                        else ["RENTING", "EMPLOYED_FULL_TIME", "GRADUATE"],
                    )
                    for field in CategoricalProfileField
                ],
            ],
        )
        documents = required_documents(evaluate([entry("nj-snap", root)], UserProfile()))
        produced = {document.subject for document in documents}
        assert subjects <= produced
        for document in documents:
            assert document.title
            assert document.detail


class TestDocumentDeduplication:
    @staticmethod
    def five_programs_needing_an_address() -> list[ProgramRuleset]:
        return [
            entry(
                f"program-{index}",
                AllOf(
                    description="Where you live and what you earn",
                    conditions=[nj_geography(), income_fpl(100.0 * (index + 1))],
                ),
            )
            for index in range(5)
        ]

    def test_a_shared_requirement_collapses_to_one_item(self) -> None:
        evaluations = evaluate(
            self.five_programs_needing_an_address(),
            UserProfile(state="NJ", household_size=1),
        )
        documents = required_documents(evaluations)
        address = next(item for item in documents if item.subject == "proof of address")
        assert len(address.program_ids) == 5
        assert len(address.program_slugs) == 5
        assert address.is_shared
        assert sum(1 for item in documents if item.subject == "proof of address") == 1

    def test_distinct_requirements_stay_separate(self) -> None:
        evaluations = evaluate(
            self.five_programs_needing_an_address(),
            UserProfile(state="NJ", household_size=1),
        )
        subjects = [item.subject for item in required_documents(evaluations)]
        assert sorted(subjects) == ["proof of address", "proof of income"]

    def test_a_single_program_requirement_is_not_marked_shared(self) -> None:
        evaluations = evaluate([entry("nj-snap", nj_geography())], UserProfile(state="NJ"))
        assert not required_documents(evaluations)[0].is_shared

    def test_most_shared_documents_come_first(self) -> None:
        entries = [
            *self.five_programs_needing_an_address(),
            entry(
                "only-one-needs-enrollment",
                StudentStatusCondition(
                    description="You are enrolled", allowed=[StudentStatus.GRADUATE]
                ),
            ),
        ]
        evaluations = evaluate(entries, UserProfile(state="NJ", household_size=1))
        documents = required_documents(evaluations)
        counts = [len(document.program_ids) for document in documents]
        assert counts == sorted(counts, reverse=True)
        assert documents[-1].subject == "proof of enrollment"

    def test_different_condition_kinds_can_share_one_document(self) -> None:
        """A geography rule and a state categorical rule both mean "show me
        where you live", so they must collapse."""
        root = AllOf(
            description="Where you live, twice over",
            conditions=[
                nj_geography(),
                CategoricalCondition(
                    description="Your state is New Jersey",
                    field=CategoricalProfileField.STATE,
                    any_of=["NJ"],
                ),
            ],
        )
        documents = required_documents(evaluate([entry("nj-snap", root)], UserProfile(state="NJ")))
        assert len(documents) == 1
        assert documents[0].condition_kinds == ("categorical", "geography")

    def test_dedup_key_is_the_shared_plan_derivation(self) -> None:
        """The key must be exactly what the action-plan builder computes, or a
        requirement found here and an item created there would not group."""
        evaluations = evaluate([entry("nj-snap", nj_geography())], UserProfile(state="NJ"))
        document = required_documents(evaluations)[0]
        assert document.dedup_key == make_dedup_key(ActionItemKind.DOCUMENT, document.subject)
        assert document.dedup_key == (f"document:{DOCUMENT_SUBJECT_ALIASES['proof-of-address']}")

    def test_keys_are_unique_per_requirement(self) -> None:
        evaluations = evaluate(
            self.five_programs_needing_an_address(),
            UserProfile(state="NJ", household_size=1),
        )
        documents = required_documents(evaluations)
        keys = [document.dedup_key for document in documents]
        assert len(keys) == len(set(keys))

    def test_output_is_deterministic(self) -> None:
        evaluations = evaluate(
            self.five_programs_needing_an_address(),
            UserProfile(state="NJ", household_size=1),
        )
        assert required_documents(evaluations) == required_documents(evaluations)


class TestDocumentDrafts:
    def test_one_draft_per_program_and_subject(self) -> None:
        entries = TestDocumentDeduplication.five_programs_needing_an_address()
        evaluations = evaluate(entries, UserProfile(state="NJ", household_size=1))
        drafts = document_action_drafts(evaluations)
        assert len(drafts) == 10  # five programs x two documents
        assert {draft.kind for draft in drafts} == {ActionItemKind.DOCUMENT}

    def test_drafts_group_exactly_as_required_documents_does(self) -> None:
        """``group_shared_documents`` is the plan layer's own collapse, so the
        two views must agree without a second rule."""
        entries = TestDocumentDeduplication.five_programs_needing_an_address()
        evaluations = evaluate(entries, UserProfile(state="NJ", household_size=1))
        run_id = RUN_ID

        items: list[ActionItem] = []
        for index, draft in enumerate(document_action_drafts(evaluations)):
            dedup_key = make_dedup_key(draft.kind, draft.subject)
            items.append(
                ActionItem(
                    id=action_item_uuid(run_id, f"{draft.program_slug}:{dedup_key}"),
                    program_id=program_uuid(draft.program_slug),
                    program_name=draft.program_slug,
                    kind=draft.kind,
                    title=draft.title,
                    detail=draft.detail,
                    sort_order=index,
                    dedup_key=dedup_key,
                )
            )
        shared = group_shared_documents(items)
        from_engine = {
            document.dedup_key: sorted(str(pid) for pid in document.program_ids)
            for document in required_documents(evaluations)
            if document.is_shared
        }
        from_plan = {
            document.dedup_key: sorted(str(pid) for pid in document.program_ids)
            for document in shared
        }
        assert from_engine == from_plan


@pytest.mark.parametrize("flag", list(ProfileFlag))
def test_every_flag_either_has_a_question_or_is_never_asked(flag: ProfileFlag) -> None:
    """A rule may test any flag, so each one must either produce a usable
    question or be deliberately silent - never a half-built question."""
    ruleset = RuleSet(
        version="2026.1",
        program_slug="nj-snap",
        effective_from=NOW.date(),
        root=FlagCondition(description=f"Flag {flag.value}", flag=flag),
        source_ids=["test-source"],
        review_status=ReviewStatus.APPROVED,
        reviewed_at=NOW,
    )
    evaluations = evaluate_all(
        [ProgramRuleset(program_uuid("nj-snap"), "nj-snap", ruleset)], UserProfile(), now=NOW
    )
    questions = missing_field_questions(evaluations)
    assert len(questions) == 1
    assert questions[0].field == f"flags.{flag.value}"
    assert len(questions[0].options) == 2


def test_income_period_alone_never_becomes_a_question() -> None:
    """``income_period`` is part of the range answer, not a question of its own."""
    evaluations = evaluate(
        [
            entry(
                "nj-usf",
                IncomeAbsoluteCondition(
                    description="Under $4,000 a month", max_monthly_cents=400_000
                ),
            )
        ],
        UserProfile(),
    )
    fields = [question.field for question in missing_field_questions(evaluations, limit=10)]
    assert fields == ["income"]
    assert "income_period" not in fields


def test_a_profile_with_an_income_period_but_no_amount_is_impossible() -> None:
    """Guards the assumption the question copy relies on: an amount always has
    a period, so asking for a range is always answerable."""
    profile = UserProfile(income_period=IncomePeriod.MONTHLY)
    assert profile.income_monthly_range_cents() == (None, None)
    assert not profile.has_income_information()
