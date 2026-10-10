"""Unit tests for the retrieval score and the scoping rules it applies.

Everything in here is the deterministic half of retrieval: the parts that are
pure functions of a profile and a catalog record. The SQL half - full-text
matching, the hard geographic filters and the query counts - needs a real
Postgres and lives in ``tests/integration/test_retrieval_db.py``.

Two properties are worth more than the individual assertions below.

**Ranking is reproducible.** The score is a weighted sum of three named
components with no clock, no randomness and no model in it, so the same catalog
and the same profile always produce the same order. Ties break on ``slug``,
which is unique.

**A missing profile field never hides a program.** Several tests assert the
negative case explicitly: an unstated student status still sees campus
programs, and an unstated county still sees county-scoped ones. Retrieval is
allowed to be generous because a candidate the engine rules out costs one
evaluation, whereas a candidate never retrieved is help the user never hears
about.

Every program here is visibly fictional, as in ``tests/conftest.py``.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import SqlProgramRepository
from app.retrieval.citations import SourceCitationVerifier
from app.retrieval.search import (
    NEED_QUERY_TERMS,
    WEIGHT_NEED_OVERLAP,
    WEIGHT_SOURCE_FRESHNESS,
    WEIGHT_TEXT_MATCH,
    HybridRetriever,
    RetrievalScore,
    _is_student,
    _matched_terms,
    _need_overlap,
    _tokenise,
    get_program_detail,
    score_program,
    search_programs,
)
from app.schemas import (
    AssistanceProgram,
    CitationVerifier,
    GeoScope,
    NeedCategory,
    ProfileFlag,
    ProgramCategory,
    ProgramSource,
    Retriever,
    SourceFreshness,
    StudentStatus,
    UserProfile,
    VerificationStatus,
    program_uuid,
)
from tests.conftest import make_profile

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def _source(slug: str, *, age_days: int = 3) -> ProgramSource:
    retrieved = NOW - timedelta(days=age_days)
    return ProgramSource(
        id=f"{slug}-source",
        url=f"https://example.org/{slug}/requirements",
        title=f"{slug} requirements (fictional test fixture)",
        publisher="Example Test Publisher",
        excerpt="A fictional excerpt that describes no real program.",
        retrieved_at=retrieved,
        content_hash=hashlib.sha256(slug.encode()).hexdigest(),
        freshness=SourceFreshness.FRESH,
    )


def _program(
    slug: str,
    *,
    category: ProgramCategory = ProgramCategory.FOOD,
    keywords: Sequence[str] = ("groceries", "nutrition assistance"),
    counties: Sequence[str] = (),
    campuses: Sequence[str] = (),
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
    source_age_days: int = 3,
    with_sources: bool = True,
) -> AssistanceProgram:
    """A visibly fictional program with exactly the attributes under test."""
    sources = [_source(slug, age_days=source_age_days)] if with_sources else []
    return AssistanceProgram(
        id=program_uuid(slug),
        slug=slug,
        official_name=f"Example {slug.replace('-', ' ').title()} - fictional test program",
        administering_org="Example County Office of Testing",
        category=category,
        description="A fictional program invented for the automated test suite.",
        plain_language_summary="A made-up program used only to test retrieval.",
        who_it_helps=["Fictional households"],
        geo_scope=GeoScope(
            nationwide=False,
            states=["NJ"],
            counties=list(counties),
            campuses=list(campuses),
            label="Fictional scope used in tests",
        ),
        official_url=f"https://example.org/{slug}",
        keywords=list(keywords),
        verification_status=verification_status,
        last_verified_at=NOW if verification_status is VerificationStatus.VERIFIED else None,
        active_rule_version="2026.1",
        sources=sources,
    )


def _score(
    program: AssistanceProgram, profile: UserProfile, *, ts_rank: float = 0.0
) -> RetrievalScore:
    return score_program(program, profile=profile, ts_rank=ts_rank, now=NOW)


class TestTheScoreFormula:
    """The weighted sum, and the properties that keep it interpretable."""

    def test_the_weights_sum_to_one(self) -> None:
        """So the total needs no clamping to stay a ConfidenceScore."""
        total = WEIGHT_TEXT_MATCH + WEIGHT_NEED_OVERLAP + WEIGHT_SOURCE_FRESHNESS
        assert total == pytest.approx(1.0)

    def test_the_total_is_the_weighted_sum_of_its_components(self) -> None:
        score = _score(_program("example-food"), make_profile(), ts_rank=0.3)
        expected = (
            WEIGHT_TEXT_MATCH * score.text_match
            + WEIGHT_NEED_OVERLAP * score.need_overlap
            + WEIGHT_SOURCE_FRESHNESS * score.source_freshness
        )
        assert score.total == pytest.approx(expected, abs=1e-6)

    def test_text_match_is_monotone_in_ts_rank_and_never_saturates(self) -> None:
        """A better lexical match always ranks higher, with no ceiling collapse."""
        program = _program("example-food")
        profile = make_profile()
        ranks = [0.0, 0.05, 0.12, 0.24, 0.61, 1.2, 8.0]
        matches = [_score(program, profile, ts_rank=rank).text_match for rank in ranks]
        assert matches == sorted(matches)
        assert len(set(matches)) == len(matches)
        assert matches[0] == 0.0
        assert matches[-1] < 1.0

    def test_every_component_stays_inside_the_unit_interval(self) -> None:
        program = _program("example-food")
        profile = make_profile()
        score = _score(program, profile, ts_rank=50.0)
        for value in (score.total, score.text_match, score.need_overlap, score.source_freshness):
            assert 0.0 <= value <= 1.0

    def test_scoring_is_deterministic(self) -> None:
        program = _program("example-food")
        profile = make_profile()
        first = _score(program, profile, ts_rank=0.42)
        second = _score(program, profile, ts_rank=0.42)
        assert first == second

    def test_the_breakdown_explains_itself(self) -> None:
        score = _score(_program("example-food"), make_profile(), ts_rank=0.3)
        explanation = score.explain()
        assert "total=" in explanation
        assert "ts_rank=" in explanation


class TestNeedOverlap:
    """Stated needs against the program's category and keywords."""

    def test_a_primary_category_outscores_a_secondary_one(self) -> None:
        """NEED_TO_PROGRAM_CATEGORIES lists the primary mapping first."""
        profile = make_profile(needs=[NeedCategory.FOOD])
        primary = _need_overlap(
            _program("example-food", category=ProgramCategory.FOOD),
            profile=profile,
            categories=None,
        )
        secondary = _need_overlap(
            _program("example-community", category=ProgramCategory.COMMUNITY_SERVICES),
            profile=profile,
            categories=None,
        )
        assert primary > secondary > 0.0

    def test_an_unrelated_category_scores_zero_without_being_excluded(self) -> None:
        profile = make_profile(needs=[NeedCategory.FOOD])
        program = _program("example-legal", category=ProgramCategory.HOUSING, keywords=("rent",))
        assert _need_overlap(program, profile=profile, categories=None) == 0.0

    def test_keyword_overlap_breaks_ties_inside_a_category(self) -> None:
        profile = make_profile(needs=[NeedCategory.FOOD])
        wordy = _program(
            "example-pantry",
            keywords=("food", "groceries", "meals", "nutrition"),
        )
        bare = _program("example-sparse", keywords=("widgets",))
        assert _need_overlap(wordy, profile=profile, categories=None) > _need_overlap(
            bare, profile=profile, categories=None
        )

    def test_an_explicitly_requested_category_counts_as_primary(self) -> None:
        profile = make_profile(needs=[])
        program = _program("example-childcare", category=ProgramCategory.CHILDCARE)
        overlap = _need_overlap(program, profile=profile, categories=[ProgramCategory.CHILDCARE])
        assert overlap == pytest.approx(0.70)

    def test_a_profile_with_no_needs_scores_every_program_the_same(self) -> None:
        """The component is neutral, not punitive, when nothing was asked for."""
        profile = make_profile(needs=[])
        overlaps = {
            _need_overlap(_program(slug, category=category), profile=profile, categories=None)
            for slug, category in (
                ("example-food", ProgramCategory.FOOD),
                ("example-rent", ProgramCategory.HOUSING),
            )
        }
        assert overlaps == {0.0}

    def test_every_need_category_has_query_terms_defined(self) -> None:
        """A new NeedCategory must not silently lose its lexical expansion."""
        assert set(NEED_QUERY_TERMS) == set(NeedCategory)


class TestVerificationAndFreshness:
    """Unverified leads are surfaced, and ranked below verified programs."""

    def test_an_unverified_program_ranks_below_an_identical_verified_one(self) -> None:
        profile = make_profile(needs=[NeedCategory.FOOD])
        verified = _score(_program("example-verified"), profile, ts_rank=0.3)
        unverified = _score(
            _program("example-unverified", verification_status=VerificationStatus.UNVERIFIED),
            profile,
            ts_rank=0.3,
        )
        assert unverified.total < verified.total
        assert unverified.verification_factor < verified.verification_factor

    def test_an_unverified_program_cannot_support_an_eligibility_claim(self) -> None:
        program = _program("example-lead", verification_status=VerificationStatus.UNVERIFIED)
        assert program.supports_eligibility_claims() is False

    def test_a_stale_source_scores_below_a_fresh_one(self) -> None:
        profile = make_profile()
        fresh = _score(_program("example-fresh", source_age_days=3), profile)
        aging = _score(_program("example-aging", source_age_days=120), profile)
        stale = _score(_program("example-stale", source_age_days=500), profile)
        assert fresh.freshness > aging.freshness > stale.freshness

    def test_a_program_with_no_source_scores_zero_freshness(self) -> None:
        """A sourceless program is still a lead; it just supports no claim."""
        bare = _program(
            "example-bare",
            verification_status=VerificationStatus.UNVERIFIED,
            with_sources=False,
        )
        score = _score(bare, make_profile())
        assert score.freshness == 0.0
        assert score.source_freshness == 0.0

    def test_freshness_is_recomputed_from_retrieved_at_not_read_from_the_row(self) -> None:
        """The stored freshness column is a snapshot and goes out of date."""
        program = _program("example-ages", source_age_days=500)
        assert program.sources[0].freshness is SourceFreshness.FRESH
        assert _score(program, make_profile()).freshness < 0.5


class TestRankingOrder:
    """Several fictional profiles, each with a sensible, stable ordering."""

    @staticmethod
    def _ranked(
        candidates: Sequence[tuple[AssistanceProgram, float]], profile: UserProfile
    ) -> list[str]:
        scored = [
            (program.slug, score_program(program, profile=profile, ts_rank=rank, now=NOW).total)
            for program, rank in candidates
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [slug for slug, _ in scored]

    def test_a_hungry_renter_sees_food_before_unrelated_help(self) -> None:
        profile = make_profile(needs=[NeedCategory.FOOD])
        candidates = [
            (_program("example-food-stamps", category=ProgramCategory.FOOD), 0.24),
            (_program("example-pantry", category=ProgramCategory.COMMUNITY_SERVICES), 0.12),
            (_program("example-tax-credit", category=ProgramCategory.CASH_TAX_CREDITS), 0.12),
        ]
        assert self._ranked(candidates, profile) == [
            "example-food-stamps",
            "example-pantry",
            "example-tax-credit",
        ]

    def test_a_student_facing_an_emergency_sees_campus_aid_first(self) -> None:
        profile = make_profile(
            student_status=StudentStatus.UNDERGRAD_FULL_TIME,
            needs=[NeedCategory.EMERGENCY_EXPENSE],
        )
        candidates = [
            (
                _program(
                    "example-campus-grant",
                    category=ProgramCategory.UNIVERSITY_EMERGENCY_AID,
                    keywords=("emergency", "grant", "crisis"),
                    campuses=("Example University-Test Campus",),
                ),
                0.24,
            ),
            (_program("example-groceries", category=ProgramCategory.FOOD), 0.24),
        ]
        assert self._ranked(candidates, profile)[0] == "example-campus-grant"

    def test_a_profile_that_stated_no_needs_falls_back_to_the_text_match(self) -> None:
        profile = make_profile(needs=[])
        candidates = [
            (_program("example-weak-match"), 0.05),
            (_program("example-strong-match"), 0.61),
        ]
        assert self._ranked(candidates, profile) == [
            "example-strong-match",
            "example-weak-match",
        ]

    def test_ties_break_on_slug_so_the_order_is_total(self) -> None:
        profile = make_profile(needs=[NeedCategory.FOOD])
        candidates = [
            (_program("example-zebra"), 0.24),
            (_program("example-aardvark"), 0.24),
        ]
        assert self._ranked(candidates, profile) == ["example-aardvark", "example-zebra"]
        assert self._ranked(list(reversed(candidates)), profile) == [
            "example-aardvark",
            "example-zebra",
        ]


class TestStudentScoping:
    """Campus scope is a claim about the program; student status may be unknown."""

    def test_an_explicit_non_student_is_recognised(self) -> None:
        assert _is_student(make_profile(student_status=StudentStatus.NOT_A_STUDENT)) is False

    def test_an_unstated_student_status_is_unknown_not_false(self) -> None:
        assert _is_student(make_profile(student_status=None)) is None
        assert _is_student(make_profile(student_status=StudentStatus.UNKNOWN)) is None

    def test_every_enrolled_status_counts_as_a_student(self) -> None:
        for status in (
            StudentStatus.UNDERGRAD_FULL_TIME,
            StudentStatus.UNDERGRAD_PART_TIME,
            StudentStatus.GRADUATE,
        ):
            assert _is_student(make_profile(student_status=status)) is True

    def test_an_other_student_status_is_not_treated_as_a_denial(self) -> None:
        """ "Other" says the options did not fit, not that they are not a student."""
        assert _is_student(make_profile(student_status=StudentStatus.OTHER)) is None

    def test_the_rutgers_flag_wins_over_a_denied_student_status(self) -> None:
        """Staff and affiliates are not students but are in scope on campus."""
        profile = make_profile(
            student_status=StudentStatus.NOT_A_STUDENT,
            flags={ProfileFlag.IS_RUTGERS_AFFILIATED: True},
        )
        assert _is_student(profile) is True


class TestQueryTokenisation:
    """What reaches ``plainto_tsquery``."""

    def test_stopwords_and_single_characters_are_dropped(self) -> None:
        assert _tokenise("I need help with the groceries") == ["groceries"]

    def test_tokens_are_deduplicated_in_first_seen_order(self) -> None:
        assert _tokenise("rent rent eviction rent") == ["rent", "eviction"]

    def test_punctuation_and_accents_are_folded(self) -> None:
        assert _tokenise("child-care; café!") == ["child", "care", "cafe"]

    def test_a_query_of_nothing_but_stopwords_yields_no_text_signal(self) -> None:
        """Which degrades to 'no query', never to 'no results'."""
        assert _tokenise("and then the of it") == []

    def test_an_empty_query_is_handled(self) -> None:
        assert _tokenise("") == []
        assert _tokenise("   ") == []

    def test_matched_terms_report_a_keyword_hit_through_a_crude_stem(self) -> None:
        program = _program("example-food", keywords=("grocery", "nutrition"))
        assert _matched_terms(["groceries", "widgets"], program) == ["groceries"]


class TestContractConformance:
    """The seams the rest of the system depends on, not the classes."""

    @staticmethod
    def _retriever() -> HybridRetriever:
        """An instance over a session that is never used.

        ``HybridRetriever.__init__`` only reads the repository's session
        attribute, so this is enough to assert on the shape of the seam without
        a database.
        """
        return HybridRetriever(SqlProgramRepository(cast("AsyncSession", object())))

    def test_the_retriever_protocol_is_satisfied(self) -> None:
        assert isinstance(self._retriever(), Retriever)

    def test_the_citation_verifier_protocol_is_satisfied(self) -> None:
        verifier = SourceCitationVerifier(SqlProgramRepository(cast("AsyncSession", object())))
        assert isinstance(verifier, CitationVerifier)

    def test_the_documented_import_path_resolves_to_the_implementation(self) -> None:
        """``docs/INTERFACES.md`` names ``app.retrieval.hybrid``."""
        from app.retrieval import hybrid

        assert hybrid.HybridRetriever is HybridRetriever

    def test_the_search_entry_point_keeps_its_keyword_only_signature(self) -> None:
        """Callers pass these by name; renaming one is a breaking change."""
        signature = inspect.signature(search_programs)
        assert list(signature.parameters) == [
            "session",
            "profile",
            "query",
            "categories",
            "limit",
        ]
        for name in ("profile", "query", "categories", "limit"):
            assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["limit"].default == 20

    def test_the_detail_entry_point_takes_a_slug_or_an_id(self) -> None:
        signature = inspect.signature(get_program_detail)
        assert list(signature.parameters) == ["session", "slug_or_id"]


def _typed_seams(
    retriever: HybridRetriever, verifier: SourceCitationVerifier
) -> tuple[Retriever, CitationVerifier]:
    """Checked by mypy, not at runtime: the signatures match the Protocols.

    A ``runtime_checkable`` Protocol only compares method names, so an argument
    renamed here would pass ``isinstance`` and fail in production. This function
    is never called; its return annotation is the assertion.
    """
    return retriever, verifier
