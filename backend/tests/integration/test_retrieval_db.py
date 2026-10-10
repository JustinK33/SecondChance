"""Integration tests for retrieval and citation verification against Postgres.

What needs a real database here is not "does the ORM work". It is the half of
retrieval that only Postgres can answer:

* the generated ``search_vector`` really does match a program through a
  *synonym stored in its keywords*, with the same text configuration the
  column was built with;
* the geographic filters are real ``jsonb`` containment tests, so a program in
  another county is genuinely absent from the result set rather than merely
  ranked lower;
* the number of SQL statements a search or a detail load issues is bounded and
  does not grow with the number of rows - the assertion that an N+1 cannot
  creep back in;
* a citation is checked against the stored row, which is the only authority on
  what a source says and which program it belongs to.

Every program here is visibly fictional, as in ``tests/conftest.py``: the URLs
are ``example.org`` and the names say so. The curated catalog in
``backend/data/programs`` is the only place real programs belong.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import date, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import SqlProgramRepository
from app.retrieval.citations import (
    SourceCitationVerifier,
    load_catalog_sources,
    verify_citations,
)
from app.retrieval.search import (
    HybridRetriever,
    get_program_detail,
    search_programs,
    search_scored_programs,
)
from app.schemas import (
    AssistanceProgram,
    Citation,
    GeoScope,
    NeedCategory,
    ProgramCategory,
    ProgramSource,
    RetrievedProgram,
    ReviewStatus,
    RuleSet,
    SourceFreshness,
    StudentStatus,
    UserProfile,
    VerificationStatus,
    program_uuid,
    utcnow,
)
from tests.conftest import make_profile, make_ruleset

pytestmark = pytest.mark.db

#: Statements that are part of transaction bookkeeping rather than work the
#: code under test asked for.
_COUNTED = re.compile(r"^\s*(select|insert|update|delete)\b", re.I)


@asynccontextmanager
async def count_statements(session: AsyncSession) -> AsyncIterator[list[str]]:
    """Collect the SQL statements issued inside the block.

    Counts only data statements: the test session is savepoint-joined (see
    ``tests/conftest.py``), so ``SAVEPOINT`` and ``RELEASE`` traffic would
    otherwise be counted as if the code under test had asked for it.
    """
    connection = await session.connection()
    engine = connection.sync_engine
    statements: list[str] = []

    def before_cursor_execute(
        conn: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        if _COUNTED.match(statement):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)


def _source(
    source_id: str,
    *,
    url: str | None = None,
    excerpt: str = (
        "A fictional excerpt used only by the automated tests. The maximum "
        "allowable monthly income is $2,461 for a household of one."
    ),
    age_days: int = 5,
) -> ProgramSource:
    return ProgramSource(
        id=source_id,
        url=url or f"https://example.org/{source_id}",
        title=f"{source_id} (fictional test fixture)",
        publisher="Example Test Publisher",
        excerpt=excerpt,
        retrieved_at=utcnow() - timedelta(days=age_days),
        content_hash=hashlib.sha256(source_id.encode()).hexdigest(),
        freshness=SourceFreshness.FRESH,
    )


def _program(
    slug: str,
    *,
    category: ProgramCategory = ProgramCategory.FOOD,
    keywords: Sequence[str] = ("groceries", "nutrition assistance"),
    states: Sequence[str] = ("NJ",),
    counties: Sequence[str] = (),
    campuses: Sequence[str] = (),
    nationwide: bool = False,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
    active_rule_version: str | None = "2026.1",
    sources: Sequence[ProgramSource] | None = None,
) -> AssistanceProgram:
    """A visibly fictional program with exactly the attributes under test."""
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
            nationwide=nationwide,
            states=list(states),
            counties=list(counties),
            campuses=list(campuses),
            label="Fictional scope used in tests",
        ),
        official_url=f"https://example.org/{slug}",
        application_url=f"https://example.org/{slug}/apply",
        keywords=list(keywords),
        verification_status=verification_status,
        last_verified_at=(utcnow() if verification_status is VerificationStatus.VERIFIED else None),
        active_rule_version=active_rule_version,
        sources=list(sources) if sources is not None else [_source(f"{slug}-source")],
    )


async def _seed(
    session: AsyncSession,
    program: AssistanceProgram,
    *,
    ruleset: RuleSet | None = None,
) -> uuid.UUID:
    """Insert one program and, unless told otherwise, its approved rule set."""
    repository = SqlProgramRepository(session)
    program_id = await repository.upsert_program(program)
    rules = ruleset if ruleset is not None else make_ruleset(program.slug)
    await repository.upsert_rule_version(rules)
    await session.flush()
    return program_id


def _slugs(candidates: Sequence[RetrievedProgram]) -> list[str]:
    return [candidate.program.slug for candidate in candidates]


@pytest.fixture
def nj_profile() -> UserProfile:
    return make_profile(needs=[NeedCategory.FOOD])


class TestFullTextSearch:
    """The generated ``search_vector``, queried the way a person types."""

    async def test_a_program_is_found_by_a_synonym_in_its_keywords(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        """ "Nutrition" appears only in the keywords, never in the prose."""
        await _seed(
            session,
            _program("example-grocery-help", keywords=("nutrition assistance", "ebt")),
        )
        found = await search_programs(session, profile=nj_profile, query="nutrition")
        assert _slugs(found) == ["example-grocery-help"]
        assert "nutrition" in found[0].matched_terms

    async def test_stemming_matches_a_plural_against_a_singular_keyword(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(session, _program("example-grocery-help", keywords=("grocery",)))
        found = await search_programs(session, profile=nj_profile, query="groceries")
        assert _slugs(found) == ["example-grocery-help"]

    async def test_a_multi_word_query_does_not_require_every_word(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        """``plainto_tsquery`` ANDs its terms, so the terms are OR-ed instead."""
        await _seed(session, _program("example-grocery-help", keywords=("groceries",)))
        found = await search_programs(
            session, profile=nj_profile, query="help with groceries and childcare"
        )
        assert _slugs(found) == ["example-grocery-help"]

    async def test_an_explicit_query_is_a_hard_filter(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(session, _program("example-grocery-help"))
        found = await search_programs(session, profile=nj_profile, query="submarine periscope")
        assert found == []

    async def test_a_query_of_only_stopwords_falls_back_to_the_profile(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        """Degrades to "no text signal", never to "no results"."""
        await _seed(session, _program("example-grocery-help"))
        found = await search_programs(session, profile=nj_profile, query="and then the of")
        assert _slugs(found) == ["example-grocery-help"]

    async def test_punctuation_in_a_query_does_not_raise(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(session, _program("example-grocery-help"))
        found = await search_programs(session, profile=nj_profile, query="groceries!? & <rent>")
        assert _slugs(found) == ["example-grocery-help"]


class TestRankingOrder:
    """Deterministic, and sensible for several fictional profiles."""

    async def _seed_catalog(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-food-benefit", category=ProgramCategory.FOOD))
        await _seed(
            session,
            _program(
                "example-pantry-network",
                category=ProgramCategory.COMMUNITY_SERVICES,
                keywords=("food pantry", "meals"),
            ),
        )
        await _seed(
            session,
            _program(
                "example-rent-relief",
                category=ProgramCategory.HOUSING,
                keywords=("rent", "eviction"),
            ),
        )
        await _seed(
            session,
            _program(
                "example-campus-emergency-grant",
                category=ProgramCategory.UNIVERSITY_EMERGENCY_AID,
                keywords=("emergency", "grant"),
                campuses=("Example University-Test Campus",),
            ),
        )

    async def test_a_hungry_household_sees_food_help_first(self, session: AsyncSession) -> None:
        await self._seed_catalog(session)
        profile = make_profile(needs=[NeedCategory.FOOD])
        order = _slugs(await search_programs(session, profile=profile))
        assert order[0] == "example-food-benefit"
        assert order.index("example-pantry-network") < order.index("example-rent-relief")

    async def test_a_household_facing_eviction_sees_housing_help_first(
        self, session: AsyncSession
    ) -> None:
        await self._seed_catalog(session)
        profile = make_profile(needs=[NeedCategory.HOUSING])
        order = _slugs(await search_programs(session, profile=profile))
        assert order[0] == "example-rent-relief"

    async def test_a_student_with_an_emergency_sees_campus_aid_first(
        self, session: AsyncSession
    ) -> None:
        await self._seed_catalog(session)
        profile = make_profile(
            student_status=StudentStatus.GRADUATE,
            needs=[NeedCategory.EMERGENCY_EXPENSE],
        )
        order = _slugs(await search_programs(session, profile=profile))
        assert order[0] == "example-campus-emergency-grant"

    async def test_the_same_profile_twice_gives_exactly_the_same_order(
        self, session: AsyncSession
    ) -> None:
        await self._seed_catalog(session)
        profile = make_profile(needs=[NeedCategory.FOOD])
        first = _slugs(await search_programs(session, profile=profile))
        second = _slugs(await search_programs(session, profile=profile))
        assert first == second
        assert len(first) == 4

    async def test_scores_are_monotonically_non_increasing(self, session: AsyncSession) -> None:
        await self._seed_catalog(session)
        found = await search_programs(session, profile=make_profile())
        scores = [candidate.retrieval_score for candidate in found]
        assert scores == sorted(scores, reverse=True)

    async def test_a_profile_with_nothing_stated_still_gets_candidates(
        self, session: AsyncSession
    ) -> None:
        """No state, no county, no needs and no query: everything is in scope.

        There is no text signal at all here, so the only thing separating the
        candidates is the record-quality component and then the slug. An empty
        result would be the worst possible answer to "I do not know where to
        start".
        """
        await self._seed_catalog(session)
        found = await search_programs(session, profile=UserProfile())
        assert len(found) == 4
        assert _slugs(found) == sorted(_slugs(found))

    async def test_the_limit_is_respected(self, session: AsyncSession) -> None:
        await self._seed_catalog(session)
        found = await search_programs(session, profile=make_profile(), limit=2)
        assert len(found) == 2

    async def test_the_score_components_come_back_with_the_total(
        self, session: AsyncSession
    ) -> None:
        await self._seed_catalog(session)
        scored = await search_scored_programs(
            session, profile=make_profile(needs=[NeedCategory.FOOD]), query="groceries"
        )
        best = scored[0]
        assert best.score.ts_rank > 0.0
        assert best.score.text_match > 0.0
        assert best.score.need_overlap > 0.0
        assert best.score.total == best.candidate.retrieval_score
        assert "total=" in best.score.explain()


class TestGeographicScoping:
    """A hard filter, because it is a claim the catalog makes."""

    async def test_a_program_in_another_county_is_excluded(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-bergen-only", counties=("Bergen",)))
        await _seed(session, _program("example-middlesex-only", counties=("Middlesex",)))
        found = await search_programs(session, profile=make_profile(county="Middlesex"))
        assert _slugs(found) == ["example-middlesex-only"]

    async def test_a_statewide_program_matches_every_county(self, session: AsyncSession) -> None:
        """Otherwise filtering by county would hide SNAP."""
        await _seed(session, _program("example-statewide", counties=()))
        found = await search_programs(session, profile=make_profile(county="Ocean"))
        assert _slugs(found) == ["example-statewide"]

    async def test_a_county_name_written_with_the_suffix_still_matches(
        self, session: AsyncSession
    ) -> None:
        await _seed(session, _program("example-suffixed", counties=("Middlesex County",)))
        found = await search_programs(session, profile=make_profile(county="Middlesex"))
        assert _slugs(found) == ["example-suffixed"]

    async def test_an_unstated_county_excludes_nothing(self, session: AsyncSession) -> None:
        """Unknown is not "no": a county we were never told cannot disqualify."""
        await _seed(session, _program("example-bergen-only", counties=("Bergen",)))
        await _seed(session, _program("example-statewide", counties=()))
        found = await search_programs(session, profile=make_profile(county=None))
        assert sorted(_slugs(found)) == ["example-bergen-only", "example-statewide"]

    async def test_a_program_in_another_state_is_excluded(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-new-york", states=("NY",)))
        await _seed(session, _program("example-new-jersey", states=("NJ",)))
        found = await search_programs(session, profile=make_profile(state="NJ", county=None))
        assert _slugs(found) == ["example-new-jersey"]

    async def test_a_nationwide_program_is_in_scope_everywhere(self, session: AsyncSession) -> None:
        await _seed(
            session, _program("example-nationwide", nationwide=True, states=(), counties=())
        )
        found = await search_programs(session, profile=make_profile(state="NJ"))
        assert _slugs(found) == ["example-nationwide"]


class TestStudentScoping:
    """Campus scope filters only when the user said they are not a student."""

    async def _seed_both(self, session: AsyncSession) -> None:
        await _seed(
            session,
            _program(
                "example-campus-grant",
                category=ProgramCategory.UNIVERSITY_EMERGENCY_AID,
                campuses=("Example University-Test Campus",),
            ),
        )
        await _seed(session, _program("example-open-to-all"))

    async def test_an_explicit_non_student_does_not_see_campus_programs(
        self, session: AsyncSession
    ) -> None:
        await self._seed_both(session)
        profile = make_profile(student_status=StudentStatus.NOT_A_STUDENT)
        assert _slugs(await search_programs(session, profile=profile)) == ["example-open-to-all"]

    async def test_a_student_sees_both(self, session: AsyncSession) -> None:
        await self._seed_both(session)
        profile = make_profile(student_status=StudentStatus.UNDERGRAD_PART_TIME)
        assert sorted(_slugs(await search_programs(session, profile=profile))) == [
            "example-campus-grant",
            "example-open-to-all",
        ]

    async def test_an_unstated_student_status_still_sees_campus_programs(
        self, session: AsyncSession
    ) -> None:
        """Hiding emergency aid from a student who only mentioned rent is the
        failure this test exists to prevent."""
        await self._seed_both(session)
        profile = make_profile(student_status=None, unknown_fields=["student_status"])
        assert sorted(_slugs(await search_programs(session, profile=profile))) == [
            "example-campus-grant",
            "example-open-to-all",
        ]


class TestVerificationAndRules:
    """What a candidate is allowed to be used for."""

    async def test_an_unverified_program_is_returned_but_supports_no_claim(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(
            session,
            _program(
                "example-unverified-lead",
                verification_status=VerificationStatus.UNVERIFIED,
            ),
        )
        found = await search_programs(session, profile=nj_profile)
        assert _slugs(found) == ["example-unverified-lead"]
        candidate = found[0]
        assert candidate.program.supports_eligibility_claims() is False
        # The flag is structural, not advisory: with no rule set the engine
        # reports "cannot assess" and cannot produce a status at all.
        assert candidate.ruleset is None
        assert any("not finished checking" in reason for reason in candidate.retrieval_reasons)

    async def test_a_verified_program_arrives_with_its_approved_rule_set(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(session, _program("example-grocery-help"))
        candidate = (await search_programs(session, profile=nj_profile))[0]
        assert candidate.ruleset is not None
        assert candidate.ruleset.version == "2026.1"
        assert candidate.ruleset.review_status is ReviewStatus.APPROVED
        assert candidate.ruleset.program_slug == "example-grocery-help"
        assert candidate.sources

    async def test_an_unapproved_rule_set_is_never_attached(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        """A draft requirement must not reach a user as an assessment."""
        await _seed(
            session,
            _program("example-draft-rules"),
            ruleset=make_ruleset("example-draft-rules", review_status=ReviewStatus.DRAFT),
        )
        candidate = (await search_programs(session, profile=nj_profile))[0]
        assert candidate.ruleset is None

    async def test_a_rule_set_not_yet_in_force_is_not_attached(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        future = date.today() + timedelta(days=30)
        await _seed(
            session,
            _program("example-future-rules"),
            ruleset=make_ruleset("example-future-rules", effective_from=future),
        )
        candidate = (await search_programs(session, profile=nj_profile))[0]
        assert candidate.ruleset is None

    async def test_a_program_without_an_active_version_is_still_surfaced(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(session, _program("example-unrated", active_rule_version=None))
        candidate = (await search_programs(session, profile=nj_profile))[0]
        assert candidate.ruleset is None
        assert candidate.program.supports_eligibility_claims() is False

    async def test_the_newest_rule_version_in_force_wins(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        """Two versions can be in force at once, and the later one applies.

        A version with a future ``effective_to`` overlaps the open version that
        superseded it, which the partial unique index permits because only one
        of them is open. "What are the rules today" must still have one answer.
        """
        slug = "example-overlapping-rules"
        await _seed(
            session,
            _program(slug),
            ruleset=make_ruleset(
                slug,
                version="2026.1",
                effective_from=date(2026, 1, 1),
                effective_to=date(2027, 1, 1),
            ),
        )
        repository = SqlProgramRepository(session)
        await repository.upsert_rule_version(
            make_ruleset(slug, version="2026.2", effective_from=date(2026, 6, 1))
        )
        await session.flush()

        candidate = (await search_programs(session, profile=nj_profile))[0]
        assert candidate.ruleset is not None
        assert candidate.ruleset.version == "2026.2"

    async def test_a_soft_deleted_program_leaves_the_results(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        program_id = await _seed(session, _program("example-retired"))
        repository = SqlProgramRepository(session)
        await repository.soft_delete_program(program_id)
        assert await search_programs(session, profile=nj_profile) == []
        # ...but a plan saved last month must still render its name and link.
        detail = await get_program_detail(session, "example-retired")
        assert detail is not None


class TestCategoryFilter:
    async def test_an_explicit_category_is_a_hard_filter(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-food-benefit", category=ProgramCategory.FOOD))
        await _seed(session, _program("example-rent-relief", category=ProgramCategory.HOUSING))
        found = await search_programs(
            session, profile=make_profile(), categories=[ProgramCategory.HOUSING]
        )
        assert _slugs(found) == ["example-rent-relief"]


class TestStatementCounts:
    """The N+1 guards. These are the assertions that stop a regression."""

    async def test_search_issues_a_bounded_number_of_statements(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        for index in range(6):
            await _seed(session, _program(f"example-program-{index}"))
        async with count_statements(session) as statements:
            found = await search_programs(session, profile=nj_profile, query="groceries")
        assert len(found) == 6
        # ranked ids, the programs, their sources (selectin), the rule versions.
        assert len(statements) == 4

    async def test_the_statement_count_does_not_grow_with_the_result_count(
        self, session: AsyncSession, nj_profile: UserProfile
    ) -> None:
        await _seed(session, _program("example-only-one"))
        async with count_statements(session) as few:
            await search_programs(session, profile=nj_profile)
        for index in range(8):
            await _seed(session, _program(f"example-program-{index}"))
        async with count_statements(session) as many:
            results = await search_programs(session, profile=nj_profile)
        assert len(results) == 9
        assert len(many) == len(few)

    async def test_the_detail_query_is_bounded(self, session: AsyncSession) -> None:
        slug = "example-well-documented"
        program = _program(
            slug,
            sources=[
                _source(f"{slug}-source"),
                _source(f"{slug}-second"),
                _source(f"{slug}-third"),
            ],
        )
        await _seed(session, program)
        repository = SqlProgramRepository(session)
        await repository.upsert_rule_version(
            make_ruleset(
                slug,
                version="2025.1",
                effective_from=date(2025, 1, 1),
                effective_to=date(2026, 1, 1),
            )
        )
        await session.flush()

        async with count_statements(session) as statements:
            detail = await get_program_detail(session, slug)
        assert detail is not None
        assert len(detail.sources) == 3
        assert detail.rule_version == "2026.1"
        # the program, its sources (selectin), the in-force rule version.
        assert len(statements) == 3

    async def test_the_detail_query_accepts_a_slug_or_an_id(self, session: AsyncSession) -> None:
        program_id = await _seed(session, _program("example-grocery-help"))
        by_slug = await get_program_detail(session, "example-grocery-help")
        by_id = await get_program_detail(session, program_id)
        by_id_string = await get_program_detail(session, str(program_id))
        assert by_slug is not None
        assert by_id is not None
        assert by_id_string is not None
        assert by_slug.program.id == by_id.program.id == by_id_string.program.id

    async def test_an_unknown_slug_returns_none(self, session: AsyncSession) -> None:
        assert await get_program_detail(session, "example-does-not-exist") is None

    async def test_the_detail_view_renders_requirements_from_the_rules(
        self, session: AsyncSession
    ) -> None:
        await _seed(session, _program("example-grocery-help"))
        detail = await get_program_detail(session, "example-grocery-help")
        assert detail is not None
        assert detail.requirements
        assert detail.rule_version == "2026.1"


class TestTheRetrieverSeam:
    """The ``Retriever`` Protocol methods, over a real session."""

    async def test_retrieve_uses_the_profile_needs(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-food-benefit", category=ProgramCategory.FOOD))
        await _seed(session, _program("example-rent-relief", category=ProgramCategory.HOUSING))
        retriever = HybridRetriever.from_session(session)
        found = await retriever.retrieve(make_profile(needs=[NeedCategory.HOUSING]))
        assert _slugs(found)[0] == "example-rent-relief"

    async def test_the_needs_argument_overrides_the_profile(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-food-benefit", category=ProgramCategory.FOOD))
        await _seed(session, _program("example-rent-relief", category=ProgramCategory.HOUSING))
        retriever = HybridRetriever.from_session(session)
        found = await retriever.retrieve(
            make_profile(needs=[NeedCategory.FOOD]), needs=[NeedCategory.HOUSING]
        )
        assert _slugs(found)[0] == "example-rent-relief"

    async def test_retrieve_for_programs_rehydrates_in_the_requested_order(
        self, session: AsyncSession
    ) -> None:
        first = await _seed(session, _program("example-food-benefit"))
        second = await _seed(session, _program("example-rent-relief"))
        retriever = HybridRetriever.from_session(session)
        found = await retriever.retrieve_for_programs([second, first])
        assert _slugs(found) == ["example-rent-relief", "example-food-benefit"]
        assert all(candidate.ruleset is not None for candidate in found)

    async def test_retrieve_for_programs_with_nothing_asks_nothing(
        self, session: AsyncSession
    ) -> None:
        retriever = HybridRetriever.from_session(session)
        async with count_statements(session) as statements:
            assert await retriever.retrieve_for_programs([]) == []
        assert statements == []

    async def test_search_sources_finds_a_passage(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-grocery-help"))
        retriever = HybridRetriever.from_session(session)
        found = await retriever.search_sources("maximum allowable monthly income")
        assert [source.id for source in found] == ["example-grocery-help-source"]


class TestCitationsAgainstTheStoredCatalog:
    """The stored row is the only authority on what a source says."""

    @staticmethod
    def _citation(
        *,
        source_id: str = "example-grocery-help-source",
        program_id: uuid.UUID | None = None,
        url: str = "https://example.org/example-grocery-help-source",
        excerpt: str = "The maximum allowable monthly income is $2,461 for a household of one",
    ) -> Citation:
        return Citation(
            source_id=source_id,
            program_id=program_id,
            url=url,
            title="Example requirements page",
            excerpt=excerpt,
        )

    async def test_a_real_quotation_of_a_stored_source_passes(self, session: AsyncSession) -> None:
        program_id = await _seed(session, _program("example-grocery-help"))
        report = await verify_citations(session, [self._citation(program_id=program_id)])
        assert report.ok is True
        assert [source.id for source in report.checked_sources] == ["example-grocery-help-source"]

    async def test_an_id_that_is_not_in_the_catalog_is_rejected(
        self, session: AsyncSession
    ) -> None:
        await _seed(session, _program("example-grocery-help"))
        report = await verify_citations(session, [self._citation(source_id="invented-source")])
        assert report.ok is False
        assert report.rejected[0].reason == "unknown_source"

    async def test_attribution_to_the_wrong_program_is_rejected(
        self, session: AsyncSession
    ) -> None:
        await _seed(session, _program("example-grocery-help"))
        other = await _seed(session, _program("example-rent-relief"))
        report = await verify_citations(session, [self._citation(program_id=other)])
        assert report.ok is False
        assert report.rejected[0].reason == "program_mismatch"

    async def test_a_mangled_quotation_is_rejected(self, session: AsyncSession) -> None:
        await _seed(session, _program("example-grocery-help"))
        report = await verify_citations(
            session,
            [self._citation(excerpt="The maximum allowable monthly income is $9,999 per month")],
        )
        assert report.ok is False
        assert report.rejected[0].reason == "excerpt_not_found"

    async def test_a_caller_supplied_source_cannot_override_a_stored_one(
        self, session: AsyncSession
    ) -> None:
        """Otherwise whatever produced the citation could also vouch for it."""
        await _seed(session, _program("example-grocery-help"))
        forged = _source(
            "example-grocery-help-source",
            excerpt="Everyone in New Jersey receives this benefit automatically.",
        )
        report = await verify_citations(
            session,
            [self._citation(excerpt="Everyone in New Jersey receives this benefit")],
            known_sources=[forged],
        )
        assert report.ok is False
        assert report.rejected[0].reason == "excerpt_not_found"

    async def test_the_verifier_resolves_through_its_repository_session(
        self, session: AsyncSession
    ) -> None:
        """The contract's concrete form, over a real session."""
        program_id = await _seed(session, _program("example-grocery-help"))
        verifier = SourceCitationVerifier(SqlProgramRepository(session))
        good = await verifier.verify([self._citation(program_id=program_id)])
        assert good.ok is True
        bad = await verifier.verify([self._citation(excerpt="Invented wording nobody wrote")])
        assert bad.ok is False
        assert bad.rejected[0].reason == "excerpt_not_found"

    async def test_verifying_nothing_asks_the_database_nothing(self, session: AsyncSession) -> None:
        async with count_statements(session) as statements:
            report = await verify_citations(session, [])
        assert report.ok is True
        assert statements == []

    async def test_resolving_many_citations_takes_one_query(self, session: AsyncSession) -> None:
        slug = "example-grocery-help"
        await _seed(
            session,
            _program(
                slug,
                sources=[_source(f"{slug}-source"), _source(f"{slug}-second")],
            ),
        )
        async with count_statements(session) as statements:
            resolved = await load_catalog_sources(
                session, [f"{slug}-source", f"{slug}-second", "invented-source"]
            )
        assert len(statements) == 1
        assert set(resolved) == {f"{slug}-source", f"{slug}-second"}

    async def test_the_program_id_comes_from_the_row_not_the_citation(
        self, session: AsyncSession
    ) -> None:
        program_id = await _seed(session, _program("example-grocery-help"))
        resolved = await load_catalog_sources(session, ["example-grocery-help-source"])
        assert resolved["example-grocery-help-source"].program_id == program_id
