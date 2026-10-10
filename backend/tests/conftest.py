"""Shared test fixtures: the test database, a rolled-back session, and factories.

Database isolation
------------------
The suite runs against ``TEST_DATABASE_URL`` - a *separate* database from the
one ``make dev-api`` uses, so running the tests can never truncate a developer's
working data. Migrations are applied once per session, and each test then runs
inside a transaction that is rolled back afterwards.

The rollback trick is worth spelling out, because it is what makes the tests
both fast and honest. The session fixture opens a connection, begins a
transaction on it, and binds the ``AsyncSession`` to that *connection* with
``join_transaction_mode="create_savepoint"``. A ``session.commit()`` inside a
test therefore releases a savepoint rather than committing to the database -
which means the code under test can commit exactly as it does in production,
constraint violations surface at the moment they really would, and nothing
survives the test. Truncating tables between tests would be slower and would
not exercise ``COMMIT`` at all.

Migrations run in a subprocess rather than through ``alembic.command``. That
keeps the test process free of the ``DATABASE_URL`` override (so nothing in the
application accidentally picks up the test database through ``get_settings()``)
and it exercises migrations the same way ``make migrate`` does.

Everything the factories build is **fictional**. The product rule is that a
program, a requirement or a citation is never fabricated, so the fixtures use
``example.org`` URLs and names that are visibly invented, and nothing in
``backend/tests`` should ever be mistaken for the curated catalog in
``backend/data/programs``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

# ``backend/`` on ``sys.path``, so ``import app`` works however pytest was
# invoked. pytest's ``prepend`` import mode inserts the directory holding each
# test module - ``tests/`` here, because the test tree has no ``__init__.py`` -
# and never the project root, so without this the whole suite fails to collect
# unless the editable install happens to be intact. Doing it in the root
# conftest fixes it for every test file rather than one at a time.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import Settings  # noqa: E402
from app.db import models  # noqa: E402
from app.db.session import create_engine  # noqa: E402
from app.schemas import (  # noqa: E402
    ActionItem,
    ActionItemKind,
    ActionPlan,
    AllOf,
    AssistanceProgram,
    EligibilityEvaluation,
    EmploymentStatus,
    GeographyCondition,
    GeoScope,
    HouseholdSizeCondition,
    HousingSituation,
    IncomePeriod,
    MatchExplanation,
    MatchStatus,
    NeedCategory,
    ProgramCategory,
    ProgramMatch,
    ProgramSource,
    ReviewStatus,
    RuleSet,
    SourceFreshness,
    StudentStatus,
    Trilean,
    UserProfile,
    VerificationStatus,
    action_item_uuid,
    make_dedup_key,
    program_uuid,
    utcnow,
)

#: Role used by ``tests/integration/test_rls.py``. Must be a non-owner,
#: non-superuser role without ``BYPASSRLS``, or the policies under test are not
#: consulted at all.
RLS_TEST_ROLE = "secondchance_rls_test"


# ---------------------------------------------------------------- settings
@pytest.fixture(scope="session")
def settings() -> Settings:
    """The real configuration, read once.

    Constructed directly rather than through ``get_settings()`` so the suite
    never populates the application's cached singleton.
    """
    return Settings()


@pytest.fixture(scope="session")
def test_database_url_async(settings: Settings) -> str:
    return settings.test_database_url_async


# ------------------------------------------------------------- migrations
@pytest.fixture(scope="session")
def migrated_database(settings: Settings) -> str:
    """Apply every migration to the test database, once per test session.

    Runs in a subprocess with ``DATABASE_URL`` pointed at the test database,
    which is exactly how ``make migrate`` migrates it. A failure here is
    re-raised with alembic's own output attached, because "the tests cannot
    connect" and "the migration is broken" need to be distinguishable at a
    glance.
    """
    env = {**os.environ, "DATABASE_URL": settings.test_database_url}
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "alembic upgrade head failed against the test database.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return settings.test_database_url


# ----------------------------------------------------------------- engine
@pytest_asyncio.fixture
async def engine(settings: Settings, migrated_database: str) -> AsyncIterator[AsyncEngine]:
    """A function-scoped engine bound to the test database.

    Function-scoped on purpose. ``asyncio_default_fixture_loop_scope`` is
    ``function``, so a session-scoped engine would hold asyncpg connections
    created on an event loop that no longer exists by the second test. Building
    an engine is cheap; debugging "attached to a different loop" is not.
    """
    created = create_engine(settings, url=settings.test_database_url_async)
    try:
        yield created
    finally:
        await created.dispose()


@pytest_asyncio.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """An ``AsyncSession`` whose every write is rolled back after the test.

    See the module docstring for why this is a savepoint-joined session rather
    than a truncate-between-tests arrangement.
    """
    async with engine.connect() as connection:
        transaction = await connection.begin()
        factory = async_sessionmaker(
            bind=connection,
            expire_on_commit=False,
            autoflush=False,
            join_transaction_mode="create_savepoint",
        )
        async with factory() as db_session:
            try:
                yield db_session
            finally:
                await db_session.close()
        await transaction.rollback()


@pytest_asyncio.fixture
async def rls_role(engine: AsyncEngine) -> AsyncIterator[str]:
    """Create the non-owner role the RLS test switches into.

    Roles are cluster-wide, so this is a fixture rather than a migration: a
    migration that created a role would be creating it in whatever database it
    happened to be applied to, which is not what cluster-wide means.

    ``NOINHERIT`` and ``NOBYPASSRLS`` are the point of the role. Without them
    the policies would be bypassed and the test would pass while proving
    nothing.
    """
    grants_user = ", ".join(models.USER_OWNED_TABLES)
    grants_catalog = "assistance_programs, program_sources, program_rule_versions"
    async with engine.begin() as connection:
        # Interpolates only this module's RLS_TEST_ROLE constant.
        await connection.execute(
            text(
                f"""
                DO $do$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{RLS_TEST_ROLE}') THEN
                        CREATE ROLE {RLS_TEST_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT;
                    END IF;
                END
                $do$;
                """  # noqa: S608
            )
        )
        await connection.execute(text(f"GRANT USAGE ON SCHEMA public TO {RLS_TEST_ROLE}"))
        await connection.execute(
            text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {grants_user} TO {RLS_TEST_ROLE}")
        )
        await connection.execute(text(f"GRANT SELECT ON {grants_catalog} TO {RLS_TEST_ROLE}"))
    yield RLS_TEST_ROLE


# ------------------------------------------------------------------- users
@pytest.fixture
def user_a() -> str:
    """A fictional user id.

    UUID-shaped because the Supabase ``sub`` claim is a UUID and the RLS
    policies compare ``user_id`` to ``auth.uid()::text``.
    """
    return str(uuid.uuid4())


@pytest.fixture
def user_b() -> str:
    """A second fictional user id, for cross-user isolation assertions."""
    return str(uuid.uuid4())


# --------------------------------------------------------------- factories
def _fictional_source(slug: str, *, age_days: int = 3) -> ProgramSource:
    retrieved = utcnow() - timedelta(days=age_days)
    return ProgramSource(
        id=f"{slug}-source",
        url=f"https://example.org/{slug}/requirements",
        title=f"{slug} requirements (fictional test fixture)",
        publisher="Example Test Publisher",
        excerpt=(
            "This is a fictional excerpt used only by the automated tests. "
            "It describes no real program and states no real requirement."
        ),
        retrieved_at=retrieved,
        content_hash=hashlib.sha256(slug.encode()).hexdigest(),
        freshness=SourceFreshness.FRESH,
    )


def make_program(
    slug: str = "example-grocery-help",
    *,
    category: ProgramCategory = ProgramCategory.FOOD,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
    keywords: Sequence[str] = ("groceries", "food", "nutrition assistance"),
    counties: Sequence[str] = ("Middlesex",),
    active_rule_version: str | None = "2026.1",
) -> AssistanceProgram:
    """A visibly fictional program, safe to store and assert against.

    Named and linked so that it could never be mistaken for a real program: the
    URLs are ``example.org`` and the title says so. The curated catalog lives in
    ``backend/data/programs`` and is the only place real programs belong.
    """
    return AssistanceProgram(
        id=program_uuid(slug),
        slug=slug,
        official_name=f"Example Grocery Help ({slug}) - fictional test program",
        administering_org="Example County Office of Testing",
        category=category,
        description=(
            "A fictional program invented for the automated test suite. It "
            "helps nobody because it does not exist."
        ),
        plain_language_summary="A made-up program used only to test the database layer.",
        who_it_helps=["Fictional households", "Test fixtures"],
        geo_scope=GeoScope(
            nationwide=False,
            states=["NJ"],
            counties=list(counties),
            label="Fictional scope used in tests",
        ),
        official_url=f"https://example.org/{slug}",
        application_url=f"https://example.org/{slug}/apply",
        phone="1-555-0100",
        keywords=list(keywords),
        verification_status=verification_status,
        last_verified_at=(utcnow() if verification_status is VerificationStatus.VERIFIED else None),
        active_rule_version=active_rule_version,
        sources=[_fictional_source(slug)],
    )


def make_ruleset(
    program_slug: str = "example-grocery-help",
    *,
    version: str = "2026.1",
    effective_from: date | None = None,
    effective_to: date | None = None,
    review_status: ReviewStatus = ReviewStatus.APPROVED,
) -> RuleSet:
    """An approved, fictional rule set with two evaluable conditions."""
    return RuleSet(
        version=version,
        program_slug=program_slug,
        effective_from=effective_from or date(2026, 1, 1),
        effective_to=effective_to,
        root=AllOf(
            description="All of these fictional requirements must be met.",
            source_ids=[f"{program_slug}-source"],
            conditions=[
                GeographyCondition(
                    description="You live in New Jersey (fictional requirement).",
                    source_ids=[f"{program_slug}-source"],
                    states=["NJ"],
                ),
                HouseholdSizeCondition(
                    description="Your household has at least one person.",
                    source_ids=[f"{program_slug}-source"],
                    min=1,
                ),
            ],
        ),
        source_ids=[f"{program_slug}-source"],
        review_status=review_status,
        limitations=["This rule set is fictional and encodes nothing real."],
        reviewed_by="test-fixture",
        reviewed_at=utcnow() if review_status is ReviewStatus.APPROVED else None,
    )


def make_profile(**overrides: Any) -> UserProfile:
    """A fictional but realistic profile.

    Deliberately partial: ``age`` and some flags are left unset, because a real
    profile is incomplete far more often than not and the rule engine has to
    stay honest about that.
    """
    fields: dict[str, Any] = {
        "state": "NJ",
        "county": "Middlesex",
        "student_status": StudentStatus.UNDERGRAD_FULL_TIME,
        "household_size": 3,
        "income_min_cents": 180_000,
        "income_max_cents": 220_000,
        "income_period": IncomePeriod.MONTHLY,
        "employment_status": EmploymentStatus.EMPLOYED_PART_TIME,
        "housing_situation": HousingSituation.RENTING,
        "needs": [NeedCategory.FOOD, NeedCategory.UTILITIES],
        "unknown_fields": ["age"],
    }
    fields.update(overrides)
    return UserProfile(**fields)


def make_evaluation(
    program: AssistanceProgram,
    *,
    status: MatchStatus = MatchStatus.POTENTIAL_MATCH,
    rule_version: str = "2026.1",
    score: float = 0.75,
) -> EligibilityEvaluation:
    """An evaluation whose ``status`` and ``outcome`` agree by construction."""
    outcome = {
        MatchStatus.POTENTIAL_MATCH: Trilean.TRUE,
        MatchStatus.MORE_INFO_NEEDED: Trilean.UNKNOWN,
        MatchStatus.UNLIKELY_MATCH: Trilean.FALSE,
    }[status]
    missing = ["household_size"] if status is MatchStatus.MORE_INFO_NEEDED else []
    return EligibilityEvaluation(
        program_id=program.id,
        program_slug=program.slug,
        rule_version=rule_version,
        status=status,
        outcome=outcome,
        score=score,
        evidence=[],
        missing_fields=missing,
        limitations=["Fictional evaluation produced by a test fixture."],
        evaluated_at=utcnow(),
    )


def make_match(
    program: AssistanceProgram,
    *,
    status: MatchStatus = MatchStatus.POTENTIAL_MATCH,
    with_explanation: bool = False,
    score: float = 0.75,
) -> ProgramMatch:
    evaluation = make_evaluation(program, status=status, score=score)
    explanation = (
        MatchExplanation(
            headline="You may be able to get help from this fictional program.",
            why_it_may_fit=["You live in the fictional service area."],
            what_we_still_need=[],
            caveats=["Fictional evaluation produced by a test fixture."],
            generated_by="deterministic_fallback",
        )
        if with_explanation
        else None
    )
    return ProgramMatch(
        program=program.to_summary(),
        status=status,
        score=score,
        evaluation=evaluation,
        explanation=explanation,
        sources=[],
    )


def make_plan(
    run_id: uuid.UUID,
    program: AssistanceProgram,
    *,
    version: int = 1,
    subjects: Sequence[str] = ("proof of address", "proof of income"),
) -> ActionPlan:
    """A plan whose item ids and dedup keys are derived the production way."""
    items: list[ActionItem] = []
    for index, subject in enumerate(subjects):
        dedup_key = make_dedup_key(ActionItemKind.DOCUMENT, subject)
        items.append(
            ActionItem(
                id=action_item_uuid(run_id, dedup_key),
                program_id=program.id,
                program_name=program.official_name,
                kind=ActionItemKind.DOCUMENT,
                title=f"Find your {subject}",
                detail=f"The fictional program asks for {subject}.",
                url=program.application_url,
                sort_order=index,
                is_complete=False,
                completed_at=None,
                dedup_key=dedup_key,
            )
        )
    return ActionPlan(
        id=uuid.uuid5(uuid.NAMESPACE_URL, f"plan:{run_id}:{version}"),
        run_id=run_id,
        version=version,
        title="Your fictional next steps",
        summary="Three sentences of fictional guidance for the test suite.",
        items=items,
        shared_documents=[],
        created_at=utcnow(),
    )


@pytest.fixture
def program_factory() -> Iterator[Any]:
    yield make_program


@pytest.fixture
def profile_factory() -> Iterator[Any]:
    yield make_profile


@pytest.fixture
def utc_now() -> datetime:
    return datetime.now(UTC)
