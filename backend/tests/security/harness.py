"""Shared machinery for driving the real ASGI app from a test.

Every test in ``tests/security`` and ``tests/integration/test_api_*.py`` talks
to the application the same way a browser does: over HTTP, through
:class:`httpx.AsyncClient` with an ``ASGITransport``, against the app that
:func:`app.main.create_app` builds. Nothing here calls a route function
directly, because half of what these tests are about - the middleware, the
exception handlers, the dependency graph, the error envelope - only exists on
the HTTP path.

What is substituted, and why only this much
-------------------------------------------
Four dependency overrides, no monkeypatching of application internals:

``get_app_settings``
    A hermetic :class:`~app.core.config.Settings` instead of the repository's
    real ``.env``, which carries a live API key. A test that silently depended
    on it would pass or fail for the wrong reason.
``get_session``
    Either the rolled-back session from the root ``session`` fixture (the
    default, so nothing survives a test) or a fresh session per request from a
    real ``async_sessionmaker`` (needed by the concurrency tests, which cannot
    share one connection).
``provide_gemini``
    :class:`~app.ai.fake.FakeGeminiClient`, which replays recorded responses
    through the same decoder and the same projections as production.
``provide_runner``
    :class:`RecordingRunner` for route tests, so ``POST /workflows`` is tested
    as an HTTP contract rather than as a workflow. The end-to-end journey
    installs the *real* runner instead; see ``scripts/e2e_journey.py``.

Everything else - the rule engine, the retriever, the citation verifier, the
repositories, the services - is the production object.

Why this module lives here
--------------------------
It is support code rather than a test, so it would read better at
``tests/support/api.py``. This task owns ``tests/security/*.py`` and
``tests/integration/test_api_*.py`` only, and inventing a shared directory
would mean editing a path somebody else owns. Recorded as a handoff note.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest_asyncio
import structlog
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ai.fake import FakeGeminiClient
from app.api.v1.deps import get_app_settings, provide_gemini, provide_runner
from app.core.config import Settings
from app.core.logging import configure_logging
from app.core.security import DEV_USER_HEADER
from app.db import models
from app.db.repositories import AccountDataRepository, SqlProgramRepository
from app.db.session import get_session
from app.main import create_app
from app.schemas import AssistanceProgram, ClarificationAnswer, RuleSet, clarification_question_id

__all__ = [
    "DEV_USER_HEADER",
    "ApiHarness",
    "RecordingRunner",
    "answer_for",
    "api_harness",
    "capture_app_logs",
    "committing_api",
    "dev_settings",
    "rolled_back_api",
    "seed_catalog",
    "supabase_settings",
]

#: Base URL for the in-process transport. Never resolved; httpx only needs it
#: to build an absolute URL for the ASGI scope.
BASE_URL = "http://api.test"


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def dev_settings(**overrides: Any) -> Settings:
    """Local settings with ``AUTH_MODE=dev``, independent of any ``.env``.

    ``gemini_api_key`` is deliberately empty: these tests inject the fixture
    client explicitly, and an empty key means a stray code path that builds its
    own port cannot reach the network either.
    """
    base: dict[str, Any] = {
        "app_env": "local",
        "auth_mode": "dev",
        "log_level": "INFO",
        "checkpointer": "memory",
        "gemini_api_key": "",
        "gemini_model": "gemini-test-base",
        "gemini_model_extraction": "gemini-test-cheap",
        "gemini_model_explanation": "gemini-test-strong",
        "gemini_model_chat": "gemini-test-chat",
        "gemini_timeout_s": 5.0,
        "gemini_max_retries": 1,
        "workflow_timeout_s": 60.0,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def supabase_settings(**overrides: Any) -> Settings:
    """Settings with real token verification switched on.

    ``supabase_url`` is a fictional project. Nothing in these tests reaches it:
    the JWKS document is served by an in-process ``httpx.MockTransport``.
    """
    base: dict[str, Any] = {
        "app_env": "local",
        "auth_mode": "supabase",
        "checkpointer": "memory",
        "gemini_api_key": "",
        "gemini_timeout_s": 5.0,
        "gemini_max_retries": 1,
        "workflow_timeout_s": 60.0,
        "supabase_url": "https://fictional-project.supabase.co",
        # Not a credential: HS256 is a separate code path and these tests sign
        # with ES256. Settings refuses to construct without a non-empty value.
        "supabase_jwt_secret": "not-a-credential-only-long-enough-to-avoid-a-key-length-warning",
    }
    base.update(overrides)
    return dev_settings(**base)


# ---------------------------------------------------------------------------
# the runner stand-in
# ---------------------------------------------------------------------------
@dataclass
class RecordingRunner:
    """Records what the API asked the workflow runner to do, and does nothing.

    Duck-typed against the surface ``app.services.workflow_service`` actually
    uses. A route test asserting on a 202 and a ``Location`` header has no
    business executing a LangGraph run, and a recording stub makes the
    scheduling itself assertable - including that a deduplicated create
    schedules *nothing*.
    """

    scheduled: list[dict[str, Any]] = field(default_factory=list)
    resumed: list[dict[str, Any]] = field(default_factory=list)
    continued: list[dict[str, Any]] = field(default_factory=list)
    cancelled: list[uuid.UUID] = field(default_factory=list)

    def schedule_run(self, **kwargs: Any) -> None:
        self.scheduled.append(kwargs)

    def schedule_resume(self, **kwargs: Any) -> None:
        self.resumed.append(kwargs)

    def schedule_continue(self, **kwargs: Any) -> None:
        self.continued.append(kwargs)

    def cancel(self, run_id: uuid.UUID) -> bool:
        self.cancelled.append(run_id)
        return True

    def is_running(self, run_id: uuid.UUID) -> bool:
        return False


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ApiHarness:
    """An HTTP client bound to a configured application."""

    client: AsyncClient
    app: Any
    settings: Settings
    gemini: FakeGeminiClient
    runner: RecordingRunner
    session_factory: async_sessionmaker[AsyncSession] | None = None

    def dev(self, user_id: str) -> dict[str, str]:
        """Headers authenticating as ``user_id`` in ``AUTH_MODE=dev``."""
        return {DEV_USER_HEADER: user_id}

    def bearer(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def cleanup_users(self, user_ids: Sequence[str]) -> None:
        """Hard-delete everything these users own, leaving no residue.

        Only needed by the committing harness; the rolled-back one discards
        its transaction instead.

        ``delete_user_data`` deliberately keeps one audit row recording that
        the deletion happened, which is right for a real user and wrong for a
        test: several suites count rows globally, so one row per test would
        accumulate until an unrelated assertion failed. The receipt row is
        removed afterwards so the database is left exactly as it was found.
        """
        if self.session_factory is None:
            return
        async with self.session_factory() as session:
            for user_id in user_ids:
                await AccountDataRepository(session).delete_user_data(user_id)
            await session.execute(
                delete(models.AuditEvent).where(models.AuditEvent.user_id.in_(list(user_ids)))
            )
            await session.commit()


@asynccontextmanager
async def api_harness(
    *,
    settings: Settings,
    session: AsyncSession | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    gemini: FakeGeminiClient | None = None,
    runner: RecordingRunner | None = None,
) -> AsyncIterator[ApiHarness]:
    """Build the app, wire the four overrides, and yield a client for it.

    Exactly one of ``session`` and ``session_factory`` must be given: a single
    rolled-back session, or a factory that opens one per request.
    """
    if (session is None) == (session_factory is None):
        msg = "pass exactly one of session= or session_factory="
        raise TypeError(msg)

    fake = gemini if gemini is not None else FakeGeminiClient(settings)
    stub_runner = runner if runner is not None else RecordingRunner()
    application = create_app(settings)

    session_dep: Callable[..., Any]
    if session is not None:
        session_dep = lambda: session  # noqa: E731 - a one-line dependency
    else:
        assert session_factory is not None

        async def session_dep() -> AsyncIterator[AsyncSession]:
            async with session_factory() as per_request:
                yield per_request

    application.dependency_overrides[get_app_settings] = lambda: settings
    application.dependency_overrides[get_session] = session_dep
    application.dependency_overrides[provide_gemini] = lambda: fake
    application.dependency_overrides[provide_runner] = lambda: stub_runner

    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:
        yield ApiHarness(
            client=client,
            app=application,
            settings=settings,
            gemini=fake,
            runner=stub_runner,
            session_factory=session_factory,
        )


# ---------------------------------------------------------------------------
# fixtures, imported by name into the test modules that want them
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def rolled_back_api(session: AsyncSession) -> AsyncIterator[ApiHarness]:
    """Dev-auth harness whose every write is rolled back after the test."""
    async with api_harness(settings=dev_settings(), session=session) as harness:
        yield harness


@pytest_asyncio.fixture
async def committing_api(engine: AsyncEngine) -> AsyncIterator[ApiHarness]:
    """Dev-auth harness that really commits, one session per request.

    Needed wherever two requests must run genuinely concurrently: an
    ``AsyncSession`` owns one DBAPI connection and is not safe for concurrent
    use, so the idempotency race cannot be exercised on a shared one. Rows are
    removed on the way out by the test that created them.
    """
    factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, autoflush=False, autobegin=True
    )
    async with api_harness(settings=dev_settings(), session_factory=factory) as harness:
        yield harness


# ---------------------------------------------------------------------------
# catalog seeding
# ---------------------------------------------------------------------------
async def seed_catalog(
    session: AsyncSession,
    programs: Sequence[AssistanceProgram],
    rulesets: Sequence[RuleSet] = (),
) -> None:
    """Upsert fictional programs and rule sets through the real repository."""
    repository = SqlProgramRepository(session)
    for program in programs:
        await repository.upsert_program(program)
    for ruleset in rulesets:
        await repository.upsert_rule_version(ruleset)
    await session.commit()


def answer_for(
    run_id: uuid.UUID, field_name: str, value: Any, *, skipped: bool = False
) -> ClarificationAnswer:
    """A clarification answer keyed the way the client keys it."""
    return ClarificationAnswer(
        question_id=clarification_question_id(run_id, field_name),
        field=field_name,
        value=None if skipped else value,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# log capture
# ---------------------------------------------------------------------------
def _uncache_structlog_proxies() -> None:
    """Force every module-level ``get_logger`` proxy to re-bind its sink.

    ``configure_logging`` sets ``cache_logger_on_first_use=True``, and
    structlog implements that by assigning ``finalized_bind`` onto the proxy
    *instance* the first time it is used. A module that has already logged
    therefore keeps writing to the sink that was configured then, which would
    make a privacy assertion against a fresh buffer vacuously true - the worst
    possible failure mode for this particular test. Deleting the instance
    attribute restores the lazy path, so the next call picks up the current
    configuration.
    """
    import sys

    for name, module in list(sys.modules.items()):
        if not (name == "app" or name.startswith("app.")) or module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, structlog._config.BoundLoggerLazyProxy):
                vars(value).pop("bind", None)


@contextmanager
def capture_app_logs(settings: Settings) -> Iterator[io.StringIO]:
    """Run the real logging pipeline into a buffer this test can read.

    The pipeline is the production one - the same processors, the same
    redaction - because a privacy test that asserts on a different pipeline
    asserts nothing.
    """
    stream = io.StringIO()
    _uncache_structlog_proxies()
    configure_logging(settings, stream=stream)
    try:
        yield stream
    finally:
        _uncache_structlog_proxies()
        structlog.reset_defaults()


# ---------------------------------------------------------------------------
# JWKS over an in-process transport
# ---------------------------------------------------------------------------
def jwks_shim(handler: Callable[[httpx.Request], httpx.Response]) -> SimpleNamespace:
    """A stand-in for the ``httpx`` module as ``app.core.security`` uses it.

    ``JwksCache._fetch`` builds its own ``httpx.AsyncClient``, so the only way
    to serve a JWKS in-process is to give that module a different ``httpx``.
    Replacing the module-level name rather than ``httpx.AsyncClient`` itself
    keeps the substitution surgical: this test's own ASGI client is a real
    ``httpx.AsyncClient`` and must stay one.
    """

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return httpx.AsyncClient(**kwargs)

    return SimpleNamespace(AsyncClient=client_factory, HTTPError=httpx.HTTPError)
