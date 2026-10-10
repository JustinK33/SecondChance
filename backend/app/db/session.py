"""Async engine, session factory and the two ways to get a session.

There are exactly two:

``get_session()``
    The FastAPI dependency. One session per request, opened on the way in and
    closed on the way out, with an implicit rollback if the handler raised.

``session_scope()``
    An async context manager for work that has no request - background
    workflow execution, the seed script, a scheduled job. It commits on clean
    exit and rolls back on an exception.

**A session is never shared across concurrent tasks.** This is the rule that
matters most in this module. :class:`~sqlalchemy.ext.asyncio.AsyncSession` is
not safe for concurrent use: it owns one DBAPI connection, and two coroutines
awaiting on it interleave their protocol traffic, which surfaces as
``InterfaceError: cannot perform operation: another operation is in progress``
or, worse, as one task's results arriving in the other. The workflow graph fans
out, so this is a live hazard rather than a theoretical one. Every parallel
branch that needs the database must open its own ``session_scope()``; objects
are passed between branches as validated Pydantic models, never as ORM
instances bound to someone else's session.

The engine, by contrast, *is* safe to share and is intended to be: it owns the
connection pool, and there must be exactly one per process.

Pooler support
--------------
Supabase publishes a session-mode pooler (port 5432) and a transaction-mode
pooler (port 6543). Transaction mode hands a different backend to each
transaction, so server-side prepared statements - which asyncpg creates for
every query it caches - break with ``prepared statement "__asyncpg_stmt_1__"
does not exist``. :func:`create_engine` detects a pooler URL and disables the
statement cache, and ``statement_cache_size`` is also a parameter so the
behaviour can be forced either way without editing code.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Final
from urllib.parse import urlsplit

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

__all__ = [
    "create_engine",
    "create_session_factory",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_session_factory",
    "resolve_connect_args",
    "session_scope",
]

_log = get_logger(__name__)

#: Hostname fragments and ports that identify a connection pooler in front of
#: Postgres rather than Postgres itself.
_POOLER_HOST_FRAGMENTS: Final[tuple[str, ...]] = ("pooler.supabase.com", "pgbouncer")
#: Supabase's transaction-mode pooler port. Session mode is 5432.
_TRANSACTION_POOLER_PORT: Final[int] = 6543

#: asyncpg's own default is 100 prepared statements per connection.
_DEFAULT_STATEMENT_CACHE_SIZE: Final[int] = 100

#: Recycle a pooled connection after this long. Comfortably under the idle
#: timeouts that managed Postgres and most load balancers apply, so the
#: application closes connections on its own terms instead of discovering a
#: dead one mid-query.
_POOL_RECYCLE_S: Final[int] = 1800

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _is_pooled_url(url: str) -> bool:
    """Whether ``url`` points at a connection pooler rather than at Postgres."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if any(fragment in host for fragment in _POOLER_HOST_FRAGMENTS):
        return True
    return parts.port == _TRANSACTION_POOLER_PORT


def _unique_statement_name() -> str:
    """Per-statement name generator for use behind a transaction-mode pooler.

    With the cache disabled asyncpg still prepares each statement once, and
    reusing a name across pooled backends is exactly the failure mode we are
    avoiding. A fresh name per statement cannot collide.
    """
    return f"__asyncpg_{uuid.uuid4().hex}__"


def resolve_connect_args(
    settings: Settings,
    url: str,
    *,
    statement_cache_size: int | None = None,
) -> dict[str, Any]:
    """The asyncpg connect arguments for ``url``.

    Split out of :func:`create_engine` so the pooler decision is a pure
    function that can be asserted on directly. An engine's connect arguments
    disappear into a pool creator closure once it is built, so testing this
    through the engine would mean reaching into SQLAlchemy internals - and the
    thing worth testing is the decision, not where SQLAlchemy stored it.

    ``statement_cache_size`` of ``None`` means "decide from the URL": ``0``
    behind a pooler, asyncpg's usual cache otherwise.
    """
    pooled = _is_pooled_url(url)
    cache_size = (
        statement_cache_size
        if statement_cache_size is not None
        else (0 if pooled else _DEFAULT_STATEMENT_CACHE_SIZE)
    )
    connect_args: dict[str, Any] = {
        # asyncpg takes the statement timeout as a server setting, which is the
        # only way to bound a query that has already left the process. A
        # runaway query otherwise holds a pooled connection for as long as the
        # database is willing to run it.
        "server_settings": {
            "statement_timeout": str(int(settings.db_statement_timeout_s * 1000)),
            "application_name": f"secondchance-{settings.app_env}",
        },
        "statement_cache_size": cache_size,
        "timeout": settings.db_pool_timeout_s,
    }
    if cache_size == 0:
        connect_args["prepared_statement_name_func"] = _unique_statement_name
    return connect_args


def create_engine(
    settings: Settings,
    *,
    url: str | None = None,
    statement_cache_size: int | None = None,
) -> AsyncEngine:
    """Build the async engine for ``settings``.

    :param url: override the URL, which is how the test suite points at
        ``TEST_DATABASE_URL`` without mutating global configuration.
    :param statement_cache_size: force asyncpg's prepared-statement cache size.
        ``None`` means "decide from the URL": ``0`` behind a pooler, asyncpg's
        default otherwise. Pass ``0`` explicitly to support a transaction-mode
        pooler that this function's host detection does not recognise.

    Pool sizing comes from :class:`~app.core.config.Settings` rather than from
    SQLAlchemy's defaults, because the ceiling that matters is the database's
    ``max_connections`` divided by the number of application processes, and
    that is a deployment decision rather than a library default.
    """
    resolved_url = url or settings.database_url_async
    connect_args = resolve_connect_args(
        settings, resolved_url, statement_cache_size=statement_cache_size
    )

    engine = create_async_engine(
        resolved_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_s,
        pool_recycle=_POOL_RECYCLE_S,
        # Costs one round trip per checkout and removes the entire class of
        # "first query after an idle period fails" bugs.
        pool_pre_ping=True,
        connect_args=connect_args,
        echo=False,
        future=True,
    )
    _log.debug(
        "db.engine.created",
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pooled_endpoint=_is_pooled_url(resolved_url),
        statement_cache_size=connect_args["statement_cache_size"],
    )
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory bound to ``engine``.

    ``expire_on_commit=False`` so that attributes read after a commit do not
    trigger a lazy refresh. In an async session that refresh would be an
    implicit IO on attribute access, which raises
    ``MissingGreenlet`` instead of doing anything useful.

    ``autoflush=False`` because this codebase flushes explicitly where order
    matters. An implicit flush in the middle of building a plan is how a
    half-built object reaches a constraint before its parent exists.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autobegin=True,
    )


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """The process-wide engine, created on first use.

    One engine per process: it owns the connection pool, so a second one would
    silently double the configured pool size.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(settings or get_settings())
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """The process-wide session factory, created on first use."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory(get_engine(settings))
    return _session_factory


async def dispose_engine() -> None:
    """Close the pool. Call from the application's lifespan shutdown.

    Without this, a reload or a test run leaves connections checked out until
    the database times them out, and a tight test loop exhausts
    ``max_connections``.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _log.debug("db.engine.disposed")
    _engine = None
    _session_factory = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding one session for the lifetime of a request.

    Does **not** commit. Handlers and services commit deliberately, because a
    dependency that commits on the way out will eventually persist half of a
    request that failed after the last successful statement.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession] | Callable[[], AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    """Transactional scope for work outside a request.

    Commits on clean exit, rolls back on an exception, and always closes. Use
    one of these per concurrent task - never one shared between them::

        async def node(state: WorkflowState) -> dict[str, object]:
            async with session_scope() as session:
                repo = ProfileRepository(session)
                ...

    :param factory: override the session factory, which is how tests bind the
        scope to a transaction that is rolled back afterwards.
    """
    make_session = factory or get_session_factory()
    session = make_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
