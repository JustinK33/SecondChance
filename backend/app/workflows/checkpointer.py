"""Checkpointer selection, lifecycle and retention.

A LangGraph checkpoint is what makes this workflow resumable: the run pauses on
a clarification question, the HTTP request that started it returns, and a
different process minutes later picks the run up exactly where it stopped. The
price is that the checkpoint row holds the most sensitive data in the product -
the narrative, the assembled profile and the user's answers - so this module is
as much a privacy control as it is a configuration switch.

Thread scoping
--------------
``thread_id`` is ``str(run_id)`` and nothing else. A run belongs to one user, a
``run_id`` is a server-generated UUID that never appears in a request body as
an authorisation, and ``workflow_runs.thread_id`` stores the same value so the
deletion flow does not have to re-derive the convention. One run, one thread,
one user: there is no composite key to get wrong and no way for one user's
resume to land on another user's state.

Retention policy
----------------
* A checkpoint lives as long as its run. There is no background expiry, because
  a user who comes back in a week to finish a paused application should find it
  waiting.
* ``DELETE /api/v1/account/data`` removes them. ``AccountDataRepository``
  deletes ``checkpoint_writes``, ``checkpoint_blobs`` and ``checkpoints`` by
  ``thread_id`` in the *same transaction* as the user's rows, which is the
  right place for it: atomic with everything else being deleted.
  :func:`delete_threads` here is the saver-level equivalent for a caller that
  holds a checkpointer and no session - a maintenance script, or a test.
* ``checkpoint_migrations`` is the checkpointer's own schema-version table and
  holds no user data, so it is never touched.
* The tables are created by ``await saver.setup()``, not by Alembic, and
  ``alembic/env.py`` excludes them. Do not add a migration for them.

psycopg specifics
-----------------
``AsyncPostgresSaver`` speaks psycopg3 and needs a plain libpq DSN
(``postgresql://``), not SQLAlchemy's ``postgresql+asyncpg://``;
:attr:`Settings.database_dsn` derives it. The connection must be
``autocommit=True`` - the saver manages its own transactions and a pooled
connection left in an implicit one will deadlock against the graph's own reads -
and ``row_factory=dict_row``, because the saver reads columns by name.
``prepare_threshold=0`` disables server-side prepared statements, which a
connection pool in front of a pooler cannot rely on.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Final
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

__all__ = [
    "CHECKPOINT_CONNECT_KWARGS",
    "CheckpointerHandle",
    "checkpointer_scope",
    "close_checkpointer",
    "delete_threads",
    "get_checkpointer",
    "thread_config",
    "thread_id_for",
]

_log = get_logger(__name__)

#: Connection settings ``AsyncPostgresSaver`` requires. See the module
#: docstring for why each one is not optional.
CHECKPOINT_CONNECT_KWARGS: Final[dict[str, Any]] = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}

_POOL_MIN: Final[int] = 1
_POOL_OPEN_TIMEOUT_S: Final[float] = 10.0


def thread_id_for(run_id: UUID | str) -> str:
    """The checkpointer thread for a run. One run, one thread, one user."""
    return str(run_id)


def thread_config(
    run_id: UUID | str,
    *,
    checkpoint_ns: str = "",
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """The ``config`` every ``ainvoke`` for this run must be given.

    Passing anything other than this run's own thread id is how checkpoint
    state would leak across runs, so callers should build the config here
    rather than assembling the dictionary themselves.
    """
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id_for(run_id), "checkpoint_ns": checkpoint_ns}
    }
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


class CheckpointerHandle:
    """A checkpointer plus whatever has to be closed with it.

    ``AsyncPostgresSaver.from_conn_string`` is an async context manager, so the
    saver it yields is only valid inside a ``with`` block - which does not fit a
    long-lived ASGI application. This holds an explicit
    ``AsyncConnectionPool`` instead, so the saver's lifetime is the
    application's lifespan and closing is something the caller does on purpose.
    """

    __slots__ = ("_closed", "_pool", "kind", "saver")

    def __init__(
        self,
        saver: Any,
        *,
        kind: str,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]] | None = None,
    ) -> None:
        self.saver = saver
        self.kind = kind
        self._pool = pool
        self._closed = False

    @property
    def is_persistent(self) -> bool:
        """Whether a restart keeps paused runs resumable."""
        return self._pool is not None

    async def aclose(self) -> None:
        """Close the pool, once. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            await self._pool.close()
            _log.info("workflow.checkpointer.closed", kind=self.kind)

    async def delete_threads(self, thread_ids: Sequence[UUID | str]) -> int:
        """Delete every checkpoint for these threads. Returns the count asked for.

        Used by the account-deletion flow's saver-level equivalent and by
        maintenance scripts. Deleting a thread that does not exist is a no-op,
        so this is safe to call for a user who never completed a run.
        """
        deleted = 0
        for thread_id in thread_ids:
            await self.saver.adelete_thread(thread_id_for(thread_id))
            deleted += 1
        if deleted:
            # Thread ids are run ids: server-generated and non-personal.
            _log.info("workflow.checkpointer.threads_deleted", count=deleted)
        return deleted


async def _build(settings: Settings) -> CheckpointerHandle:
    if settings.checkpointer == "memory":
        # Legal only on a developer machine: Settings refuses a non-postgres
        # checkpointer outside APP_ENV=local, because a restart would strand
        # every paused run.
        _log.info("workflow.checkpointer.ready", kind="memory", persistent=False)
        return CheckpointerHandle(InMemorySaver(), kind="memory")

    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=settings.database_dsn,
        min_size=_POOL_MIN,
        max_size=settings.db_pool_size,
        open=False,
        kwargs=CHECKPOINT_CONNECT_KWARGS,
    )
    await pool.open(wait=True, timeout=_POOL_OPEN_TIMEOUT_S)
    saver = AsyncPostgresSaver(pool)
    # Idempotent: creates the four checkpoint tables on a fresh database and
    # applies the checkpointer's own migrations on an existing one. Called once
    # per process, from the application's lifespan.
    await saver.setup()
    _log.info(
        "workflow.checkpointer.ready",
        kind="postgres",
        persistent=True,
        max_size=settings.db_pool_size,
    )
    return CheckpointerHandle(saver, kind="postgres", pool=pool)


_handle: CheckpointerHandle | None = None
_lock = asyncio.Lock()


async def get_checkpointer(settings: Settings | None = None) -> CheckpointerHandle:
    """The process-wide checkpointer, built and migrated once.

    Call this from the ASGI lifespan so ``setup()`` runs before the first
    request, and :func:`close_checkpointer` on shutdown. The lock means two
    concurrent first requests cannot both open a pool and run ``setup()``.
    """
    global _handle
    if _handle is not None:
        return _handle
    async with _lock:
        if _handle is None:
            _handle = await _build(settings or get_settings())
    return _handle


async def close_checkpointer() -> None:
    """Close and forget the process-wide checkpointer."""
    global _handle
    if _handle is None:
        return
    await _handle.aclose()
    _handle = None


@asynccontextmanager
async def checkpointer_scope(settings: Settings | None = None) -> AsyncIterator[CheckpointerHandle]:
    """A checkpointer owned by this block. For scripts and tests.

    Independent of the process-wide one, so a test can run against Postgres
    checkpoints without disturbing a running application's pool.
    """
    handle = await _build(settings or get_settings())
    try:
        yield handle
    finally:
        await handle.aclose()


async def delete_threads(
    thread_ids: Sequence[UUID | str], *, settings: Settings | None = None
) -> int:
    """Delete checkpoints for these threads using the process-wide saver.

    Prefer ``AccountDataRepository.delete_user_data``, which removes the same
    rows inside the transaction that deletes the rest of the user's data and is
    therefore atomic with it. This exists for callers that hold no session.
    """
    if not thread_ids:
        return 0
    handle = await get_checkpointer(settings)
    return await handle.delete_threads(thread_ids)
