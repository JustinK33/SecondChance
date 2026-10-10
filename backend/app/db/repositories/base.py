"""Shared repository machinery.

The single most important rule in this package is stated once here and relied
on everywhere: **a user-scoped query always filters by ``user_id``, and that
``user_id`` always comes from the verified token.**

An id taken from a request body or a URL is an *input*, never an
authorisation. So the methods below take ``user_id`` as a separate, mandatory
argument and put it in the ``WHERE`` clause alongside the row id. The effect is
that asking for somebody else's run returns ``None`` - the caller then raises
``NotFoundError`` rather than ``ForbiddenError``, so the API never confirms that
an id it will not serve actually exists.

Row-level security in the database is the second line of defence, not the
first. The local ``AUTH_MODE=dev`` path connects as the table owner and
therefore bypasses RLS entirely, so a repository that forgot its ``user_id``
filter would leak in development and be caught only in production. These
filters are what make the behaviour the same in both.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import CursorResult, Executable, Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "MAX_PAGE_LIMIT",
    "Repository",
    "UserScopedRepository",
    "as_utc",
    "count_of",
    "dedupe",
    "execute_dml",
    "normalise_page",
    "to_float",
    "uuid_list",
]

#: Matches ``Page.limit``'s upper bound in ``app.schemas.common``.
MAX_PAGE_LIMIT = 200


class Repository:
    """Base for every repository: holds one session and nothing else.

    An instance is bound to the session it was constructed with, so it inherits
    that session's transaction and its concurrency rule - one session per task,
    never shared. Repositories are cheap; create one per session rather than
    caching one and handing it different sessions.

    Repositories never commit. The caller owns the transaction boundary,
    because only the caller knows whether this write is the whole unit of work
    or one step of several.
    """

    __slots__ = ("session",)

    def __init__(self, session: AsyncSession) -> None:
        self.session = session


class UserScopedRepository(Repository):
    """Base for repositories over tables carrying a ``user_id``."""

    __slots__ = ()


def normalise_page(limit: int, offset: int) -> tuple[int, int]:
    """Clamp pagination arguments into the range ``Page`` will accept.

    Clamped rather than rejected: pagination is not a security boundary, and a
    client that asks for 10 000 rows should get a page, not a 422.
    """
    return max(1, min(limit, MAX_PAGE_LIMIT)), max(0, offset)


async def count_of(session: AsyncSession, statement: Select[Any]) -> int:
    """Total rows a filtered statement would return, ignoring its window.

    Wraps the statement as a subquery instead of rebuilding the filters, so the
    count and the page can never disagree about what they are counting.
    """
    subquery = statement.order_by(None).limit(None).offset(None).subquery()
    total = await session.scalar(select(func.count()).select_from(subquery))
    return int(total or 0)


async def execute_dml(session: AsyncSession, statement: Executable) -> int:
    """Run an ``UPDATE`` or ``DELETE`` and return the number of rows affected.

    ``AsyncSession.execute`` is typed as returning a generic ``Result``, whose
    ``rowcount`` is only meaningful - and only typed - for DML. Narrowing it
    here once keeps the cast out of every call site and makes "how many rows
    did that touch" a normal thing to ask, which matters because the deletion
    receipt is built entirely out of those numbers.
    """
    result = cast("CursorResult[Any]", await session.execute(statement))
    return int(result.rowcount or 0)


def as_utc(value: datetime | date | None) -> datetime | None:
    """Coerce a stored timestamp or date into a timezone-aware UTC datetime.

    Every ``TIMESTAMPTZ`` already comes back aware, so this mostly matters for
    ``DATE`` columns (``RuleSet.effective_from``) that have to be presented as
    the ``UtcDatetime`` the schemas use. Midnight UTC is the convention; a
    naive value is treated as UTC rather than guessed at, which cannot happen
    anyway because the type map forbids naive columns.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def to_float(value: Decimal | float | None) -> float:
    """Present a ``NUMERIC`` score as the ``float`` the schemas declare."""
    if value is None:
        return 0.0
    return float(value)


def dedupe(values: Sequence[str] | None) -> list[str]:
    """First-seen-order deduplication. Keeps list columns deterministic."""
    return list(dict.fromkeys(values or ()))


def uuid_list(values: Sequence[uuid.UUID] | None) -> list[uuid.UUID]:
    return list(dict.fromkeys(values or ()))
