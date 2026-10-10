"""Bounded concurrency, retries and deadlines.

Three primitives, used by retrieval fan-out, the workflow runner and anything
else that talks to a dependency it does not control.

* :func:`bounded_gather` runs awaitables concurrently with a ceiling. The
  ceiling is the point: an unbounded ``asyncio.gather`` over a user's twenty
  retrieved programs would open twenty sessions and fire twenty model calls,
  and the first user to do that would exhaust the connection pool for everyone
  else.
* :func:`retry_async` retries **transient** failures only, with exponential
  backoff and *full* jitter. Full jitter rather than half or none, because a
  provider blip makes every in-flight request fail at the same instant and a
  deterministic backoff then reproduces the thundering herd one second later.
* :func:`deadline` bounds a block of work in wall-clock time and converts the
  resulting ``TimeoutError`` into the application's own error type, so a
  timeout reaches the user as plain-language copy rather than as a bare
  framework exception.

The transient predicate is deliberately narrow. :attr:`AppError.is_retryable`
is already the vetted answer to "could the identical request succeed next
time?", so this module reads that flag instead of maintaining a second,
drifting list of retryable conditions. A ``ValueError`` from a validator is not
transient and retrying it only wastes the user's time.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable
from contextlib import asynccontextmanager
from typing import Any, Final, cast

from app.core.errors import AppError, UpstreamTimeoutError
from app.core.logging import get_logger

__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BASE_DELAY_S",
    "DEFAULT_MAX_DELAY_S",
    "RetryObserver",
    "backoff_delay_s",
    "bounded_gather",
    "deadline",
    "is_transient",
    "retry_async",
]

_log = get_logger(__name__)

#: One initial attempt plus two retries. Past that the user is better served by
#: an honest "try again in a moment" than by a request that hangs.
DEFAULT_ATTEMPTS: Final[int] = 3
DEFAULT_BASE_DELAY_S: Final[float] = 0.25
DEFAULT_MAX_DELAY_S: Final[float] = 8.0

#: Jitter source. ``SystemRandom`` so that several processes starting at the
#: same moment do not share a seeded sequence and re-synchronise their retries.
_JITTER: Final[random.SystemRandom] = random.SystemRandom()

#: Called as ``observer(attempt, exc, delay_s)`` before each sleep. Used to
#: increment the retry counter in :mod:`app.core.metrics`; takes codes and
#: numbers only, never a payload.
RetryObserver = Callable[[int, BaseException, float], None]


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying.

    ``True`` for an :class:`~app.core.errors.AppError` that declares itself
    retryable, and for the three transport-level failures that mean "the
    connection did not survive" rather than "the request was wrong".
    Everything else is ``False``, including ``asyncio.CancelledError``, which
    must never be retried - a cancelled operation has been asked to stop.
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, AppError):
        return exc.is_retryable
    return isinstance(exc, TimeoutError | ConnectionError | OSError)


def backoff_delay_s(
    attempt: int,
    *,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
) -> float:
    """Full-jitter exponential backoff for a 1-based ``attempt`` number.

    The window doubles per attempt and the delay is drawn uniformly from the
    whole window, so two callers that failed together almost never retry
    together.
    """
    if attempt < 1:
        msg = f"attempt must be >= 1, got {attempt}"
        raise ValueError(msg)
    window = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
    return _JITTER.uniform(0.0, window)


async def bounded_gather[T](
    limit: int,
    *awaitables: Awaitable[T],
    return_exceptions: bool = False,
) -> list[T]:
    """Await every item in ``awaitables`` with at most ``limit`` in flight.

    Results are in argument order, like :func:`asyncio.gather`. An awaitable
    does not begin until it has acquired a slot, so the ceiling bounds the work
    actually started rather than only the work completed.

    With ``return_exceptions=False`` a failure propagates immediately and the
    remaining awaitables are cancelled, which is what a caller that cannot use
    a partial result wants. Retrieval fan-out passes ``True`` instead, because
    nineteen programs out of twenty is a useful answer and the twentieth
    becomes a limitation the user reads.
    """
    if limit < 1:
        msg = f"limit must be >= 1, got {limit}"
        raise ValueError(msg)
    if not awaitables:
        return []

    semaphore = asyncio.Semaphore(limit)

    async def guarded(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    tasks = [asyncio.ensure_future(guarded(item)) for item in awaitables]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    except BaseException:
        for task in tasks:
            task.cancel()
        # Let the cancellations settle so no task is garbage-collected while
        # still pending, which would surface as "Task was destroyed but it is
        # pending" noise that hides the real failure.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return cast(list[T], results)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
    retry_on: Callable[[BaseException], bool] = is_transient,
    observer: RetryObserver | None = None,
    operation_name: str = "operation",
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``operation`` until it succeeds, up to ``attempts`` times.

    :param operation: a zero-argument factory returning a fresh awaitable.
        A factory rather than an awaitable because a coroutine object can only
        be awaited once - retrying the same object raises ``RuntimeError``.
    :param retry_on: the transient-only predicate. Defaults to
        :func:`is_transient`; pass a narrower one for an operation that is not
        idempotent.
    :param observer: notified before each sleep, for metrics.
    :param sleep: injectable so tests do not spend real seconds.

    The last failure is re-raised unchanged, so the caller still sees the real
    error type and the API layer still maps it to the right status code.
    """
    if attempts < 1:
        msg = f"attempts must be >= 1, got {attempts}"
        raise ValueError(msg)

    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            last = exc
            if attempt == attempts or not retry_on(exc):
                raise
            delay = backoff_delay_s(attempt, base_delay_s=base_delay_s, max_delay_s=max_delay_s)
            if observer is not None:
                observer(attempt, exc, delay)
            # Type name and code only: an exception message may quote the
            # payload that caused it.
            _log.info(
                "concurrency.retry",
                operation=operation_name,
                attempt=attempt,
                attempts=attempts,
                delay_ms=round(delay * 1000),
                error_type=type(exc).__name__,
                error_code=exc.code.value if isinstance(exc, AppError) else None,
            )
            await sleep(delay)
    raise cast(BaseException, last)  # pragma: no cover - loop always returns or raises


@asynccontextmanager
async def deadline(
    seconds: float | None,
    *,
    as_error: type[AppError] | None = UpstreamTimeoutError,
    operation_name: str = "operation",
    observer: Callable[[str], None] | None = None,
) -> AsyncIterator[None]:
    """Bound the enclosed block in wall-clock time.

    ``seconds=None`` means "no deadline" and is a legitimate choice, so a
    caller can pass a configured value straight through without branching.

    On expiry the block is cancelled and ``as_error`` is raised with the
    ``TimeoutError`` as its cause. Pass ``as_error=None`` to re-raise the bare
    ``TimeoutError`` instead, which is what the workflow runner wants: it
    records the timeout itself and must not confuse it with an upstream one.
    """
    if seconds is None:
        yield
        return
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError as exc:
        if observer is not None:
            observer(operation_name)
        _log.warning(
            "concurrency.deadline_exceeded",
            operation=operation_name,
            budget_ms=round(seconds * 1000),
        )
        if as_error is None:
            raise
        raise as_error(cause=exc) from exc


async def cancel_all(
    tasks: Iterable[asyncio.Task[Any] | Coroutine[Any, Any, Any]],
) -> None:
    """Cancel every task and wait for it to finish unwinding.

    Used on shutdown. Awaiting the cancellations rather than only requesting
    them is what gives each task's ``finally`` a chance to run, which is where
    a run's status is recorded.
    """
    pending = [task for task in tasks if isinstance(task, asyncio.Task) and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
