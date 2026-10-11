"""Request-scoped middleware: correlation ids, access logs, timeouts and CORS.

All of these are written as raw ASGI middleware rather than as Starlette's
``BaseHTTPMiddleware`` subclasses. ``BaseHTTPMiddleware`` wraps each request in
an anonymous task and buffers the response through a memory stream, which
breaks ``asyncio.timeout`` propagation and makes a cancelled request surface as
an opaque ``anyio`` error instead of as the timeout it was. A raw ASGI class is
about the same amount of code and behaves the way the rest of this module
documents.

Ordering, outermost first, as built by ``app.main.create_app``:

1. ``CORSMiddleware`` - must see the preflight before anything else does, and
   must be able to answer ``OPTIONS`` without a correlation id.
2. :class:`RequestContextMiddleware` - mints or adopts the request id, binds
   the structlog context, writes the access log and records the request metric.
   Outside the timeout so that a timed-out request is still logged and still
   carries ``X-Request-ID``.
3. :class:`RequestBodySizeLimitMiddleware` - refuses an oversized body before
   anything buffers it. Inside the context middleware so its 413 is logged and
   correlated like any other response, and outside the timeout because
   rejecting a body is not work that can time out.
4. :class:`RequestTimeoutMiddleware` - bounds the handler in wall-clock time.
5. The router.

Privacy rules this module enforces:

* the access log carries method, route **template**, status and duration. Never
  the raw path (it holds ids), never the query string, never a header value and
  never a body.
* the user is logged as ``hash_user_id`` output, written into the context by
  the auth dependency rather than parsed out of a header here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import MutableMapping
from typing import Any, Final

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import Settings
from app.core.errors import ErrorCode, UpstreamTimeoutError
from app.core.logging import bind_request_context, clear_request_context, get_logger, new_request_id
from app.core.metrics import MetricsRegistry, get_metrics

__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_S",
    "MAX_REQUEST_BODY_BYTES",
    "REQUEST_ID_HEADER",
    "UNTIMED_PATHS",
    "RequestBodySizeLimitMiddleware",
    "RequestContextMiddleware",
    "RequestTimeoutMiddleware",
    "cors_kwargs",
    "route_template",
]

_log = get_logger(__name__)

#: Inbound value is adopted when present so a correlation id set by a proxy,
#: the frontend or a load balancer survives into our logs.
REQUEST_ID_HEADER: Final[str] = "x-request-id"

#: A single HTTP request is never allowed to run longer than this. Workflow
#: execution is deliberately not an HTTP request - it is a background task with
#: its own, much larger budget - so nothing legitimate needs more.
DEFAULT_REQUEST_TIMEOUT_S: Final[float] = 30.0

#: Largest request body this API will accept, in bytes.
#:
#: Every body here is a small JSON document: the largest field in the whole
#: schema is a 4000-character narrative, and the largest realistic request -
#: a narrative plus a full set of clarification answers - is a few kilobytes.
#: 128 KiB is therefore around twenty times the biggest legitimate request and
#: still small enough that a burst of them cannot matter.
#:
#: Without a cap, ``json.loads`` on a body the client chose the size of is a
#: denial-of-service primitive available to anyone with ``curl``: a 20 MB body
#: is read into memory in full, and then again as a decoded string, *before*
#: Pydantic gets to reject the 4000-character limit. Measured against this
#: process before the cap existed, a single 20 MB request was accepted and
#: parsed in 117 ms.
MAX_REQUEST_BODY_BYTES: Final[int] = 128 * 1024

#: Methods whose body is read when no ``Content-Length`` is given. A body on
#: any other method is not something this API has a use for, so those requests
#: are passed through untouched rather than pre-read.
_BODY_METHODS: Final[frozenset[str]] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Paths exempt from the request timeout. The liveness probe must answer even
#: when the process is saturated, and the metrics scrape is read by a collector
#: with its own timeout.
UNTIMED_PATHS: Final[frozenset[str]] = frozenset(
    {"/healthz", "/api/v1/health", "/internal/metrics"}
)

#: Copy for the 413. Plain language, and it says what to do about it.
_BODY_TOO_LARGE_MESSAGE: Final[str] = (
    "That request was too large to accept. Please shorten what you sent and try again."
)

#: Body of the 504 the timeout middleware emits. Pre-rendered because building
#: it needs no request state, and a timeout is the worst moment to do work.
_TIMEOUT_BODY_TEMPLATE: Final[str] = (
    '{{"error":{{"code":"{code}","message":"{message}","detail":null,"request_id":{request_id}}}}}'
)


def route_template(scope: Scope) -> str:
    """The matched route's full path template, or a safe placeholder.

    The template is what makes a metric label and a log line safe:
    ``/api/v1/workflows/{run_id}`` is bounded cardinality and carries no id,
    whereas the raw path is unbounded and carries one. An unmatched request
    (404, or a timeout before routing) reports ``"unmatched"`` rather than its
    path, because an attacker's probe path is still an unbounded label.

    FastAPI nests included routers rather than flattening them, so
    ``route.path_format`` is relative to the router's mount point -
    ``/workflows/{run_id}``, not ``/api/v1/workflows/{run_id}`` - and two
    routers can expose the same relative template. The prefix is recovered by
    filling the template from ``path_params`` and matching it against the
    request path, which needs no knowledge of how the routers were assembled.
    """
    route = scope.get("route")
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    if not isinstance(template, str) or not template:
        return "unmatched"

    path = scope.get("path")
    params = scope.get("path_params") or {}
    if isinstance(path, str) and path:
        filled = template
        for name, value in params.items():
            filled = filled.replace("{" + name + "}", str(value))
        if filled and path.endswith(filled):
            return path[: len(path) - len(filled)] + template
    return template


def _json_bytes(code: ErrorCode, message: str, request_id: str | None) -> bytes:
    rendered = _TIMEOUT_BODY_TEMPLATE.format(
        code=code.value,
        message=message,
        request_id=f'"{request_id}"' if request_id else "null",
    )
    return rendered.encode("utf-8")


class RequestContextMiddleware:
    """Correlation id, structlog binding, access logging and request metrics.

    The request id is written to ``scope["state"]["request_id"]`` so a
    dependency can read it without importing a context variable, and echoed in
    the ``X-Request-ID`` response header so a user can quote it.
    """

    __slots__ = ("app", "metrics")

    def __init__(self, app: ASGIApp, *, metrics: MetricsRegistry | None = None) -> None:
        self.app = app
        self.metrics = metrics if metrics is not None else get_metrics()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        inbound = headers.get(REQUEST_ID_HEADER)
        request_id = _clean_request_id(inbound) or new_request_id()

        state: MutableMapping[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id

        method = str(scope.get("method", "GET"))
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers_out = MutableHeaders(scope=message)
                # Starlette's ServerErrorMiddleware sits outside this one, so a
                # 500 envelope is written by the exception handler and already
                # carries the header. Setting rather than appending keeps a
                # single value on every response.
                if REQUEST_ID_HEADER not in headers_out:
                    headers_out.append("X-Request-ID", request_id)
            await send(message)

        bind_request_context(request_id)
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # The exception handlers installed on the application turn this
            # into a 500 envelope; by the time it reaches here the response has
            # already been sent or the server is about to 500 it. Log the shape
            # and re-raise so nothing is swallowed.
            duration_s = time.perf_counter() - started
            self._record(method, scope, status_code, duration_s)
            raise
        else:
            duration_s = time.perf_counter() - started
            self._record(method, scope, status_code, duration_s)
        finally:
            clear_request_context()

    def _record(self, method: str, scope: Scope, status_code: int, duration_s: float) -> None:
        route = route_template(scope)
        self.metrics.observe_request(
            method=method, route=route, status_code=status_code, duration_s=duration_s
        )
        # Route template and status only. No path, no query string, no body.
        _log.info(
            "http.request",
            method=method,
            route=route,
            status=status_code,
            duration_ms=round(duration_s * 1000, 2),
        )


class RequestBodySizeLimitMiddleware:
    """Refuses a request body larger than :data:`MAX_REQUEST_BODY_BYTES`.

    Two paths, because the two cases have different costs.

    ``Content-Length`` present
        The declared length is checked and an oversized request is answered
        immediately, without reading a byte of the body. This is the case for
        every ordinary client.

    ``Content-Length`` absent (``Transfer-Encoding: chunked``)
        A chunked sender does not say how much it intends to send, so the only
        way to know is to count. The body is read here, up to the cap plus one
        byte, and then *replayed* to the application from the buffer. Buffering
        is bounded by the cap by construction, so this cannot be turned into
        the thing it is preventing, and the application sees an ordinary
        ``receive`` channel either way.

    The answer is ``413`` with the standard error envelope and code
    ``validation_error``: an oversized body is a request that failed
    validation, and the frontend already has a branch for that code. It is
    rendered here rather than raised, because raising would require the
    exception to survive the application's own handler stack - and the whole
    point of this layer is to answer before the application reads the body.
    """

    __slots__ = ("app", "max_bytes", "metrics")

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = MAX_REQUEST_BODY_BYTES,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.metrics = metrics if metrics is not None else get_metrics()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_content_length(scope)
        if declared is not None:
            if declared > self.max_bytes:
                await self._reject(scope, send, size=declared)
                return
            await self.app(scope, receive, send)
            return

        if str(scope.get("method", "")).upper() not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        buffered, measured = await self._read_bounded(receive)
        if measured > self.max_bytes:
            await self._reject(scope, send, size=measured)
            return
        await self.app(scope, _replaying(buffered, receive), send)

    async def _read_bounded(self, receive: Receive) -> tuple[list[Message], int]:
        """Read the body until it ends or exceeds the cap, whichever is first.

        Returns the messages read and the bytes counted. The count stops one
        message after the cap is passed, so it is "at least this big" rather
        than the true length - which is all a rejection needs, and is what
        keeps the buffer bounded.
        """
        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                # A disconnect ends the body as surely as ``more_body: False``.
                return buffered, total
            total += len(message.get("body") or b"")
            if total > self.max_bytes or not message.get("more_body", False):
                return buffered, total

    async def _reject(self, scope: Scope, send: Send, *, size: int) -> None:
        route = route_template(scope)
        self.metrics.increment_error(code=ErrorCode.VALIDATION_ERROR.value, route=route)
        # A byte count is a count, not content, so it is safe to log and it is
        # the one number that makes the line actionable.
        _log.warning(
            "http.request.body_too_large",
            route=route,
            limit_bytes=self.max_bytes,
            body_bytes=size,
        )
        request_id = scope.get("state", {}).get("request_id")
        body = _json_bytes(
            ErrorCode.VALIDATION_ERROR,
            _BODY_TOO_LARGE_MESSAGE,
            request_id if isinstance(request_id, str) else None,
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _declared_content_length(scope: Scope) -> int | None:
    """``Content-Length`` as an int, or ``None`` if absent or unparseable."""
    raw = Headers(scope=scope).get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _replaying(buffered: list[Message], receive: Receive) -> Receive:
    """A ``receive`` that yields already-read messages, then the real channel."""
    pending = list(buffered)

    async def replay() -> Message:
        if pending:
            return pending.pop(0)
        return await receive()

    return replay


class RequestTimeoutMiddleware:
    """Bounds every HTTP request in wall-clock time.

    A request that exceeds the budget before its response has started is
    answered with a 504 in the standard error envelope. If the response has
    already started there is nothing honest to do but let the connection end -
    the status line is already on the wire.
    """

    __slots__ = ("app", "exempt_paths", "metrics", "timeout_s")

    def __init__(
        self,
        app: ASGIApp,
        *,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        exempt_paths: frozenset[str] = UNTIMED_PATHS,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.app = app
        self.timeout_s = timeout_s
        self.exempt_paths = exempt_paths
        self.metrics = metrics if metrics is not None else get_metrics()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or str(scope.get("path", "")) in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            async with asyncio.timeout(self.timeout_s):
                await self.app(scope, receive, send_wrapper)
        except TimeoutError:
            self.metrics.increment_timeout(operation="http_request")
            self.metrics.increment_error(
                code=ErrorCode.UPSTREAM_TIMEOUT.value, route=route_template(scope)
            )
            _log.warning(
                "http.request.timeout",
                route=route_template(scope),
                budget_ms=round(self.timeout_s * 1000),
            )
            if response_started:
                return
            request_id = scope.get("state", {}).get("request_id")
            body = _json_bytes(
                ErrorCode.UPSTREAM_TIMEOUT,
                UpstreamTimeoutError.default_message,
                request_id if isinstance(request_id, str) else None,
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 504,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})


def _clean_request_id(value: str | None) -> str | None:
    """Adopt an inbound correlation id, if it is safe to log and echo.

    An inbound header is attacker-controlled, so it is accepted only when it is
    short and alphanumeric-ish. Anything else is discarded in favour of a fresh
    id, which keeps header injection and log forging out of the correlation
    path.
    """
    if not value:
        return None
    candidate = value.strip()
    if not 8 <= len(candidate) <= 64:
        return None
    if not all(character.isalnum() or character in "-_" for character in candidate):
        return None
    return candidate


def cors_kwargs(settings: Settings) -> dict[str, Any]:
    """Keyword arguments for ``CORSMiddleware``.

    Exact origins from ``FRONTEND_URL``, never ``"*"``: requests carry an
    ``Authorization`` header, and a wildcard origin with credentials is both
    rejected by browsers and wrong.
    """
    return {
        "allow_origins": list(settings.cors_origins),
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "X-Request-ID",
            "X-Dev-User",
        ],
        "expose_headers": ["X-Request-ID"],
        "max_age": 600,
    }
