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
3. :class:`RequestTimeoutMiddleware` - bounds the handler in wall-clock time.
4. The router.

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
    "REQUEST_ID_HEADER",
    "UNTIMED_PATHS",
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

#: Paths exempt from the request timeout. The liveness probe must answer even
#: when the process is saturated, and the metrics scrape is read by a collector
#: with its own timeout.
UNTIMED_PATHS: Final[frozenset[str]] = frozenset(
    {"/healthz", "/api/v1/health", "/internal/metrics"}
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
