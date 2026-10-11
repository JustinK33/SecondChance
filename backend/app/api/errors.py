"""Exception handlers: one wire shape for every non-2xx response.

The frontend has exactly one error branch to write, so every failure - raised
by this application, by FastAPI's validation layer, or by nothing at all -
leaves here as ``{"error": {code, message, detail, request_id}}``.

Four handlers, in the order they are consulted:

:class:`~app.core.errors.AppError`
    Already carries a vetted plain-language ``message``, a stable ``code`` and
    a user-safe ``detail``. Rendered as-is with the request id attached.

``RequestValidationError``
    422 ``validation_error``. The detail names the **field locations only** -
    never the submitted values. A validation error quoting
    ``income_min_cents: 184000`` would put a user's income in an error
    response, a browser console, a log aggregator and quite possibly a
    screenshot in a support ticket.

``StarletteHTTPException``
    A framework-raised 404 or 405, mapped through
    :func:`~app.core.errors.error_code_for_status` so even the router's own
    errors keep the shape.

Anything else
    500 ``internal_error`` with the class default message. The exception is
    logged with ``exc_info`` and **nothing** from it reaches the response: an
    unexpected exception's text is written for an engineer reading a stack
    trace and may quote the very content this product promises never to
    surface.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import (
    AppError,
    ErrorCode,
    ErrorResponse,
    InternalError,
    InvalidRequestError,
    RateLimitedError,
    error_code_for_status,
)
from app.core.logging import get_logger
from app.core.metrics import get_metrics
from app.core.middleware import route_template

__all__ = ["error_responses", "install_exception_handlers", "request_id_of"]

_log = get_logger(__name__)

#: How many field locations a validation error reports. A request that got
#: twenty fields wrong does not need twenty of them echoed back.
_MAX_REPORTED_FIELDS = 10


def request_id_of(request: Request) -> str | None:
    """The correlation id the middleware put on this request."""
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def _envelope(
    request: Request, error: AppError, *, headers: dict[str, str] | None = None
) -> JSONResponse:
    request_id = request_id_of(request)
    get_metrics().increment_error(code=error.code.value, route=route_template(request.scope))
    body: ErrorResponse = error.to_response(request_id)
    merged = dict(headers or {})
    if request_id:
        # The unhandled-exception handler runs inside Starlette's
        # ServerErrorMiddleware, which is *outside* RequestContextMiddleware,
        # so a 500 would otherwise go out without the correlation header even
        # though its body carries the id.
        merged.setdefault("X-Request-ID", request_id)
    return JSONResponse(
        status_code=error.status_code,
        content=body.model_dump(mode="json"),
        headers=merged or None,
    )


def _field_locations(errors: list[dict[str, Any]]) -> list[str]:
    """Dotted field paths from a validation error, values excluded.

    ``loc`` is a tuple like ``("body", "household_size")``. Only the path is
    kept; ``msg``, ``input`` and ``ctx`` are all dropped because any of them
    can echo the submitted value.
    """
    seen: dict[str, None] = {}
    for entry in errors:
        location = entry.get("loc") or ()
        parts = [str(part) for part in location if part not in {"body", "query", "path"}]
        if not parts:
            parts = [str(part) for part in location]
        if parts:
            seen.setdefault(".".join(parts), None)
    return list(seen)[:_MAX_REPORTED_FIELDS]


def install_exception_handlers(app: FastAPI) -> None:
    """Register the four handlers on ``app``."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: Exception) -> JSONResponse:
        error = exc if isinstance(exc, AppError) else InternalError(cause=exc)
        headers: dict[str, str] | None = None
        if isinstance(error, RateLimitedError) and error.retry_after_s is not None:
            headers = {"Retry-After": str(error.retry_after_s)}
        # Codes and the route template only. ``error.message`` is vetted copy
        # and ``__cause__`` is where the untrusted text lives, so neither is
        # interpolated into a log line here.
        _log.warning(
            "api.error",
            code=error.code.value,
            status=error.status_code,
            route=route_template(request.scope),
            error_type=type(error).__name__,
        )
        return _envelope(request, error, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
        fields: list[str] = []
        if isinstance(exc, RequestValidationError):
            fields = _field_locations(list(exc.errors()))
        error = InvalidRequestError(detail={"fields": fields} if fields else None)
        _log.info(
            "api.validation_error",
            route=route_template(request.scope),
            field_count=len(fields),
            fields=fields,
        )
        return _envelope(request, error)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: Exception) -> JSONResponse:
        status_code = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
        code = error_code_for_status(status_code)
        # A framework exception carries no vetted copy of its own, so it is
        # adapted into an AppError whose status and code are set per instance.
        error = AppError(_message_for(code))
        error.status_code = status_code
        error.code = code
        headers = getattr(exc, "headers", None)
        return _envelope(request, error, headers=headers if isinstance(headers, dict) else None)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # ``exc_info`` goes to the log, which is redacted by
        # ``app.core.logging``; the response carries the class default message
        # and the request id, and nothing else.
        _log.error(
            "api.unhandled_exception",
            route=route_template(request.scope),
            error_type=type(exc).__name__,
            exc_info=exc,
        )
        return _envelope(request, InternalError(cause=exc))


def _message_for(code: ErrorCode) -> str:
    """Vetted copy for a framework-raised status."""
    return _FRAMEWORK_MESSAGES.get(code, InternalError.default_message)


_FRAMEWORK_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.NOT_FOUND: "We could not find what you were looking for.",
    ErrorCode.UNAUTHORIZED: "Please sign in to continue.",
    ErrorCode.FORBIDDEN: "You do not have access to this.",
    ErrorCode.VALIDATION_ERROR: (
        "Some of the information sent was not valid. Please check and try again."
    ),
    ErrorCode.CONFLICT: "That change conflicts with something that already exists.",
    ErrorCode.RATE_LIMITED: "Too many requests. Please wait a moment and try again.",
    ErrorCode.UPSTREAM_TIMEOUT: (
        "This is taking longer than expected. You can try again in a moment."
    ),
    ErrorCode.UPSTREAM_UNAVAILABLE: (
        "A service we rely on is temporarily unavailable. Please try again shortly."
    ),
}


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries for the documented failures of a route.

    Declaring them is what makes the generated TypeScript useful: without it
    the client's error type is ``unknown`` and every call site invents its own
    guess at the envelope.

    Response *headers* are declared here too, for the one status where a header
    is the actionable part of the answer: a 429 without ``Retry-After`` tells a
    client to back off without telling it how far, so the client guesses.
    """
    responses: dict[int | str, dict[str, Any]] = {}
    for status in sorted(set(status_codes)):
        entry: dict[str, Any] = {
            "model": ErrorResponse,
            "description": _OPENAPI_DESCRIPTIONS.get(status, "Error"),
        }
        headers = _OPENAPI_HEADERS.get(status)
        if headers:
            entry["headers"] = headers
        responses[status] = entry
    return responses


_OPENAPI_DESCRIPTIONS: dict[int, str] = {
    401: "No valid credentials were supplied.",
    403: "Authenticated, but not allowed to do this.",
    404: "No such resource, or it belongs to someone else.",
    409: "Conflicts with the current state of the resource.",
    413: "The request body was larger than this API accepts.",
    422: "The request body or query string failed validation.",
    429: "Rate limited. Retry after the interval in the Retry-After header.",
    500: "Unexpected server-side failure.",
    502: "A model answered, but the answer failed schema or grounding validation.",
    503: "A dependency is unavailable.",
    504: "A dependency did not answer in time.",
}


#: Declared response headers, by status. Kept next to the descriptions so a
#: route cannot document the status and forget the header that makes it usable.
_OPENAPI_HEADERS: dict[int, dict[str, Any]] = {
    429: {
        "Retry-After": {
            "description": "Seconds to wait before retrying.",
            "schema": {"type": "integer", "minimum": 0},
        }
    },
}
