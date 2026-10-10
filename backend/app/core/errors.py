"""Error taxonomy and the single error envelope used by every API response.

Three ideas hold this together:

1. **One wire shape.** Every non-2xx response body is an :class:`ErrorResponse`,
   so the frontend has exactly one error branch to write.
2. **Machine code, human message.** ``code`` is a stable
   :class:`ErrorCode` the frontend switches on; ``message`` is plain-language
   copy that is safe to render to a nontechnical user as-is.
3. **Nothing leaks.** ``AppError.message`` always falls back to a vetted
   class-level string, and ``detail`` is a deliberately populated, user-safe
   mapping. Exception text from a database driver or model SDK never becomes a
   user-visible message unless an author explicitly writes one.

This module holds types only. The FastAPI exception handlers that turn an
:class:`AppError` into a response live in ``app/api`` and are owned by the
backend-infrastructure agent.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AppError",
    "ConflictError",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "ForbiddenError",
    "InternalError",
    "InvalidModelOutputError",
    "InvalidRequestError",
    "NotFoundError",
    "RateLimitedError",
    "UnauthorizedError",
    "UpstreamTimeoutError",
    "UpstreamUnavailableError",
    "WorkflowFailedError",
    "error_code_for_status",
]


class ErrorCode(StrEnum):
    """Stable machine-readable error identifiers.

    These values are part of the public API contract. Add new members freely;
    never rename or repurpose an existing one.
    """

    VALIDATION_ERROR = "validation_error"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    WORKFLOW_FAILED = "workflow_failed"
    INTERNAL_ERROR = "internal_error"


class ErrorDetail(BaseModel):
    """The inner object of the error envelope."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode = Field(description="Stable machine-readable error identifier.")
    message: str = Field(
        description="Plain-language, user-safe explanation. Safe to render verbatim."
    )
    detail: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional structured, user-safe context, for example "
            '{"field": "household_size"}. Never contains personal content, '
            "upstream exception text or stack traces."
        ),
    )
    request_id: str | None = Field(
        default=None,
        description="Correlation id for this request. Quote it when reporting a problem.",
    )


class ErrorResponse(BaseModel):
    """The body of every non-2xx API response: ``{"error": {...}}``."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail


class AppError(Exception):
    """Base class for every error this application raises deliberately.

    Subclasses set :attr:`status_code`, :attr:`code` and
    :attr:`default_message`. Raising the base class directly yields a generic
    500.

    ``message`` is for the end user. ``detail`` carries structured context the
    frontend can act on. Anything that must not reach a user - driver messages,
    prompts, profile content - belongs in ``__cause__`` and in the logs, which
    are redacted by :mod:`app.core.logging`.
    """

    status_code: int = 500
    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    default_message: str = "Something went wrong on our side. Please try again."
    #: Whether retrying the identical request could plausibly succeed.
    is_retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.detail = detail
        super().__init__(self.message)
        if cause is not None:
            self.__cause__ = cause

    def to_detail(self, request_id: str | None = None) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            message=self.message,
            detail=self.detail,
            request_id=request_id,
        )

    def to_response(self, request_id: str | None = None) -> ErrorResponse:
        """Render this error as the envelope body for an HTTP response."""
        return ErrorResponse(error=self.to_detail(request_id))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r}, status={self.status_code})"


class InvalidRequestError(AppError):
    """The request was well-formed JSON but is not valid for this operation.

    Named to avoid colliding with ``pydantic.ValidationError``; its wire code is
    still ``validation_error``.
    """

    status_code = 422
    code = ErrorCode.VALIDATION_ERROR
    default_message = "Some of the information sent was not valid. Please check and try again."


class UnauthorizedError(AppError):
    status_code = 401
    code = ErrorCode.UNAUTHORIZED
    default_message = "Please sign in to continue."


class ForbiddenError(AppError):
    status_code = 403
    code = ErrorCode.FORBIDDEN
    default_message = "You do not have access to this."


class NotFoundError(AppError):
    status_code = 404
    code = ErrorCode.NOT_FOUND
    default_message = "We could not find what you were looking for."


class ConflictError(AppError):
    status_code = 409
    code = ErrorCode.CONFLICT
    default_message = "That change conflicts with something that already exists."


class RateLimitedError(AppError):
    status_code = 429
    code = ErrorCode.RATE_LIMITED
    default_message = "Too many requests. Please wait a moment and try again."
    is_retryable = True

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_s: int | None = None,
        detail: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        merged = dict(detail or {})
        if retry_after_s is not None:
            merged["retry_after_s"] = retry_after_s
        self.retry_after_s = retry_after_s
        super().__init__(message, detail=merged or None, cause=cause)


class UpstreamTimeoutError(AppError):
    """A dependency (Gemini, an official source) did not answer in time."""

    status_code = 504
    code = ErrorCode.UPSTREAM_TIMEOUT
    default_message = "This is taking longer than expected. You can try again in a moment."
    is_retryable = True


class UpstreamUnavailableError(AppError):
    """A dependency is failing or refusing requests."""

    status_code = 503
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    default_message = "A service we rely on is temporarily unavailable. Please try again shortly."
    is_retryable = True


class InvalidModelOutputError(AppError):
    """The model answered, but the answer failed schema or grounding validation.

    Deliberately distinct from :class:`UpstreamUnavailableError`: the provider
    worked, the output did not. Retrying may help; persisting the output must
    never happen.
    """

    status_code = 502
    code = ErrorCode.INVALID_MODEL_OUTPUT
    default_message = "We could not read that result reliably, so we did not use it."
    is_retryable = True


class WorkflowFailedError(AppError):
    """A workflow run ended without producing results."""

    status_code = 500
    code = ErrorCode.WORKFLOW_FAILED
    default_message = (
        "We could not finish checking your programs. You can try again without losing your answers."
    )
    is_retryable = True


class InternalError(AppError):
    """An unexpected defect. The cause is logged, never returned."""

    status_code = 500
    code = ErrorCode.INTERNAL_ERROR


_STATUS_TO_CODE: Final[dict[int, ErrorCode]] = {
    400: ErrorCode.VALIDATION_ERROR,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.VALIDATION_ERROR,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    502: ErrorCode.UPSTREAM_UNAVAILABLE,
    503: ErrorCode.UPSTREAM_UNAVAILABLE,
    504: ErrorCode.UPSTREAM_TIMEOUT,
}


def error_code_for_status(status_code: int) -> ErrorCode:
    """Map a bare HTTP status onto an :class:`ErrorCode`.

    Used by the handler that converts framework-raised ``HTTPException`` and
    ``StarletteHTTPException`` instances into the standard envelope, so even
    errors raised inside FastAPI keep the one wire shape.
    """
    return _STATUS_TO_CODE.get(status_code, ErrorCode.INTERNAL_ERROR)
