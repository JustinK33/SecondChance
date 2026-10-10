"""Failure taxonomy for the model layer.

Everything that can go wrong with a model call is one of five things, and the
five stay distinguishable all the way to the API envelope because each one needs
a different reaction:

======================  ============================  ==========  =============
Error                   Cause                         Retryable?  HTTP
======================  ============================  ==========  =============
:class:`ModelTimeout`   we gave up waiting            yes         504
:class:`ModelRateLimited` provider returned 429       yes         429
:class:`ModelUnavailable` 5xx, refusal, or transport  depends     503
:class:`ModelInvalidOutput` provider answered, the    **no**      502
                        answer did not validate
:class:`ModelBlocked`   safety / recitation filter    **no**      502
======================  ============================  ==========  =============

The important line in that table is :class:`ModelInvalidOutput`. "The provider
is down" and "our prompt or schema is wrong" are the two failures that get
conflated in practice, and conflating them is expensive: an unavailable provider
should be retried and then fallen back from, whereas a schema violation is a
defect in this repository that retrying only makes more expensive. So they are
separate classes, and the retry loop in :mod:`app.ai.gemini` never retries this
one.

:class:`ModelUnavailable` carries :attr:`ModelUnavailable.transient`, because a
503 and a 403 ``PERMISSION_DENIED`` are both "the provider would not serve this"
but only the first is worth retrying. The class-level
``AppError.is_retryable`` flag describes the *wire* contract (may the client
retry the whole request) and stays as the base class defines it; ``transient``
is the narrower question the retry loop asks.

The ``Model*`` names have no ``Error`` suffix, which is why every class carries
``# noqa: N818``: these five names are the agreed vocabulary for this layer, and
they read better at the call site - ``except ModelRateLimited`` - than a
``ModelRateLimitedError`` would. The suffix rule is waived deliberately, once,
here.

Every class here subclasses the shared hierarchy in :mod:`app.core.errors`, so
the API exception handler needs no knowledge of the AI layer, and ``detail``
only ever carries the task name, the resolved model id and a status code -
never a prompt, a narrative or any profile content.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import (
    InvalidModelOutputError,
    RateLimitedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)

__all__ = [
    "ModelBlocked",
    "ModelError",
    "ModelInvalidOutput",
    "ModelRateLimited",
    "ModelTimeout",
    "ModelUnavailable",
]


def _detail(
    task: str | None,
    model: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a user-safe ``detail`` mapping for a model failure.

    Only non-personal, low-cardinality values go in here: the logical task, the
    resolved model id and a provider status code. It is returned verbatim to the
    client inside the error envelope, so anything else would be a leak.
    """
    detail: dict[str, Any] = {}
    if task is not None:
        detail["task"] = task
    if model is not None:
        detail["model"] = model
    if extra:
        detail.update(extra)
    return detail or None


class ModelError(Exception):
    """Marker mixin so ``except ModelError`` catches the whole AI taxonomy.

    A mixin rather than a base class because each member must keep the HTTP
    status and :class:`~app.core.errors.ErrorCode` of its counterpart in
    :mod:`app.core.errors`; multiple inheritance keeps both facts true at once.
    """


class ModelTimeout(ModelError, UpstreamTimeoutError):  # noqa: N818
    """The model did not answer inside ``settings.gemini_timeout_s``."""

    default_message = "The assistant took too long to respond. Please try again in a moment."

    def __init__(
        self,
        message: str | None = None,
        *,
        task: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.timeout_s = timeout_s
        extra = {"timeout_s": timeout_s} if timeout_s is not None else None
        super().__init__(message, detail=_detail(task, model, extra), cause=cause)

    #: Waiting longer may well work; the retry loop treats this as transient.
    transient = True


class ModelRateLimited(ModelError, RateLimitedError):  # noqa: N818
    """The provider returned 429, and the retry budget was exhausted."""

    default_message = "The assistant is busy right now. Please wait a moment and try again."

    def __init__(
        self,
        message: str | None = None,
        *,
        task: str | None = None,
        model: str | None = None,
        retry_after_s: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.task = task
        self.model = model
        super().__init__(
            message,
            retry_after_s=retry_after_s,
            detail=_detail(task, model),
            cause=cause,
        )

    transient = True


class ModelUnavailable(ModelError, UpstreamUnavailableError):  # noqa: N818
    """The provider failed or refused to serve the request.

    Covers 5xx responses, transport failures, and the non-retryable refusals -
    403 ``PERMISSION_DENIED`` and 404 ``NOT_FOUND`` for a retired model - which
    are what this project's API key currently returns (``docs/TOOLCHAIN.md``).
    ``transient`` is what separates them.
    """

    default_message = "The assistant is temporarily unavailable. Please try again shortly."

    def __init__(
        self,
        message: str | None = None,
        *,
        task: str | None = None,
        model: str | None = None,
        status: int | None = None,
        transient: bool = True,
        cause: BaseException | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.status = status
        self.transient = transient
        extra = {"status": status} if status is not None else None
        super().__init__(message, detail=_detail(task, model, extra), cause=cause)


class ModelInvalidOutput(ModelError, InvalidModelOutputError):  # noqa: N818
    """The provider answered and the answer failed validation.

    Our problem, not theirs: an unparseable body, a schema violation, or a
    response truncated at ``MAX_TOKENS``. Never retried - the same prompt
    produces the same shape - and never persisted.
    """

    default_message = "We could not read that result reliably, so we did not use it."

    def __init__(
        self,
        message: str | None = None,
        *,
        task: str | None = None,
        model: str | None = None,
        reason: str | None = None,
        validation_summary: list[str] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.reason = reason
        #: Field paths and error types from a schema violation. Never the
        #: offending values - see :mod:`app.ai.decode`.
        self.validation_summary = validation_summary or []
        extra = {"reason": reason} if reason is not None else None
        super().__init__(message, detail=_detail(task, model, extra), cause=cause)

    transient = False


class ModelBlocked(ModelError, InvalidModelOutputError):  # noqa: N818
    """A safety, recitation or similar filter stopped the response.

    Shares ``InvalidModelOutputError``'s 502 because the shape of the problem is
    the same - the provider worked, there is no usable output - but it is a
    distinct class so the caller can fall back silently instead of reporting a
    defect, and so a spike in blocked responses is visible in the logs on its
    own.
    """

    default_message = "We could not produce an answer for that. Please try rewording it."

    def __init__(
        self,
        message: str | None = None,
        *,
        task: str | None = None,
        model: str | None = None,
        reason: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.reason = reason
        extra = {"reason": reason} if reason is not None else None
        super().__init__(message, detail=_detail(task, model, extra), cause=cause)

    transient = False
