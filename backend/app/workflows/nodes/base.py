"""Shared plumbing for the workflow nodes.

Three things every node needs, in one place so they cannot drift:

* **Phase publication.** :func:`entering` returns the ``phase`` write for a
  node, or nothing for a fan-out branch (which must not write ``phase`` - see
  :mod:`app.workflows.state`).
* **Recoverable failure.** :func:`recovered` turns an exception into the state
  update that records it: a user-safe :class:`UserFacingError` on ``errors``
  and a plain-language caveat on ``limitations``. The graph keeps running and
  the user gets a degraded but honest result instead of a dead run.
* **Logging that cannot leak.** :func:`log_failure` records a node name, an
  error code and a retryability flag. Never a narrative, an income, a prompt or
  an exception string from a provider.

What is *not* recoverable is deliberately short. If profile extraction fails
there is no profile, so there is nothing to retrieve, evaluate or explain, and
pretending otherwise would persist an empty result as if it were an answer.
Those exceptions propagate out of ``ainvoke`` and
:func:`~app.workflows.graph.run_workflow` turns them into a failed run whose
checkpoint is still resumable.
"""

from __future__ import annotations

import uuid
from typing import Any, Final

from app.core.errors import (
    AppError,
    ErrorCode,
    InternalError,
    InvalidModelOutputError,
    RateLimitedError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from app.core.logging import get_logger
from app.schemas import UserFacingError, UserProfile, utcnow
from app.workflows.state import PHASE_BY_NODE, NavigatorState, StateUpdate

__all__ = [
    "RECOVERABLE_ERRORS",
    "entering",
    "log_failure",
    "profile_of",
    "recovered",
    "run_id_of",
    "user_facing",
]

_log = get_logger(__name__)

#: Failures a single branch can absorb. All four mean "a dependency did not
#: answer usefully this time", and every one of them leaves the deterministic
#: half of the product - retrieval, the rule engine, the evidence - intact.
RECOVERABLE_ERRORS: Final[tuple[type[AppError], ...]] = (
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    RateLimitedError,
    InvalidModelOutputError,
)

#: What the user can actually do about each failure. Short, concrete, and never
#: a suggestion to contact support about something they cannot describe.
_RETRY_HINTS: Final[dict[ErrorCode, str]] = {
    ErrorCode.UPSTREAM_TIMEOUT: "Try again in a moment - your answers are saved.",
    ErrorCode.UPSTREAM_UNAVAILABLE: "Try again shortly - your answers are saved.",
    ErrorCode.RATE_LIMITED: "We are busy right now. Try again in a minute.",
    ErrorCode.INVALID_MODEL_OUTPUT: "Your results are still here; only the wording is missing.",
    ErrorCode.WORKFLOW_FAILED: "You can run this again without re-entering anything.",
}


def user_facing(exc: BaseException) -> UserFacingError:
    """A user-safe description of a failure.

    An :class:`~app.core.errors.AppError` already carries vetted plain-language
    copy. Anything else is deliberately flattened into the generic internal
    error: an unexpected exception's message is written for an engineer reading
    a stack trace, and may quote the very content this product promises never
    to surface.
    """
    error = exc if isinstance(exc, AppError) else InternalError(cause=exc)
    return UserFacingError(
        code=error.code,
        message=error.message,
        is_retryable=error.is_retryable,
        retry_hint=_RETRY_HINTS.get(error.code),
        occurred_at=utcnow(),
    )


def entering(node: str) -> StateUpdate:
    """The phase write for ``node``, or an empty update for a fan-out branch."""
    phase = PHASE_BY_NODE.get(node)
    return {"phase": phase} if phase is not None else {}


def log_failure(node: str, exc: BaseException, *, recoverable: bool) -> None:
    """Record a failure as codes and flags only."""
    error = exc if isinstance(exc, AppError) else None
    _log.warning(
        "workflow.node.failed",
        node=node,
        error_code=error.code.value if error else ErrorCode.INTERNAL_ERROR.value,
        error_type=type(exc).__name__,
        recoverable=recoverable,
        is_retryable=bool(error and error.is_retryable),
    )


def recovered(
    exc: BaseException,
    *,
    node: str,
    limitation: str,
    extra: dict[str, Any] | None = None,
) -> StateUpdate:
    """The state update for a failure one branch absorbed.

    ``limitation`` is the sentence the user reads. It says what is missing from
    their results, never what went wrong technically, because "the explanation
    service timed out" is not information a person can act on.
    """
    log_failure(node, exc, recoverable=True)
    update: StateUpdate = {
        "errors": [user_facing(exc)],
        "limitations": [limitation],
    }
    if extra:
        update.update(extra)
    return update


def run_id_of(state: NavigatorState) -> uuid.UUID:
    """The run id, which is also the checkpointer's ``thread_id``."""
    run_id = state.get("run_id")
    if run_id is None:  # pragma: no cover - build_initial_state always sets it
        msg = "workflow state has no run_id"
        raise InternalError(msg)
    return run_id


def profile_of(state: NavigatorState) -> UserProfile:
    """The current profile, or an empty one.

    An empty profile is a legitimate state, not an error: every rule resolves
    to ``UNKNOWN`` against it and every program comes back as
    ``MORE_INFO_NEEDED``, which is exactly the honest answer when we know
    nothing yet.
    """
    return state.get("profile") or UserProfile()
