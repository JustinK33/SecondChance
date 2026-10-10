"""Choosing which :class:`~app.schemas.contracts.GeminiPort` the app runs on.

``app.ai`` owns the implementations. This module owns the *selection*, because
which port to inject is an application-boundary decision and both the HTTP
layer and the background runner have to make the same one.

The rule is config-driven and has no escape hatch:

* ``GEMINI_API_KEY`` set -> the live client. Every call is a real model call and
  ``ModelUsage.source`` records ``"live"``.
* ``GEMINI_API_KEY`` empty -> the fixture-replaying client, and only when
  ``APP_ENV=local``. Outside local, ``Settings`` already refuses to construct
  with an empty key, so this branch is unreachable in a deployment.

Why the second branch exists at all: per ``docs/TOOLCHAIN.md`` no live Gemini
call can currently succeed with this project's key - the 2.5 generation is
retired for new projects and this project is not authorised for 3.x - so
without it the product could not be run or demonstrated end to end today. The
honesty constraint is handled by the type system rather than by discipline:
``ModelUsage.source`` is ``"cache"`` for every replayed response, the workflow
records a user-visible limitation saying which reader ran, and nothing can
present a replayed answer as live model output.

This is a stopgap. The handoff note asks ``app.ai`` for the
``DeterministicAi`` implementation and the ``build_gemini_port`` factory that
``docs/INTERFACES.md`` specifies; when they land, this module becomes a
one-line delegation to that factory.
"""

from __future__ import annotations

from app.ai.fake import FakeGeminiClient
from app.ai.gemini import provide_gemini_port
from app.core.config import Settings, get_settings
from app.core.errors import InternalError
from app.core.logging import get_logger
from app.schemas import GeminiPort

__all__ = ["build_ai_port", "provide_ai_port"]

_log = get_logger(__name__)

_cached: GeminiPort | None = None
_cached_for: tuple[bool, str] | None = None


def build_ai_port(settings: Settings) -> GeminiPort:
    """The port for ``settings``. Not cached; use :func:`provide_ai_port`."""
    if settings.gemini_enabled:
        return provide_gemini_port()
    if not settings.is_local:  # pragma: no cover - Settings rejects this first
        msg = "GEMINI_API_KEY must be set outside APP_ENV=local"
        raise InternalError(msg)
    _log.warning(
        "ai.port.replaying_fixtures",
        reason="no_api_key",
        note="responses are replayed fixtures and report source='cache'",
    )
    return FakeGeminiClient(settings)


def provide_ai_port(settings: Settings | None = None) -> GeminiPort:
    """The process-wide port, built once.

    Cached because the live client holds an HTTP connection pool, and one pool
    per process is the point of a pool. The cache key includes the resolved
    mode so a test that flips the key gets a different port.
    """
    global _cached, _cached_for
    resolved = settings or get_settings()
    key = (resolved.gemini_enabled, resolved.app_env)
    if _cached is None or _cached_for != key:
        _cached = build_ai_port(resolved)
        _cached_for = key
    return _cached


def reset_ai_port() -> None:
    """Drop the cached port. For tests and configuration reloads."""
    global _cached, _cached_for
    _cached = None
    _cached_for = None
