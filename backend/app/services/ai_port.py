"""Choosing which :class:`~app.schemas.contracts.GeminiPort` the app runs on.

``app.ai`` owns the implementations. This module owns the *selection*, because
which port to inject is an application-boundary decision and both the HTTP
layer and the background runner have to make the same one.

Why this is not a one-line factory
----------------------------------
Per ``docs/TOOLCHAIN.md`` **no live Gemini call can currently succeed with this
project's key**: the 2.5 generation is retired for new projects and the project
is not authorised for 3.x, so ``generateContent`` answers ``404`` for every
configured model. The key itself is valid, so ``GEMINI_API_KEY`` is set and
``Settings.gemini_enabled`` is ``True``.

Selecting purely on ``gemini_enabled`` therefore produced a product that
returned ``503 upstream_unavailable`` from ``POST /api/v1/intake`` and failed
every workflow run - with a non-model implementation sitting right there. The
port built here is a *degrading* one instead:

* try the live client,
* on any :class:`~app.ai.errors.ModelError`, serve the non-model implementation
  for that call and latch into a cool-off so the next few hundred calls do not
  each pay a failing round trip,
* re-probe the live client once the cool-off expires, so restoring model access
  needs no restart and no configuration change.

The honesty constraint is handled structurally, not by discipline. Every
degraded result is relabelled ``ModelUsage.source = "fallback"``, which is what
:attr:`~app.schemas.ai.GeminiResult.is_fallback` means - "came from the
deterministic path, not the model". The workflow nodes read that field and
attach a user-visible limitation saying the wording was not model-written, so
nothing can present a non-model answer as live model output. Relabelling *up*
to ``"live"`` is impossible: the live client is the only code that sets it.

The non-model implementation is currently :class:`~app.ai.fake.FakeGeminiClient`,
which replays validated fixtures through the same decoder and the same
projections as production. ``docs/INTERFACES.md`` specifies
``app.ai.fallback.DeterministicAi`` and ``app.ai.factory.build_gemini_port``;
neither exists yet. When they land, :func:`_build_deterministic` is the single
line that changes.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Final

from pydantic import BaseModel

from app.ai.base import BaseGeminiPort, CallContext, StructuredResult, TextResult
from app.ai.errors import ModelError
from app.ai.fake import FakeGeminiClient
from app.ai.gemini import provide_gemini_port
from app.ai.prompts.base import PromptSpec
from app.ai.schemas import GeminiTask
from app.core.config import Settings, get_settings
from app.core.errors import InternalError
from app.core.logging import get_logger
from app.schemas import GeminiPort, ModelUsage

__all__ = [
    "DEGRADED_COOL_OFF_S",
    "DegradingGeminiPort",
    "build_ai_port",
    "provide_ai_port",
    "reset_ai_port",
]

_log = get_logger(__name__)

#: How long one live failure suppresses further live attempts.
#:
#: A full run makes up to ~20 model calls. Without a latch, a retired model
#: costs 20 failing HTTPS round trips per run plus the client's own retries,
#: which turns a degraded run into a slow one. Five minutes is short enough
#: that granting model access is picked up on the next run.
DEGRADED_COOL_OFF_S: Final[float] = 300.0


class DegradingGeminiPort(BaseGeminiPort):
    """A live port with a non-model port behind it.

    Wrapping at the ``structured`` / ``text`` primitives rather than at the five
    :class:`~app.schemas.contracts.GeminiPort` methods is deliberate: prompt
    building and response projection live in
    :class:`~app.ai.base.BaseGeminiPort` and stay shared, so a degraded answer
    goes through exactly the same validation and the same "drop anything the
    catalog does not support" projection as a live one. There is no second code
    path to keep honest.
    """

    __slots__ = ("_cool_off_s", "_degraded_until", "_deterministic", "_failures", "_live")

    def __init__(
        self,
        *,
        live: BaseGeminiPort | None,
        deterministic: BaseGeminiPort,
        cool_off_s: float = DEGRADED_COOL_OFF_S,
    ) -> None:
        self._live = live
        self._deterministic = deterministic
        self._cool_off_s = cool_off_s
        self._degraded_until = 0.0
        self._failures = 0

    # ------------------------------------------------------------------ state
    @property
    def has_live_client(self) -> bool:
        """Whether a live client was configured at all."""
        return self._live is not None

    @property
    def is_degraded(self) -> bool:
        """Whether live calls are currently suppressed."""
        return self._live is None or time.monotonic() < self._degraded_until

    @property
    def failures(self) -> int:
        """Live calls that have failed over this port's lifetime."""
        return self._failures

    def _enter_cool_off(self, task: GeminiTask, exc: ModelError) -> None:
        first = self._failures == 0
        self._failures += 1
        self._degraded_until = time.monotonic() + self._cool_off_s
        # Type and task only. A model error's message can quote the prompt,
        # which is the narrative.
        _log.warning(
            "ai.port.degraded",
            task=task.value,
            error_type=type(exc).__name__,
            cool_off_s=self._cool_off_s,
            failures=self._failures,
            first=first,
        )

    # ------------------------------------------------------------- primitives
    async def structured[M: BaseModel](
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        schema: type[M],
        context: CallContext | None = None,
    ) -> StructuredResult[M]:
        live = self._live
        if live is not None and not self.is_degraded:
            try:
                return await live.structured(task, prompt, schema, context)
            except ModelError as exc:
                self._enter_cool_off(task, exc)
                return await self._deterministic_structured(task, prompt, schema, context, exc)
        return await self._deterministic_structured(task, prompt, schema, context, None)

    async def text(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        context: CallContext | None = None,
    ) -> TextResult:
        live = self._live
        if live is not None and not self.is_degraded:
            try:
                return await live.text(task, prompt, context)
            except ModelError as exc:
                self._enter_cool_off(task, exc)
                return await self._deterministic_text(task, prompt, context, exc)
        return await self._deterministic_text(task, prompt, context, None)

    # ------------------------------------------------------- degraded answers
    async def _deterministic_structured[M: BaseModel](
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        schema: type[M],
        context: CallContext | None,
        live_failure: ModelError | None,
    ) -> StructuredResult[M]:
        try:
            result = await self._deterministic.structured(task, prompt, schema, context)
        except Exception as exc:
            raise self._unrecoverable(task, live_failure, exc) from exc
        return dataclasses.replace(result, usage=_as_fallback(result))

    async def _deterministic_text(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        context: CallContext | None,
        live_failure: ModelError | None,
    ) -> TextResult:
        try:
            result = await self._deterministic.text(task, prompt, context)
        except Exception as exc:
            raise self._unrecoverable(task, live_failure, exc) from exc
        return dataclasses.replace(result, usage=_as_fallback(result))

    def _unrecoverable(
        self, task: GeminiTask, live_failure: ModelError | None, exc: Exception
    ) -> BaseException:
        """The exception to raise when the non-model path also failed.

        The live failure is preferred when there was one: it is the real cause,
        it already carries a plain-language message and a retryable error code,
        and "the model is unavailable" is a truer thing to tell a user than
        whatever went wrong in the stand-in.
        """
        _log.error(
            "ai.port.deterministic_failed",
            task=task.value,
            error_type=type(exc).__name__,
            had_live_failure=live_failure is not None,
        )
        if live_failure is not None:
            return live_failure
        if isinstance(exc, ModelError):
            return exc
        msg = f"no usable model port for task {task.value!r}"
        return InternalError(msg, cause=exc)


def _as_fallback(result: StructuredResult[BaseModel] | TextResult) -> ModelUsage:
    """``result.usage`` relabelled as non-model output.

    ``FakeGeminiClient`` reports ``source="cache"``, which is true about the
    mechanism and wrong about what the caller needs to know. The workflow asks
    one question - did a model write this? - via ``source == "fallback"``, and
    an answer served because the model was unreachable has to answer "no", or
    the run completes with no limitation attached and the interface renders
    non-model wording as model wording.
    """
    return result.usage.model_copy(update={"source": "fallback"})


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def _build_deterministic(settings: Settings) -> BaseGeminiPort:
    """The non-model implementation.

    One line, one place. ``app.ai.fallback.DeterministicAi`` replaces
    ``FakeGeminiClient`` here when the AI agent lands it; nothing else in the
    application needs to know.
    """
    return FakeGeminiClient(settings)


def build_ai_port(settings: Settings) -> GeminiPort:
    """The port for ``settings``. Not cached; use :func:`provide_ai_port`."""
    deterministic = _build_deterministic(settings)
    if settings.gemini_enabled:
        live = provide_gemini_port()
        if not isinstance(live, BaseGeminiPort):
            # ``set_gemini_port_override`` installed something that is not one
            # of the prompt-sharing implementations - a benchmark stub, a test
            # double. An explicit override is a decision, so it is honoured
            # verbatim rather than wrapped in degradation logic it cannot
            # participate in.
            return live
        return DegradingGeminiPort(live=live, deterministic=deterministic)
    if not settings.is_local:  # pragma: no cover - Settings rejects this first
        msg = "GEMINI_API_KEY must be set outside APP_ENV=local"
        raise InternalError(msg)
    _log.warning(
        "ai.port.no_model_access",
        reason="no_api_key",
        note="answers are non-model output and report source='fallback'",
    )
    return DegradingGeminiPort(live=None, deterministic=deterministic)


_cached: GeminiPort | None = None
_cached_for: tuple[bool, str] | None = None


def provide_ai_port(settings: Settings | None = None) -> GeminiPort:
    """The process-wide port, built once.

    Cached because the live client holds an HTTP connection pool, and one pool
    per process is the point of a pool - and because the degradation cool-off
    is only useful if every caller shares it. The cache key includes the
    resolved mode so a test that flips the key gets a different port.
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
