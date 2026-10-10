"""The live Gemini client: transport, timeouts, retries and usage accounting.

Division of labour
------------------
:class:`~app.ai.base.BaseGeminiPort` owns prompt building and response
projection, and is shared with the fake. This module owns everything that
touches the network: model selection, the ``GenerateContentConfig``, the
timeout, the retry policy, the response sanity checks that a 200 does not
guarantee, and the :class:`~app.schemas.ai.ModelUsage` record.

Two primitives do all the I/O - :meth:`GeminiClient.structured` (a prompt plus a
pydantic schema in, a validated instance out) and :meth:`GeminiClient.text`.
Everything else is built on them.

Retry policy
------------
Only four things are retried: HTTP 429, 5xx, transport failures, and our own
timeout. Everything else fails immediately, and the one that matters is
:class:`~app.ai.errors.ModelInvalidOutput` - a schema violation is a defect in
this repository, and retrying it only pays for the same wrong answer twice.

Backoff is exponential with **full jitter** - ``uniform(0, base * 2**attempt)``
rather than ``base * 2**attempt`` - because a workflow fans out several
explanation calls at once, and synchronised retries are how one rate limit turns
into a thundering herd. A server-provided ``RetryInfo`` delay is honoured as a
floor on the wait, capped so a hostile or buggy value cannot park a request for
an hour.

Every attempt is wrapped in ``asyncio.timeout(settings.gemini_timeout_s)`` as
well as the SDK's own ``http_options`` timeout: the SDK timeout covers one HTTP
round trip, while the budget we care about is the whole call including its
streaming tail. ``Settings`` already refuses to start if
``gemini_timeout_s * (gemini_max_retries + 1)`` exceeds the workflow budget.

Logging
-------
One structured line per logical call: task, model, prompt name and version,
attempts, latency, token counts, cost and outcome. Never the prompt, the
narrative, the profile, or any part of the response. Token counts are logged as
``tok_in`` / ``tok_out`` rather than ``prompt_tokens`` / ``output_tokens``
because :mod:`app.core.logging` redacts any key containing ``token``. The counts
are not sensitive, but the denylist cannot know that, and renaming a key is
cheaper than weakening a privacy control.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final, cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.ai import decode
from app.ai.base import BaseGeminiPort, CallContext, StructuredResult, TextResult
from app.ai.errors import (
    ModelBlocked,
    ModelInvalidOutput,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.prompts.base import PromptSpec
from app.ai.schemas import MODEL_RESPONSE_SCHEMAS, GeminiTask
from app.ai.usage import UsageAccumulator, record_usage
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.ai import ModelUsage, PromptRef
from app.schemas.contracts import GeminiPort

__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_MAX_S",
    "MODEL_STAGE_FOR_TASK",
    "RETRY_AFTER_MAX_S",
    "TASK_SCHEMAS",
    "GeminiClient",
    "get_gemini_client",
    "provide_gemini_port",
    "reset_gemini_client",
    "set_gemini_port_override",
]

log = get_logger(__name__)

#: First backoff ceiling, in seconds. Not a ``Settings`` field because it is a
#: property of the provider's rate limiter rather than of a deployment.
BACKOFF_BASE_S: Final[float] = 0.5
#: Ceiling for one jittered wait.
BACKOFF_MAX_S: Final[float] = 8.0
#: Ceiling for a server-requested wait. A provider asking for five minutes is
#: better served by failing the call and letting the workflow fall back.
RETRY_AFTER_MAX_S: Final[float] = 30.0

#: Which configured model each task uses. Extraction and clarification are
#: mechanical and run on the cheap model; explanation and planning write the
#: user-visible prose and run on the stronger one; chat has its own so a
#: conversational model can be chosen independently. All three fall back to
#: ``GEMINI_MODEL`` - see ``Settings.extraction_model`` and friends.
MODEL_STAGE_FOR_TASK: Final[dict[GeminiTask, str]] = {
    GeminiTask.EXTRACT_PROFILE: "extraction_model",
    GeminiTask.PROPOSE_CLARIFICATIONS: "extraction_model",
    GeminiTask.EXPLAIN_MATCH: "explanation_model",
    GeminiTask.DRAFT_ACTION_PLAN: "explanation_model",
    GeminiTask.ANSWER_QUESTION: "chat_model",
}

#: Task -> response schema, re-exported so callers need not know which module
#: defines the schemas.
TASK_SCHEMAS: Final[dict[GeminiTask, type[BaseModel]]] = dict(MODEL_RESPONSE_SCHEMAS)

#: Finish reasons that mean a filter stopped the response.
_BLOCKED_FINISH_REASONS: Final[frozenset[str]] = frozenset(
    {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}
)
_TRUNCATED_FINISH_REASONS: Final[frozenset[str]] = frozenset({"MAX_TOKENS"})
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
#: Full jitter needs randomness, not secure randomness, but ``random`` trips the
#: linter's security rules; ``SystemRandom`` is the same API without a waiver.
_JITTER: Final[secrets.SystemRandom] = secrets.SystemRandom()


class GeminiClient(BaseGeminiPort):
    """Resilient google-genai client implementing :class:`GeminiPort`.

    ``client`` is injectable so a test can supply a stand-in with the same
    ``aio.models.generate_content`` surface, and ``sleep`` is injectable so a
    retry test can assert on the jittered delays without waiting for them.
    """

    def __init__(
        self,
        settings: Settings,
        client: genai.Client | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        accumulator: UsageAccumulator | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._client_lock = threading.Lock()
        self._sleep = sleep if sleep is not None else asyncio.sleep
        #: Running totals for this process. Superseded by ``app.core.metrics``
        #: once that module exists; see :mod:`app.ai.usage`.
        self.usage = accumulator if accumulator is not None else UsageAccumulator()

    # ------------------------------------------------------------------ setup
    @property
    def settings(self) -> Settings:
        return self._settings

    def model_for(self, task: GeminiTask) -> str:
        """Resolve the configured model id for one task."""
        model: str = getattr(self._settings, MODEL_STAGE_FOR_TASK[task])
        return model

    def _genai(self) -> genai.Client:
        """The underlying SDK client, created on first use.

        Lazy because constructing it reads the API key, and a process that never
        makes a model call - a migration, a unit test, the deterministic path -
        should not need one to exist.
        """
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                if not self._settings.gemini_enabled:
                    raise ModelUnavailable(
                        "The assistant is not configured on this server.",
                        transient=False,
                    )
                self._client = genai.Client(
                    api_key=self._settings.gemini_api_key.get_secret_value()
                )
            return self._client

    # --------------------------------------------------------------- low level
    async def structured[M: BaseModel](
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        schema: type[M],
        context: CallContext | None = None,
    ) -> StructuredResult[M]:
        """Run one structured call and return the validated response.

        The schema is handed to the SDK as ``response_schema`` *and* used to
        validate the body here, because ``response.parsed`` is ``None`` whenever
        the SDK's own parse failed; trusting it blindly would turn a schema
        violation into an ``AttributeError`` three layers up.
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            system_instruction=prompt.system_instruction,
            http_options=types.HttpOptions(timeout=self._settings.gemini_timeout_ms),
        )
        response, usage = await self._generate(task, prompt, config, context)
        value, notes = decode.decode_structured(
            schema,
            text=_response_text(response),
            parsed=getattr(response, "parsed", None),
            task=task.value,
            model=usage.model,
        )
        return StructuredResult(value=value, usage=usage, notes=notes)

    async def text(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        context: CallContext | None = None,
    ) -> TextResult:
        """Run one free-form call and return its text."""
        config = types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=prompt.system_instruction,
            http_options=types.HttpOptions(timeout=self._settings.gemini_timeout_ms),
        )
        response, usage = await self._generate(task, prompt, config, context)
        body = _response_text(response)
        if body is None or not body.strip():
            raise ModelInvalidOutput(task=task.value, model=usage.model, reason="empty_response")
        return TextResult(text=body.strip(), usage=usage)

    # ------------------------------------------------------------- the engine
    async def _generate(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        config: types.GenerateContentConfig,
        context: CallContext | None,
    ) -> tuple[Any, ModelUsage]:
        """Call the provider with a timeout, bounded retries and full jitter."""
        model_id = self.model_for(task)
        max_attempts = self._settings.gemini_max_retries + 1
        started = time.monotonic()

        for attempt in range(max_attempts):
            attempts = attempt + 1
            failure: Exception
            try:
                async with asyncio.timeout(self._settings.gemini_timeout_s):
                    response = await self._genai().aio.models.generate_content(
                        model=model_id,
                        contents=prompt.prompt,
                        config=config,
                    )
            except (TimeoutError, httpx.TimeoutException) as exc:
                failure = ModelTimeout(
                    task=task.value,
                    model=model_id,
                    timeout_s=self._settings.gemini_timeout_s,
                    cause=exc,
                )
            except genai_errors.APIError as exc:
                failure = _classify_api_error(exc, task=task.value, model=model_id)
            except (httpx.HTTPError, OSError) as exc:
                failure = ModelUnavailable(
                    task=task.value, model=model_id, transient=True, cause=exc
                )
            else:
                usage = self._usage_for(
                    response,
                    task=task,
                    model_id=model_id,
                    prompt=prompt.ref,
                    attempts=attempts,
                    started=started,
                )
                self._check_response(response, task=task, model_id=model_id, usage=usage)
                self._log_call(task, usage, context)
                record_usage(usage, self.usage)
                return response, usage

            is_last = attempt == max_attempts - 1
            if not getattr(failure, "transient", False) or is_last:
                self._log_failure(task, model_id, prompt.ref, attempts, started, context, failure)
                raise failure
            delay = _backoff_delay(attempt, _retry_after_s(failure))
            log.warning(
                "gemini.retry",
                task=task.value,
                model=model_id,
                prompt_name=prompt.ref.name,
                prompt_version=prompt.ref.version,
                attempt=attempts,
                delay_ms=int(delay * 1000),
                error=type(failure).__name__,
                status=getattr(failure, "status", None),
                request_id=_request_id(context),
            )
            await self._sleep(delay)

        # Unreachable: the loop either returns or raises on its last attempt.
        raise ModelUnavailable(task=task.value, model=model_id, transient=False)

    def _usage_for(
        self,
        response: Any,
        *,
        task: GeminiTask,
        model_id: str,
        prompt: PromptRef,
        attempts: int,
        started: float,
    ) -> ModelUsage:
        """Build the usage record, including cost at the configured prices."""
        metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = _count(getattr(metadata, "prompt_token_count", None))
        output_tokens = _count(getattr(metadata, "candidates_token_count", None))
        thought_tokens = _count(getattr(metadata, "thoughts_token_count", None))
        total_tokens = _count(getattr(metadata, "total_token_count", None))
        if not total_tokens:
            total_tokens = prompt_tokens + output_tokens + thought_tokens
        # Gemini bills thinking tokens as output, so they are priced as output.
        cost = self._settings.estimate_cost_usd(prompt_tokens, output_tokens + thought_tokens)
        reported = getattr(response, "model_version", None)
        resolved = reported if isinstance(reported, str) and reported else model_id
        return ModelUsage(
            model=resolved,
            prompt=prompt,
            source="live",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
            total_tokens=total_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            estimated_cost_usd=cost,
            truncated=False,
        )

    def _check_response(
        self, response: Any, *, task: GeminiTask, model_id: str, usage: ModelUsage
    ) -> None:
        """Reject a response that cannot be trusted even though HTTP succeeded.

        A safety block and a ``MAX_TOKENS`` truncation both arrive as a 200.
        Truncation is the dangerous one: the body often still parses, and the
        missing tail reads downstream as "the model did not mention it", which
        is indistinguishable from a real negative.
        """
        feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason is not None:
            raise ModelBlocked(
                task=task.value, model=model_id, reason=_enum_name(block_reason).lower()
            )
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise ModelInvalidOutput(task=task.value, model=model_id, reason="no_candidates")
        finish = _enum_name(getattr(candidates[0], "finish_reason", None))
        if finish in _BLOCKED_FINISH_REASONS:
            raise ModelBlocked(task=task.value, model=model_id, reason=finish.lower())
        if finish in _TRUNCATED_FINISH_REASONS:
            log.warning(
                "gemini.truncated",
                task=task.value,
                model=usage.model,
                tok_out=usage.output_tokens,
                attempts=usage.attempts,
            )
            raise ModelInvalidOutput(task=task.value, model=model_id, reason="max_tokens")

    # ------------------------------------------------------------------ logging
    def _log_call(self, task: GeminiTask, usage: ModelUsage, context: CallContext | None) -> None:
        log.info(
            "gemini.call",
            task=task.value,
            model=usage.model,
            prompt_name=usage.prompt.name if usage.prompt else None,
            prompt_version=usage.prompt.version if usage.prompt else None,
            source=usage.source,
            outcome="ok",
            attempts=usage.attempts,
            retries=usage.retries,
            latency_ms=usage.latency_ms,
            tok_in=usage.prompt_tokens,
            tok_out=usage.output_tokens,
            tok_thought=usage.thought_tokens,
            tok_total=usage.total_tokens,
            cost_usd=round(usage.estimated_cost_usd, 8),
            request_id=_request_id(context),
            run_id=context.run_id if context else None,
        )

    def _log_failure(
        self,
        task: GeminiTask,
        model_id: str,
        prompt: PromptRef,
        attempts: int,
        started: float,
        context: CallContext | None,
        failure: Exception,
    ) -> None:
        log.warning(
            "gemini.failed",
            task=task.value,
            model=model_id,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            attempts=attempts,
            latency_ms=int((time.monotonic() - started) * 1000),
            error=type(failure).__name__,
            status=getattr(failure, "status", None),
            reason=getattr(failure, "reason", None),
            transient=getattr(failure, "transient", None),
            validation=getattr(failure, "validation_summary", None),
            request_id=_request_id(context),
            run_id=context.run_id if context else None,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _count(value: Any) -> int:
    """Token counts are ``int | None`` on every SDK field. Coalesce to 0."""
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    return 0


def _request_id(context: CallContext | None) -> str | None:
    return context.request_id if context is not None else None


def _response_text(response: Any) -> str | None:
    """``response.text`` without the SDK's warning path for empty candidates."""
    try:
        text = response.text
    except (AttributeError, ValueError, IndexError):
        return None
    return cast("str | None", text)


def _enum_name(value: Any) -> str:
    """Name of an SDK enum member, or its string form. ``""`` for ``None``."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", maxsplit=1)[-1]


def _classify_api_error(
    error: genai_errors.APIError, *, task: str, model: str
) -> ModelRateLimited | ModelUnavailable | ModelInvalidOutput:
    """Map a provider error onto the taxonomy, deciding retryability.

    * 429 - rate limited, retryable, honours ``RetryInfo``.
    * 5xx, 408, 409, 529 - the provider is struggling, retryable.
    * 400 - the provider rejected *our* request, so our schema or prompt is
      wrong. Classified as invalid output and never retried.
    * every other 4xx - 401, 403 ``PERMISSION_DENIED``, 404 for a retired model
      - a refusal that will be identical next time, so not retried. This is
      what this project's key currently returns (``docs/TOOLCHAIN.md``).
    """
    status = int(getattr(error, "code", 0) or 0)
    if status == 429:
        delay = _retry_info_delay_s(error)
        return ModelRateLimited(
            task=task,
            model=model,
            retry_after_s=int(delay) if delay is not None else None,
            cause=error,
        )
    if status == 400:
        return ModelInvalidOutput(
            task=task, model=model, reason="provider_rejected_request", cause=error
        )
    transient = status == 0 or status >= 500 or status in _RETRYABLE_STATUS
    return ModelUnavailable(
        task=task, model=model, status=status or None, transient=transient, cause=error
    )


def _retry_after_s(failure: Exception) -> float | None:
    value = getattr(failure, "retry_after_s", None)
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def _retry_info_delay_s(error: genai_errors.APIError) -> float | None:
    """Read ``google.rpc.RetryInfo.retryDelay`` out of an error body.

    The provider sends it as a protobuf duration string (``"21s"``, ``"1.5s"``)
    nested somewhere inside ``error.details``. Anything unexpected is ignored
    rather than guessed at.
    """
    for candidate in (getattr(error, "details", None), getattr(error, "response_json", None)):
        delay = _find_retry_delay(candidate)
        if delay is not None:
            return delay
    return None


def _find_retry_delay(node: Any, depth: int = 0) -> float | None:
    if depth > 6:
        return None
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"retryDelay", "retry_delay", "retryAfter", "retry_after"}:
                parsed = _parse_duration_s(value)
                if parsed is not None:
                    return parsed
            found = _find_retry_delay(value, depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_retry_delay(item, depth + 1)
            if found is not None:
                return found
    return None


def _parse_duration_s(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str):
        try:
            seconds = float(value.strip().rstrip("s"))
        except ValueError:
            return None
        return seconds if seconds > 0 else None
    return None


def _backoff_delay(attempt: int, server_delay_s: float | None = None) -> float:
    """Exponential backoff with full jitter, floored by any server request.

    Full jitter - ``uniform(0, ceiling)`` rather than ``ceiling`` or
    ``ceiling/2 + jitter`` - because several explanation calls retry at once and
    the property worth having is that they stop colliding, not that each one
    waits a particular length of time.
    """
    ceiling = min(BACKOFF_BASE_S * (2**attempt), BACKOFF_MAX_S)
    delay = _JITTER.uniform(0.0, ceiling)
    if server_delay_s is not None:
        delay = min(max(delay, server_delay_s), RETRY_AFTER_MAX_S)
    return delay


# ---------------------------------------------------------------------------
# process-wide access
# ---------------------------------------------------------------------------
_singleton: GeminiClient | None = None
_singleton_lock = threading.Lock()
_override: GeminiPort | None = None


def get_gemini_client(settings: Settings | None = None) -> GeminiClient:
    """The process-wide client.

    One instance, because the SDK client holds an HTTP connection pool and one
    pool per process is the point of a pool.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = GeminiClient(settings or get_settings())
        return _singleton


def reset_gemini_client() -> None:
    """Drop the singleton. For tests and for a configuration reload."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def set_gemini_port_override(port: GeminiPort | None) -> None:
    """Install (or clear) a process-wide :class:`GeminiPort` implementation.

    The companion to ``app.dependency_overrides[provide_gemini_port]``: the
    FastAPI override covers request-scoped injection, and this covers the paths
    that are not requests - a LangGraph node, a benchmark, a script.
    """
    global _override
    _override = port


def provide_gemini_port() -> GeminiPort:
    """FastAPI dependency returning the active :class:`GeminiPort`.

    Override it in tests with::

        app.dependency_overrides[provide_gemini_port] = lambda: FakeGeminiClient(settings)
    """
    if _override is not None:
        return _override
    return get_gemini_client()
