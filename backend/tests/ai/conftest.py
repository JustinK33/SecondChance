"""Fixtures for the AI layer: a stand-in SDK client and a built settings object.

The stub client mimics only the surface the production code actually touches -
``client.aio.models.generate_content`` - and is scripted with a list of
responses or exceptions. That keeps the retry tests honest: the assertion is on
how many times the real code path called the provider, not on a mock's recorded
intentions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import Settings

#: A narrative that must never appear in a log line. Distinctive enough that a
#: substring search for it is a meaningful assertion.
SECRET_NARRATIVE = "zzqq-private-narrative-marker my rent is 1850 dollars and I live with my niece"


def make_settings(**overrides: Any) -> Settings:
    """A fully specified local Settings, independent of any ``.env`` file.

    ``_env_file=None`` is what makes this deterministic: the repository's real
    ``backend/.env`` carries a live API key and real model ids, and a unit test
    that silently depended on them would pass or fail for the wrong reasons.
    """
    base: dict[str, Any] = {
        "app_env": "local",
        "auth_mode": "dev",
        "log_level": "INFO",
        "gemini_api_key": "test-key-not-a-real-credential",
        "gemini_model": "gemini-test-base",
        "gemini_model_extraction": "gemini-test-cheap",
        "gemini_model_explanation": "gemini-test-strong",
        "gemini_model_chat": "gemini-test-chat",
        "gemini_timeout_s": 5.0,
        "gemini_max_retries": 2,
        "gemini_price_input_per_mtok": 0.10,
        "gemini_price_output_per_mtok": 0.40,
        "workflow_timeout_s": 120.0,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def make_usage_metadata(
    *,
    prompt_tokens: int = 1200,
    output_tokens: int = 240,
    thought_tokens: int = 0,
    total_tokens: int | None = None,
) -> SimpleNamespace:
    """The subset of ``usage_metadata`` the client reads."""
    return SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        thoughts_token_count=thought_tokens,
        total_token_count=(
            total_tokens
            if total_tokens is not None
            else prompt_tokens + output_tokens + thought_tokens
        ),
    )


def make_response(
    text: str | None = None,
    *,
    parsed: Any = None,
    finish_reason: Any = types.FinishReason.STOP,
    block_reason: Any = None,
    usage: SimpleNamespace | None = None,
    model_version: str | None = None,
    candidates: list[Any] | None = None,
) -> SimpleNamespace:
    """A stand-in ``GenerateContentResponse``.

    Deliberately a namespace rather than a real ``types.GenerateContentResponse``:
    constructing the real one requires a full candidate tree, and the client
    reads these attributes defensively precisely so that a partial response from
    a future SDK version does not crash it.
    """
    return SimpleNamespace(
        text=text,
        parsed=parsed,
        candidates=(
            candidates if candidates is not None else [SimpleNamespace(finish_reason=finish_reason)]
        ),
        prompt_feedback=SimpleNamespace(block_reason=block_reason),
        usage_metadata=usage if usage is not None else make_usage_metadata(),
        model_version=model_version,
    )


def client_error(status: int, *, retry_delay: str | None = None) -> genai_errors.APIError:
    """A real ``google.genai`` error, built the way the SDK builds one."""
    body: dict[str, Any] = {"error": {"code": status, "message": "test", "status": "TEST"}}
    if retry_delay is not None:
        body["error"]["details"] = [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        ]
    if status >= 500:
        return genai_errors.ServerError(status, body)
    return genai_errors.ClientError(status, body)


@dataclass
class StubModels:
    """Scripted ``aio.models`` stand-in.

    ``script`` entries are returned in order; an ``Exception`` entry is raised
    instead. The last entry repeats, so a test that wants "always 429" supplies
    one error rather than counting attempts.
    """

    script: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        index = min(len(self.calls) - 1, len(self.script) - 1)
        item = self.script[index]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return await item()
        return item


class StubGenaiClient:
    """Just enough of ``genai.Client`` for :class:`app.ai.gemini.GeminiClient`."""

    def __init__(self, *script: Any) -> None:
        self.models = StubModels(list(script))
        self.aio = SimpleNamespace(models=self.models)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.models.calls


class SleepRecorder:
    """Replacement for ``asyncio.sleep`` that records instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture
def settings() -> Settings:
    """Overrides the session-scoped ``settings`` fixture from the root conftest.

    The root one reads the real ``.env``; everything under ``tests/ai`` wants a
    hermetic one instead.
    """
    return make_settings()


@pytest.fixture
def sleeper() -> SleepRecorder:
    return SleepRecorder()


@pytest.fixture
def log_stream(settings, monkeypatch):
    """The real logging pipeline, writing into a buffer this test can read.

    Two wrinkles, both from ``cache_logger_on_first_use=True`` in
    ``configure_logging``: a module-level ``get_logger`` proxy binds its sink the
    first time it is used, so reconfiguring the pipeline afterwards does not move
    it. Each logger under test is therefore replaced with a fresh proxy, and the
    global configuration is reset on the way out so later tests see the default
    pipeline.
    """
    import io

    import structlog

    from app.ai import base as ai_base
    from app.ai import gemini as ai_gemini
    from app.core.logging import configure_logging

    stream = io.StringIO()
    configure_logging(settings, stream=stream)
    monkeypatch.setattr(ai_gemini, "log", structlog.get_logger("app.ai.gemini"))
    monkeypatch.setattr(ai_base, "log", structlog.get_logger("app.ai.base"))
    try:
        yield stream
    finally:
        structlog.reset_defaults()
