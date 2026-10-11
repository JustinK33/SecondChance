"""Tests for the live client's transport behaviour.

Nothing here reaches the network: :class:`StubGenaiClient` is scripted with
responses and real ``google.genai`` errors, so the assertions are about what our
retry loop, classifier, usage accounting and logging do with them.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from google.genai import Client, types

from app.ai import gemini
from app.ai.errors import (
    ModelBlocked,
    ModelInvalidOutput,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.prompts import extraction
from app.ai.prompts.base import PromptSpec
from app.ai.schemas import ExtractionResponse, GeminiTask
from app.core.config import Settings
from app.schemas.contracts import GeminiPort
from tests.ai.conftest import (
    SECRET_NARRATIVE,
    StubGenaiClient,
    client_error,
    make_response,
    make_settings,
    make_usage_metadata,
)

VALID_EXTRACTION = {
    "state": "NJ",
    "county": "Middlesex",
    "household_size": 3,
    "income_min_cents": 190000,
    "income_max_cents": 210000,
    "income_period": "MONTHLY",
    "needs": ["FOOD"],
    "unknown_fields": ["age"],
    "ambiguities": [],
}


def extraction_prompt() -> PromptSpec:
    return extraction.build(SECRET_NARRATIVE)


def make_client(settings: Settings, stub: StubGenaiClient, **kwargs: Any) -> gemini.GeminiClient:
    """A real client with the scripted stub standing in for the SDK's.

    The cast is the whole substitution: ``StubGenaiClient`` implements only
    ``aio.models.generate_content``, which is all the production code touches,
    and declaring that as a ``genai.Client`` keeps the call sites readable.
    """
    return gemini.GeminiClient(settings, cast(Client, stub), **kwargs)


async def test_structured_output_validates_and_returns_usage(settings, sleeper):
    stub = StubGenaiClient(
        make_response(json.dumps(VALID_EXTRACTION), model_version="gemini-test-cheap-001")
    )
    client = make_client(settings, stub, sleep=sleeper)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert isinstance(result.value, ExtractionResponse)
    assert result.value.state == "NJ"
    assert result.value.household_size == 3
    assert result.usage.source == "live"
    assert result.usage.model == "gemini-test-cheap-001"
    assert result.usage.prompt_tokens == 1200
    assert result.usage.output_tokens == 240
    assert result.usage.total_tokens == 1440
    assert result.usage.attempts == 1
    assert result.usage.retries == 0
    assert result.usage.latency_ms >= 0
    assert result.usage.prompt is not None
    assert (result.usage.prompt.name, result.usage.prompt.version) == (
        extraction.NAME,
        extraction.VERSION,
    )
    assert sleeper.delays == []
    assert client.usage.totals.calls == 1


async def test_structured_prefers_sdk_parsed_object(settings):
    """``response.parsed`` is already validated, so it is used as-is."""
    parsed = ExtractionResponse(state="NJ", household_size=2)
    stub = StubGenaiClient(make_response("ignored, parsed wins", parsed=parsed))
    client = make_client(settings, stub)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert result.value is parsed
    assert result.notes == []


async def test_rate_limited_then_success_retries_with_full_jitter(settings, sleeper):
    stub = StubGenaiClient(client_error(429), make_response(json.dumps(VALID_EXTRACTION)))
    client = make_client(settings, stub, sleep=sleeper)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert result.usage.attempts == 2
    assert result.usage.retries == 1
    assert len(stub.calls) == 2
    assert len(sleeper.delays) == 1
    # Full jitter: uniform(0, base * 2**attempt) for attempt 0.
    assert 0.0 <= sleeper.delays[0] <= gemini.BACKOFF_BASE_S


async def test_three_consecutive_rate_limits_raise_and_stop(sleeper):
    settings = make_settings(gemini_max_retries=2)
    stub = StubGenaiClient(client_error(429))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelRateLimited) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert len(stub.calls) == 3, "max_retries=2 means three attempts, not four"
    assert len(sleeper.delays) == 2, "no sleep after the final attempt"
    assert caught.value.status_code == 429
    assert caught.value.code.value == "rate_limited"
    assert caught.value.is_retryable is True
    assert client.usage.totals.calls == 0, "a failed call records no usage"


async def test_backoff_delays_grow_and_stay_within_bounds(sleeper):
    settings = make_settings(gemini_max_retries=3)
    stub = StubGenaiClient(client_error(503))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelUnavailable):
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert len(sleeper.delays) == 3
    for attempt, delay in enumerate(sleeper.delays):
        ceiling = min(gemini.BACKOFF_BASE_S * (2**attempt), gemini.BACKOFF_MAX_S)
        assert 0.0 <= delay <= ceiling


def test_backoff_delay_is_jittered_and_bounded():
    """The property, sampled directly: full jitter, a ceiling, and variety."""
    samples = [gemini._backoff_delay(2) for _ in range(200)]
    ceiling = min(gemini.BACKOFF_BASE_S * 4, gemini.BACKOFF_MAX_S)
    assert all(0.0 <= value <= ceiling for value in samples)
    assert len(set(samples)) > 100, "full jitter must not be a constant"
    assert min(samples) < ceiling / 2, "the low half of the window must be reachable"
    high = [gemini._backoff_delay(20) for _ in range(20)]
    assert all(value <= gemini.BACKOFF_MAX_S for value in high)


def test_server_retry_delay_is_a_floor_and_is_capped():
    assert gemini._backoff_delay(0, 2.0) >= 2.0
    assert gemini._backoff_delay(0, 10_000.0) == gemini.RETRY_AFTER_MAX_S


async def test_server_retry_delay_is_honoured(sleeper):
    settings = make_settings(gemini_max_retries=1)
    stub = StubGenaiClient(
        client_error(429, retry_delay="3s"), make_response(json.dumps(VALID_EXTRACTION))
    )
    client = make_client(settings, stub, sleep=sleeper)

    await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert sleeper.delays[0] >= 3.0


async def test_timeout_raises_model_timeout(sleeper):
    settings = make_settings(gemini_timeout_s=0.05, gemini_max_retries=0)

    async def never_answers():
        await asyncio.sleep(5)
        raise AssertionError("unreachable")

    stub = StubGenaiClient(never_answers)
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelTimeout) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.status_code == 504
    assert caught.value.timeout_s == 0.05
    assert caught.value.detail == {
        "task": "extract_profile",
        "model": "gemini-test-cheap",
        "timeout_s": 0.05,
    }
    assert sleeper.delays == [], "retries are exhausted, so nothing is slept on"


async def test_timeout_is_retried_when_budget_allows(sleeper):
    settings = make_settings(gemini_timeout_s=0.05, gemini_max_retries=1)

    async def slow_then_fast():
        await asyncio.sleep(5)

    stub = StubGenaiClient(slow_then_fast, make_response(json.dumps(VALID_EXTRACTION)))
    client = make_client(settings, stub, sleep=sleeper)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert result.usage.attempts == 2
    assert len(sleeper.delays) == 1


async def test_malformed_json_raises_invalid_output_and_is_not_retried(settings, sleeper):
    stub = StubGenaiClient(make_response("here you go: {state: NJ"))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelInvalidOutput) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.reason == "malformed_json"
    assert caught.value.status_code == 502
    assert len(stub.calls) == 1, "a bad body is our defect; retrying buys the same answer"
    assert sleeper.delays == []


async def test_schema_violation_is_not_retried_and_logs_no_values(settings, sleeper):
    body = json.dumps({"state": "NJ", "household_size": "three people"})
    stub = StubGenaiClient(make_response(body))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelInvalidOutput) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.reason == "schema_violation"
    assert caught.value.validation_summary == ["household_size:int_parsing"]
    assert "three people" not in str(caught.value.validation_summary)
    assert len(stub.calls) == 1


async def test_empty_body_raises_invalid_output(settings):
    stub = StubGenaiClient(make_response(None))
    client = make_client(settings, stub)

    with pytest.raises(ModelInvalidOutput) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.reason == "empty_response"


async def test_fenced_json_is_recovered_with_a_note(settings):
    body = f"```json\n{json.dumps(VALID_EXTRACTION)}\n```"
    stub = StubGenaiClient(make_response(body))
    client = make_client(settings, stub)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert result.notes == ["stripped_code_fence"]
    assert result.value.state == "NJ"


async def test_truncated_response_is_refused(settings):
    stub = StubGenaiClient(
        make_response(json.dumps(VALID_EXTRACTION), finish_reason=types.FinishReason.MAX_TOKENS)
    )
    client = make_client(settings, stub)

    with pytest.raises(ModelInvalidOutput) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.reason == "max_tokens"


async def test_safety_block_raises_model_blocked(settings):
    stub = StubGenaiClient(
        make_response(None, finish_reason=types.FinishReason.SAFETY, candidates=None)
    )
    client = make_client(settings, stub)

    with pytest.raises(ModelBlocked) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.reason == "safety"
    assert caught.value.transient is False


async def test_prompt_feedback_block_raises_model_blocked(settings):
    stub = StubGenaiClient(make_response(None, block_reason="PROHIBITED_CONTENT"))
    client = make_client(settings, stub)

    with pytest.raises(ModelBlocked):
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)


@pytest.mark.parametrize(
    ("status", "expected", "transient"),
    [
        (500, ModelUnavailable, True),
        (503, ModelUnavailable, True),
        (403, ModelUnavailable, False),
        (404, ModelUnavailable, False),
        (401, ModelUnavailable, False),
        (400, ModelInvalidOutput, False),
    ],
)
async def test_error_classification(status, expected, transient, sleeper):
    settings = make_settings(gemini_max_retries=0)
    stub = StubGenaiClient(client_error(status))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(expected) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert caught.value.transient is transient


async def test_non_transient_refusal_is_never_retried(sleeper):
    """403 PERMISSION_DENIED is what this project's key returns today."""
    settings = make_settings(gemini_max_retries=3)
    stub = StubGenaiClient(client_error(403))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelUnavailable) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    assert len(stub.calls) == 1
    assert sleeper.delays == []
    assert caught.value.status == 403


async def test_transport_error_is_retried(sleeper):
    import httpx

    settings = make_settings(gemini_max_retries=1)
    stub = StubGenaiClient(
        httpx.ConnectError("connection refused"), make_response(json.dumps(VALID_EXTRACTION))
    )
    client = make_client(settings, stub, sleep=sleeper)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert result.usage.attempts == 2
    assert len(sleeper.delays) == 1


async def test_httpx_timeout_is_classified_as_model_timeout(sleeper):
    import httpx

    settings = make_settings(gemini_max_retries=0)
    stub = StubGenaiClient(httpx.ReadTimeout("too slow"))
    client = make_client(settings, stub, sleep=sleeper)

    with pytest.raises(ModelTimeout):
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)


# --------------------------------------------------------------------- cost
async def test_cost_arithmetic_matches_configured_prices():
    settings = make_settings(gemini_price_input_per_mtok=0.10, gemini_price_output_per_mtok=0.40)
    usage = make_usage_metadata(
        prompt_tokens=1_000_000, output_tokens=500_000, thought_tokens=100_000
    )
    stub = StubGenaiClient(make_response(json.dumps(VALID_EXTRACTION), usage=usage))
    client = make_client(settings, stub)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    # 1M input at $0.10 + 600k billable output (thinking is billed as output)
    # at $0.40 = 0.10 + 0.24.
    assert result.usage.estimated_cost_usd == pytest.approx(0.34)
    assert result.usage.thought_tokens == 100_000
    assert result.usage.billable_output_tokens == 600_000
    assert result.usage.estimated_cost_usd == pytest.approx(
        settings.estimate_cost_usd(1_000_000, 600_000)
    )


async def test_usage_accumulator_sums_across_calls(settings):
    stub = StubGenaiClient(make_response(json.dumps(VALID_EXTRACTION)))
    client = make_client(settings, stub)

    for _ in range(3):
        await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    totals = client.usage.totals
    assert totals.calls == 3
    assert totals.prompt_tokens == 3600
    assert totals.output_tokens == 720
    assert totals.estimated_cost_usd == pytest.approx(3 * (1200 * 0.1 + 240 * 0.4) / 1e6)
    client.usage.reset()
    assert client.usage.totals.calls == 0


async def test_missing_total_token_count_is_derived(settings):
    usage = make_usage_metadata(prompt_tokens=10, output_tokens=5, total_tokens=0)
    stub = StubGenaiClient(make_response(json.dumps(VALID_EXTRACTION), usage=usage))
    client = make_client(settings, stub)

    result = await client.structured(
        GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse
    )

    assert result.usage.total_tokens == 15


# ------------------------------------------------------------------ logging
async def test_logs_contain_telemetry_but_no_prompt_or_profile_content(settings, log_stream):
    """The privacy assertion, made against the real logging pipeline."""
    stub = StubGenaiClient(make_response(json.dumps(VALID_EXTRACTION)))
    client = make_client(settings, stub)
    await client.extract_profile(SECRET_NARRATIVE)

    output = log_stream.getvalue()
    assert "gemini.call" in output
    lines = [json.loads(line) for line in output.strip().splitlines()]
    call = next(line for line in lines if line["event"] == "gemini.call")
    assert call["task"] == "extract_profile"
    assert call["model"] == "gemini-test-cheap"
    assert call["prompt_name"] == "extract_profile"
    assert call["tok_in"] == 1200
    assert call["tok_out"] == 240
    assert call["source"] == "live"

    # Nothing from the narrative, the prompt or the response body.
    assert "zzqq-private-narrative-marker" not in output
    assert "niece" not in output
    assert "1850" not in output
    assert "Middlesex" not in output
    assert "ignore previous instructions" not in output
    assert "UNTRUSTED_DATA" not in output
    for forbidden in ("prompt_tokens", "narrative", "contents"):
        assert forbidden not in call


async def test_retry_and_failure_log_lines_carry_no_content(settings, sleeper, log_stream):
    stub = StubGenaiClient(client_error(429))
    client = make_client(settings, stub, sleep=sleeper)
    with pytest.raises(ModelRateLimited):
        await client.extract_profile(SECRET_NARRATIVE)

    output = log_stream.getvalue()
    events = [json.loads(line)["event"] for line in output.strip().splitlines()]
    assert "gemini.retry" in events
    assert "gemini.failed" in events
    assert "zzqq-private-narrative-marker" not in output


# --------------------------------------------------------- wiring and config
def test_per_task_model_selection(settings):
    client = gemini.GeminiClient(settings)
    assert client.model_for(GeminiTask.EXTRACT_PROFILE) == "gemini-test-cheap"
    assert client.model_for(GeminiTask.PROPOSE_CLARIFICATIONS) == "gemini-test-cheap"
    assert client.model_for(GeminiTask.EXPLAIN_MATCH) == "gemini-test-strong"
    assert client.model_for(GeminiTask.DRAFT_ACTION_PLAN) == "gemini-test-strong"
    assert client.model_for(GeminiTask.ANSWER_QUESTION) == "gemini-test-chat"


def test_per_task_models_fall_back_to_the_base_model():
    settings = make_settings(
        gemini_model="only-one-model",
        gemini_model_extraction=None,
        gemini_model_explanation=None,
        gemini_model_chat=None,
    )
    client = gemini.GeminiClient(settings)
    assert {client.model_for(task) for task in GeminiTask} == {"only-one-model"}


async def test_structured_sends_schema_timeout_and_system_instruction(settings):
    stub = StubGenaiClient(make_response(json.dumps(VALID_EXTRACTION)))
    client = make_client(settings, stub)

    await client.structured(GeminiTask.EXTRACT_PROFILE, extraction_prompt(), ExtractionResponse)

    config = stub.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is ExtractionResponse
    assert config.temperature == 0.0
    assert config.http_options.timeout == settings.gemini_timeout_ms == 5000
    assert "ABSOLUTE RULES" in config.system_instruction
    assert stub.calls[0]["model"] == "gemini-test-cheap"


async def test_text_primitive_returns_text_and_usage(settings):
    stub = StubGenaiClient(make_response("  a plain answer  "))
    client = make_client(settings, stub)

    result = await client.text(GeminiTask.ANSWER_QUESTION, extraction_prompt())

    assert result.text == "a plain answer"
    assert result.usage.source == "live"
    assert stub.calls[0]["config"].response_schema is None


async def test_text_primitive_rejects_an_empty_answer(settings):
    stub = StubGenaiClient(make_response("   "))
    client = make_client(settings, stub)

    with pytest.raises(ModelInvalidOutput):
        await client.text(GeminiTask.ANSWER_QUESTION, extraction_prompt())


def test_client_without_an_api_key_refuses_to_build_an_sdk_client():
    settings = make_settings(gemini_api_key="")
    client = gemini.GeminiClient(settings)
    assert settings.gemini_enabled is False
    with pytest.raises(ModelUnavailable) as caught:
        client._genai()
    assert caught.value.transient is False


def test_singleton_is_shared_and_resettable(settings):
    gemini.reset_gemini_client()
    first = gemini.get_gemini_client(settings)
    assert gemini.get_gemini_client(settings) is first
    gemini.reset_gemini_client()
    assert gemini.get_gemini_client(settings) is not first
    gemini.reset_gemini_client()


def test_provider_honours_an_override(settings):
    from app.ai.fake import FakeGeminiClient

    fake = FakeGeminiClient(settings)
    gemini.reset_gemini_client()
    try:
        gemini.set_gemini_port_override(fake)
        assert gemini.provide_gemini_port() is fake
    finally:
        gemini.set_gemini_port_override(None)
        gemini.reset_gemini_client()
    assert isinstance(gemini.provide_gemini_port(), gemini.GeminiClient)
    gemini.reset_gemini_client()


def test_both_implementations_satisfy_the_port(settings):
    from app.ai.fake import FakeGeminiClient

    assert isinstance(gemini.GeminiClient(settings), GeminiPort)
    assert isinstance(FakeGeminiClient(settings), GeminiPort)
