"""The four model failure modes, side by side, asserted to be distinguishable.

``tests/ai/test_gemini_client.py`` asserts each failure in isolation. The claim
this file exists to defend is the one the README makes: a timeout, a 429, a
malformed body and a schema-violating body are *four different events*, and the
difference survives all the way to the error envelope the browser receives.

Why that matters, concretely:

* a timeout or a 429 is the provider's problem, so it is ``transient``, the
  retry loop backs off and tries again, and nothing is billed;
* a malformed or schema-violating body is a defect in this repository. The
  tokens were already spent, so the call *is* billed, but retrying buys the
  same answer at twice the cost and the loop must not do it. The two are kept
  apart by ``ModelInvalidOutput.reason``, because "the provider wrote prose
  instead of JSON" and "our schema and our prompt disagree" are fixed in
  different places.

``is_retryable`` is deliberately not the axis these assertions use. It is the
*wire* contract - may the browser retry this request - and it is ``True`` for
all five model errors. The question the retry loop asks is ``transient``.

Every case drives the real :class:`~app.ai.gemini.GeminiClient` against a
scripted SDK stub, so the assertions are on what the production retry loop,
classifier and decoder actually did - including how many times the provider was
called and how long the loop slept.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from google.genai import Client

from app.ai import gemini
from app.ai.errors import (
    ModelError,
    ModelInvalidOutput,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.prompts import extraction
from app.ai.prompts.base import PromptSpec
from app.ai.schemas import ExtractionResponse, GeminiTask
from app.core.errors import AppError, ErrorCode
from tests.ai.conftest import (
    SECRET_NARRATIVE,
    SleepRecorder,
    StubGenaiClient,
    client_error,
    make_response,
    make_settings,
)

VALID_EXTRACTION: dict[str, Any] = {
    "state": "NJ",
    "county": "Middlesex",
    "household_size": 3,
    "needs": ["FOOD"],
    "unknown_fields": [],
    "ambiguities": [],
}

#: Two retries, so a transient failure is attempted three times in total.
MAX_RETRIES = 2


def prompt() -> PromptSpec:
    return extraction.build(SECRET_NARRATIVE)


async def never_answers() -> Any:
    await asyncio.sleep(30)
    raise AssertionError("the timeout should have fired first")


@dataclass(frozen=True, slots=True)
class Mode:
    """One failure mode and everything that must stay true about it."""

    name: str
    script: tuple[Any, ...]
    exception: type[ModelError]
    status_code: int
    code: ErrorCode
    reason: str | None
    attempts: int
    transient: bool
    billed_calls: int
    timeout_s: float = 5.0

    def client(self, sleep: SleepRecorder) -> tuple[gemini.GeminiClient, StubGenaiClient]:
        settings = make_settings(gemini_max_retries=MAX_RETRIES, gemini_timeout_s=self.timeout_s)
        stub = StubGenaiClient(*self.script)
        return gemini.GeminiClient(settings, cast(Client, stub), sleep=sleep), stub


MODES: tuple[Mode, ...] = (
    Mode(
        name="timeout",
        script=(never_answers,),
        exception=ModelTimeout,
        status_code=504,
        code=ErrorCode.UPSTREAM_TIMEOUT,
        reason=None,
        attempts=MAX_RETRIES + 1,
        transient=True,
        billed_calls=0,
        timeout_s=0.02,
    ),
    Mode(
        name="rate_limited",
        script=(client_error(429),),
        exception=ModelRateLimited,
        status_code=429,
        code=ErrorCode.RATE_LIMITED,
        reason=None,
        attempts=MAX_RETRIES + 1,
        transient=True,
        billed_calls=0,
    ),
    Mode(
        name="malformed_json",
        script=(make_response("Sure! Here is the profile: {state: NJ, household"),),
        exception=ModelInvalidOutput,
        status_code=502,
        code=ErrorCode.INVALID_MODEL_OUTPUT,
        reason="malformed_json",
        attempts=1,
        transient=False,
        billed_calls=1,
    ),
    Mode(
        name="schema_violation",
        script=(make_response(json.dumps({"state": "NJ", "household_size": "three people"})),),
        exception=ModelInvalidOutput,
        status_code=502,
        code=ErrorCode.INVALID_MODEL_OUTPUT,
        reason="schema_violation",
        attempts=1,
        transient=False,
        billed_calls=1,
    ),
)

BY_NAME = {mode.name: mode for mode in MODES}
IDS = [mode.name for mode in MODES]


async def run(mode: Mode, sleeper: SleepRecorder) -> tuple[AppError, StubGenaiClient]:
    """Drive one mode and hand back the error it raised.

    Typed as :class:`~app.core.errors.AppError` because that is the half of the
    taxonomy the API envelope is built from; ``ModelError`` is only the marker
    mixin and carries no attributes of its own.
    """
    client, stub = mode.client(sleeper)
    with pytest.raises(mode.exception) as caught:
        await client.structured(GeminiTask.EXTRACT_PROFILE, prompt(), ExtractionResponse)
    return cast(AppError, caught.value), stub


# ---------------------------------------------------------------------------
# each mode, on its own terms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES, ids=IDS)
async def test_the_failure_reaches_the_caller_as_its_own_error(mode: Mode) -> None:
    error, _stub = await run(mode, SleepRecorder())

    assert isinstance(error, mode.exception)
    assert error.status_code == mode.status_code
    assert error.code is mode.code
    assert getattr(error, "transient") is mode.transient  # noqa: B009
    assert getattr(error, "reason", None) == mode.reason
    assert error.is_retryable is True, "every model failure is retryable on the wire"


@pytest.mark.parametrize("mode", MODES, ids=IDS)
async def test_only_a_provider_side_failure_is_retried(mode: Mode) -> None:
    """Retrying our own defect costs money and changes nothing."""
    sleeper = SleepRecorder()
    _error, stub = await run(mode, sleeper)

    assert len(stub.calls) == mode.attempts
    assert len(sleeper.delays) == mode.attempts - 1, "there is no sleep after the last attempt"


@pytest.mark.parametrize("mode", MODES, ids=IDS)
async def test_the_message_is_renderable_and_carries_no_content(mode: Mode) -> None:
    error, _stub = await run(mode, SleepRecorder())

    assert error.message, "every failure needs a sentence a user can read"
    rendered = json.dumps(error.to_response(request_id="r" * 8).model_dump(mode="json"))
    assert SECRET_NARRATIVE not in rendered
    assert "zzqq" not in rendered
    assert "three people" not in rendered, "a rejected value must not be echoed back"


@pytest.mark.parametrize("mode", MODES, ids=IDS)
async def test_cost_accounting_records_exactly_the_tokens_that_were_spent(mode: Mode) -> None:
    """A failure that never produced tokens is free; one that did is not.

    Under-reporting the second kind would make the cost benchmark lie in the
    one direction that matters, because an unusable answer is still paid for.
    """
    client, stub = mode.client(SleepRecorder())
    with pytest.raises(mode.exception):
        await client.structured(GeminiTask.EXTRACT_PROFILE, prompt(), ExtractionResponse)

    totals = client.usage.totals
    assert totals.calls == mode.billed_calls
    if mode.billed_calls == 0:
        assert totals.estimated_cost_usd == 0.0
        assert totals.total_tokens == 0
    else:
        assert totals.estimated_cost_usd > 0.0
        assert totals.total_tokens > 0
    assert stub.calls, "the provider was called, so the stub really ran"


# ---------------------------------------------------------------------------
# and distinguishable from each other
# ---------------------------------------------------------------------------
async def test_the_four_modes_are_pairwise_distinguishable() -> None:
    """The whole point: four failures, four reactions, no collapsing."""
    seen: list[tuple[str, int, str | None, bool]] = []
    for mode in MODES:
        error, _stub = await run(mode, SleepRecorder())
        seen.append(
            (
                type(error).__name__,
                error.status_code,
                getattr(error, "reason", None),
                getattr(error, "transient"),  # noqa: B009
            )
        )

    assert len(set(seen)) == len(MODES), f"two failure modes look identical: {seen}"
    assert len({entry[1] for entry in seen}) == 3, (
        "502 is shared by the two output failures on purpose; the other two are distinct"
    )
    assert {entry[0] for entry in seen} == {
        "ModelTimeout",
        "ModelRateLimited",
        "ModelInvalidOutput",
    }


async def test_the_two_bad_bodies_are_told_apart_by_reason() -> None:
    """Both are 502s, so ``reason`` is the only thing that separates them."""
    malformed, _ = await run(BY_NAME["malformed_json"], SleepRecorder())
    violating, _ = await run(BY_NAME["schema_violation"], SleepRecorder())

    assert type(malformed) is type(violating)
    assert malformed.status_code == violating.status_code == 502
    assert isinstance(malformed, ModelInvalidOutput)
    assert isinstance(violating, ModelInvalidOutput)
    assert malformed.reason == "malformed_json"
    assert violating.reason == "schema_violation"
    assert malformed.validation_summary == []
    assert violating.validation_summary == ["household_size:int_parsing"]


async def test_a_rate_limit_backs_off_before_each_retry() -> None:
    sleeper = SleepRecorder()
    _error, stub = await run(BY_NAME["rate_limited"], sleeper)

    assert len(stub.calls) == MAX_RETRIES + 1
    assert len(sleeper.delays) == MAX_RETRIES
    for attempt, delay in enumerate(sleeper.delays):
        ceiling = min(gemini.BACKOFF_BASE_S * (2**attempt), gemini.BACKOFF_MAX_S)
        assert 0.0 <= delay <= ceiling, f"attempt {attempt} slept {delay}s"
    assert sum(sleeper.delays) > 0.0, "a 429 that is retried immediately is not a backoff"


async def test_a_retry_after_header_overrides_the_backoff_floor() -> None:
    """The provider's own advice wins over our curve, up to a sane ceiling."""
    sleeper = SleepRecorder()
    settings = make_settings(gemini_max_retries=1)
    stub = StubGenaiClient(
        client_error(429, retry_delay="4s"), make_response(json.dumps(VALID_EXTRACTION))
    )
    client = gemini.GeminiClient(settings, cast(Client, stub), sleep=sleeper)

    result = await client.structured(GeminiTask.EXTRACT_PROFILE, prompt(), ExtractionResponse)

    assert sleeper.delays[0] >= 4.0
    assert result.usage.retries == 1


async def test_a_transient_refusal_and_a_permanent_one_are_not_the_same_503() -> None:
    """``transient`` is what the retry loop asks; the wire status is the same."""
    sleeper = SleepRecorder()
    settings = make_settings(gemini_max_retries=1)

    server = StubGenaiClient(client_error(503))
    with pytest.raises(ModelUnavailable) as transient:
        await gemini.GeminiClient(settings, cast(Client, server), sleep=sleeper).structured(
            GeminiTask.EXTRACT_PROFILE, prompt(), ExtractionResponse
        )
    assert transient.value.transient is True
    assert len(server.calls) == 2

    denied = StubGenaiClient(client_error(403))
    with pytest.raises(ModelUnavailable) as permanent:
        await gemini.GeminiClient(settings, cast(Client, denied), sleep=sleeper).structured(
            GeminiTask.EXTRACT_PROFILE, prompt(), ExtractionResponse
        )
    assert permanent.value.transient is False
    assert len(denied.calls) == 1, "a permission failure is not a traffic jam"


async def test_every_mode_is_caught_by_the_shared_marker() -> None:
    """``except ModelError`` is what the workflow nodes rely on to degrade."""
    for mode in MODES:
        client, _stub = mode.client(SleepRecorder())
        with pytest.raises(ModelError):
            await client.structured(GeminiTask.EXTRACT_PROFILE, prompt(), ExtractionResponse)


def test_the_matrix_covers_the_modes_the_readme_claims() -> None:
    assert {mode.name for mode in MODES} == {
        "timeout",
        "rate_limited",
        "malformed_json",
        "schema_violation",
    }
    assert len({mode.exception for mode in MODES}) == 3
    assert {mode.reason for mode in MODES} == {None, "malformed_json", "schema_violation"}
