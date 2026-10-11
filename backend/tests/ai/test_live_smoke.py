"""Live smoke tests against the real Gemini API. Opt-in, and deliberately tiny.

Run with::

    RUN_LIVE_GEMINI=1 .venv/bin/python -m pytest tests/ai/test_live_smoke.py -q -s -m live

Everything here is skipped unless ``RUN_LIVE_GEMINI=1``, so CI never spends
money or depends on a sampled model. One call per prompt path, ``max_retries=0``,
and the shortest realistic inputs: five requests total.

Account-side refusals
---------------------
``docs/TOOLCHAIN.md`` records that this project's API key cannot currently reach
any model - the 2.5 generation is retired for new projects (404) and the 3.x
generation returns 403 ``PERMISSION_DENIED``. That is an entitlement problem,
not a code problem, so these tests distinguish the two:

* a documented account-side refusal (403/404, surfaced by the client as
  :class:`~app.ai.errors.ModelUnavailable` with ``transient=False``) **skips**
  with the status in the reason, and :func:`test_live_api_key_is_valid` still
  proves the key and the transport work;
* anything else - a timeout, a 5xx, a schema violation - **fails**, because that
  would be ours.

No secret is ever printed. The key is read through ``Settings`` and only its
presence is ever asserted on.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from app.ai.errors import ModelRateLimited, ModelUnavailable
from app.ai.gemini import GeminiClient
from app.ai.prompts import action_plan, chat, clarification, explanation, extraction
from app.ai.schemas import (
    MODEL_RESPONSE_SCHEMAS,
    ActionPlanResponse,
    ChatAnswerResponse,
    ClarificationResponse,
    ExplanationResponse,
    ExtractionResponse,
    GeminiTask,
)
from app.core.config import Settings
from app.schemas import ChatRequest, RetrievedProgram
from tests.conftest import make_evaluation, make_match, make_profile, make_program, make_ruleset

pytestmark = pytest.mark.live

#: Statuses that mean "this project is not entitled to this model". See the
#: module docstring.
ACCOUNT_SIDE_STATUSES = frozenset({401, 403, 404})

#: Fictional narrative, short on purpose: this is a smoke test, not an eval.
NARRATIVE = (
    "I live in New Brunswick, New Jersey with my two kids. I work part time and "
    "make about $2,000 a month. I need help with groceries and my electric bill."
)


def requires_live() -> None:
    if os.environ.get("RUN_LIVE_GEMINI") != "1":
        pytest.skip("live Gemini tests are opt-in; set RUN_LIVE_GEMINI=1")


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    """The real configuration, with retries off so one test is one call."""
    requires_live()
    settings = Settings()
    if not settings.gemini_enabled:
        pytest.skip("GEMINI_API_KEY is empty, so no live call is possible")
    return settings.model_copy(update={"gemini_max_retries": 0})


@pytest.fixture
def live_client(live_settings: Settings) -> GeminiClient:
    """Function-scoped on purpose.

    ``asyncio_default_fixture_loop_scope`` is ``function``, and the SDK's async
    client caches an ``httpx.AsyncClient`` bound to the loop it first ran on, so
    a module-scoped client makes the second live test fail with
    ``RuntimeError: Event loop is closed``.
    """
    return GeminiClient(live_settings)


def report(task: GeminiTask, model: str, usage: Any, elapsed_ms: int) -> None:
    """Print the measurement. ``-s`` shows it; nothing sensitive is in it."""
    print(
        f"\n[live] {task.value}: model={model} latency={elapsed_ms}ms "
        f"tok_in={usage.prompt_tokens} tok_out={usage.output_tokens} "
        f"tok_thought={usage.thought_tokens} tok_total={usage.total_tokens} "
        f"cost_usd={usage.estimated_cost_usd:.6f} attempts={usage.attempts}"
    )


def _report_refusal(task: GeminiTask, client: GeminiClient, status: str, started: float) -> None:
    """Print the round-trip time of a refusal.

    A refused call still measures the transport, and reporting it is the
    difference between "we never tried" and "we tried and the provider said no
    in 340ms".
    """
    print(
        f"\n[live] {task.value}: model={client.model_for(task)} "
        f"refused with {status} after {int((time.monotonic() - started) * 1000)}ms"
    )


async def run_once(client: GeminiClient, task: GeminiTask, spec: Any, schema: type) -> Any:
    """One real call, with account-side refusals turned into a skip."""
    started = time.monotonic()
    result: Any
    try:
        result = await client.structured(task, spec, schema)
    except ModelUnavailable as error:
        if error.status in ACCOUNT_SIDE_STATUSES and not error.transient:
            _report_refusal(task, client, f"HTTP {error.status}", started)
            pytest.skip(
                f"account-side refusal for {task.value}: the API key's project is not "
                f"entitled to model {client.model_for(task)} (HTTP {error.status}). "
                "See docs/TOOLCHAIN.md; this is an entitlement issue, not a code issue."
            )
        raise
    except ModelRateLimited:
        _report_refusal(task, client, "HTTP 429", started)
        # Measured on 2026-10-10: the pro models this key *can* reach return 429
        # with "free_tier_requests, limit: 0", which is an entitlement problem
        # wearing a rate limit's clothes. Retrying cannot fix a limit of zero.
        pytest.skip(
            f"quota refusal for {task.value}: model {client.model_for(task)} returned 429. "
            "If the free-tier limit for this model is 0 this is an entitlement issue; "
            "see docs/TOOLCHAIN.md."
        )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    assert isinstance(result.value, schema)
    assert result.usage.source == "live"
    assert result.usage.prompt_tokens > 0, "usage_metadata must report input tokens"
    assert result.usage.total_tokens > 0
    assert result.usage.attempts == 1
    assert result.usage.latency_ms > 0
    assert result.usage.estimated_cost_usd >= 0.0
    report(task, result.usage.model, result.usage, elapsed_ms)
    return result


def test_live_api_key_is_valid(live_settings: Settings) -> None:
    """Prove the key and the transport work, without a generateContent call.

    This is the test that still passes while the project is denied model access,
    and it is the one that distinguishes "the key is wrong" from "the key is
    fine but the project is not entitled".
    """
    from google import genai

    client = genai.Client(api_key=live_settings.gemini_api_key.get_secret_value())
    names = [model.name for model in client.models.list()]

    assert names, "models.list() returned nothing, so the key or network is broken"
    print(f"\n[live] models.list() returned {len(names)} models")
    configured = live_settings.gemini_model
    print(f"[live] configured model present in list: {f'models/{configured}' in names}")


async def test_live_extraction(live_client: GeminiClient) -> None:
    await run_once(
        live_client,
        GeminiTask.EXTRACT_PROFILE,
        extraction.build(NARRATIVE),
        ExtractionResponse,
    )


async def test_live_clarification(live_client: GeminiClient) -> None:
    program = make_program()
    candidate = RetrievedProgram(
        program=program, ruleset=make_ruleset(), sources=program.sources, retrieval_score=0.9
    )
    await run_once(
        live_client,
        GeminiTask.PROPOSE_CLARIFICATIONS,
        clarification.build(
            make_profile(household_size=None, unknown_fields=["household_size", "age"]),
            missing_fields=["household_size", "age"],
            candidates=[candidate],
            max_questions=2,
        ),
        ClarificationResponse,
    )


async def test_live_explanation(live_client: GeminiClient) -> None:
    program = make_program()
    await run_once(
        live_client,
        GeminiTask.EXPLAIN_MATCH,
        explanation.build(make_profile(), program, make_evaluation(program)),
        ExplanationResponse,
    )


async def test_live_action_plan(live_client: GeminiClient) -> None:
    program = make_program()
    await run_once(
        live_client,
        GeminiTask.DRAFT_ACTION_PLAN,
        action_plan.build(make_profile(), [make_match(program)]),
        ActionPlanResponse,
    )


async def test_live_chat(live_client: GeminiClient) -> None:
    program = make_program()
    await run_once(
        live_client,
        GeminiTask.ANSWER_QUESTION,
        chat.build(
            ChatRequest(message="What does this program say it asks for?"),
            sources=program.sources,
        ),
        ChatAnswerResponse,
    )


def test_every_prompt_path_is_covered_by_this_module() -> None:
    """A new task must come with a live smoke call, not just unit tests."""
    covered = {
        GeminiTask.EXTRACT_PROFILE,
        GeminiTask.PROPOSE_CLARIFICATIONS,
        GeminiTask.EXPLAIN_MATCH,
        GeminiTask.DRAFT_ACTION_PLAN,
        GeminiTask.ANSWER_QUESTION,
    }
    assert covered == set(MODEL_RESPONSE_SCHEMAS)
