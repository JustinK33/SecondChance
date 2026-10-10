"""A deterministic :class:`~app.schemas.contracts.GeminiPort` backed by fixtures.

What this is for
----------------
No live Gemini call can currently succeed with this project's API key
(``docs/TOOLCHAIN.md``), and even when one can, a test suite that depends on a
sampled model is a test suite that fails on Fridays. So CI runs against recorded
responses, replayed through the *same* decoder, the same response schemas and
the same projections as production - see :mod:`app.ai.base` for why that
sharing is the whole point.

This is not the deterministic product fallback. ``app.ai.fallback.DeterministicAi``
(a sibling task) answers from templates and heuristics when the model is
unreachable in a real run, and its ``usage.source`` is ``"fallback"``. This class
replays a previously validated response, so its ``usage.source`` is ``"cache"``,
which is exactly what that value means. Neither one may ever be presented as
live model output.

Failure modes
-------------
``failure_mode`` makes the fake misbehave on purpose, with the same exception
types the live client raises:

===========================  ==================================================
``timeout``                  :class:`~app.ai.errors.ModelTimeout`
``rate_limited``             :class:`~app.ai.errors.ModelRateLimited`
``unavailable``              :class:`~app.ai.errors.ModelUnavailable` (503)
``malformed_json``           :class:`~app.ai.errors.ModelInvalidOutput`
``schema_violation``         :class:`~app.ai.errors.ModelInvalidOutput`
``empty_response``           :class:`~app.ai.errors.ModelInvalidOutput`
``injected_instructions``    a *successful* response that obeyed an injection
===========================  ==================================================

The last one is the interesting one: it is not an error. It replays a response
in which the model did what a poisoned program record told it to - cited a
source id that does not exist, used the banned vocabulary, pointed at an
off-catalog URL - so a test can assert that the projection layer, not the
model's good behaviour, is what keeps those out of the product.

Fixture format
--------------
One file per task under ``tests/fixtures/gemini/<task>.json``, where ``<task>``
is a :class:`~app.ai.schemas.GeminiTask` value::

    {
      "task": "extract_profile",
      "note": "why this fixture exists; everything in it is fictional",
      "usage": {"prompt_tokens": 1180, "output_tokens": 240, "thought_tokens": 0},
      "text": "a free-form answer, for GeminiClient.text()",
      "cases": {
        "default":               {"response": { ...valid model JSON... }},
        "injected_instructions": {"response": { ...the model obeyed... }},
        "schema_violation":      {"response": { ...wrong types... }},
        "malformed_json":        {"raw": "{\\"state\\": \\"NJ\\","},
        "empty_response":        {"raw": ""}
      }
    }

``response`` is re-serialised and fed through
:func:`app.ai.decode.decode_structured`, so a fixture that no longer matches its
schema fails the suite instead of silently diverging. ``raw`` is passed through
verbatim, which is how the malformed cases stay malformed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from app.ai import decode
from app.ai.base import BaseGeminiPort, CallContext, StructuredResult, TextResult
from app.ai.errors import (
    ModelInvalidOutput,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
)
from app.ai.gemini import MODEL_STAGE_FOR_TASK
from app.ai.prompts.base import PromptSpec
from app.ai.schemas import GeminiTask
from app.ai.usage import UsageAccumulator
from app.core.config import Settings
from app.schemas.ai import ModelUsage

__all__ = ["DEFAULT_FIXTURE_DIR", "FailureMode", "FakeGeminiClient", "FixtureNotFound"]

#: ``backend/tests/fixtures/gemini`` - this file is ``backend/app/ai/fake.py``.
DEFAULT_FIXTURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "gemini"
)

DEFAULT_CASE: Final[str] = "default"
_FALLBACK_MALFORMED: Final[str] = '{"state": "NJ", '
#: A body that violates every response schema in :data:`MODEL_RESPONSE_SCHEMAS`,
#: used when a fixture does not record its own ``schema_violation`` case. It has
#: to be a *type* violation rather than an unexpected key, because the schemas
#: ignore unexpected keys on purpose - see ``MODEL_OUTPUT_CONFIG``. Each entry
#: is the wrong type for a required or typed field of one of the five schemas.
_FALLBACK_SCHEMA_VIOLATION: Final[dict[str, Any]] = {
    "household_size": "not a number",
    "questions": "not a list",
    "headline": ["not", "a", "string"],
    "title": ["not", "a", "string"],
    "answer": ["not", "a", "string"],
}
_DEFAULT_LATENCY_MS: Final[int] = 12


class FailureMode(StrEnum):
    """How the fake should misbehave."""

    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_VIOLATION = "schema_violation"
    INJECTED_INSTRUCTIONS = "injected_instructions"
    EMPTY_RESPONSE = "empty_response"


class FixtureNotFound(LookupError):  # noqa: N818
    """A fixture file or case is missing.

    A programming error rather than a model failure, so it is deliberately not
    part of the :mod:`app.ai.errors` taxonomy: a test asking for a case that
    does not exist should say so, not look like a provider outage.
    """


class FakeGeminiClient(BaseGeminiPort):
    """Fixture-replaying :class:`GeminiPort` with switchable failure modes.

    ``failure_mode`` applies to every task; ``failure_modes`` overrides it for
    specific tasks, so a test can fail one leg of a fan-out and let the others
    succeed. ``cases`` picks a named fixture case per task for the success path.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        failure_mode: FailureMode | None = None,
        failure_modes: Mapping[GeminiTask, FailureMode] | None = None,
        cases: Mapping[GeminiTask, str] | None = None,
        fixture_dir: Path | None = None,
        latency_ms: int = _DEFAULT_LATENCY_MS,
        accumulator: UsageAccumulator | None = None,
    ) -> None:
        self._settings = settings
        self.failure_mode = failure_mode
        self.failure_modes: dict[GeminiTask, FailureMode] = dict(failure_modes or {})
        self.cases: dict[GeminiTask, str] = dict(cases or {})
        self.fixture_dir = fixture_dir or DEFAULT_FIXTURE_DIR
        self.latency_ms = latency_ms
        self.usage = accumulator if accumulator is not None else UsageAccumulator()
        #: Tasks called, in order. Ids and enums only - never prompts, so this
        #: list is safe to print in a failing assertion.
        self.calls: list[GeminiTask] = []
        #: The last prompt built, for tests that assert on prompt construction.
        #: Held in memory only and never logged.
        self.last_prompt: PromptSpec | None = None
        self._fixtures: dict[GeminiTask, dict[str, Any]] = {}

    # ------------------------------------------------------------------ config
    @property
    def settings(self) -> Settings:
        return self._settings

    def failure_for(self, task: GeminiTask) -> FailureMode | None:
        return self.failure_modes.get(task, self.failure_mode)

    def model_for(self, task: GeminiTask) -> str:
        """Mirror the live client's per-task model selection."""
        model: str = getattr(self._settings, MODEL_STAGE_FOR_TASK[task])
        return model

    # ---------------------------------------------------------------- fixtures
    def fixture(self, task: GeminiTask) -> dict[str, Any]:
        """Load and cache one task's fixture file."""
        cached = self._fixtures.get(task)
        if cached is not None:
            return cached
        path = self.fixture_dir / f"{task.value}.json"
        if not path.is_file():
            msg = f"no fixture for task {task.value!r} at {path}"
            raise FixtureNotFound(msg)
        with path.open(encoding="utf-8") as handle:
            loaded: dict[str, Any] = json.load(handle)
        self._fixtures[task] = loaded
        return loaded

    def case(self, task: GeminiTask, name: str) -> dict[str, Any] | None:
        cases = self.fixture(task).get("cases", {})
        found = cases.get(name)
        return found if isinstance(found, dict) else None

    # -------------------------------------------------------------- primitives
    async def structured[M: BaseModel](
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        schema: type[M],
        context: CallContext | None = None,
    ) -> StructuredResult[M]:
        """Replay a recorded response for ``task`` and validate it."""
        self.calls.append(task)
        self.last_prompt = prompt
        self._raise_transport_failure(task)
        body, case = self._response_body(task)
        usage = self._usage(task, prompt, case)
        value, notes = decode.decode_structured(
            schema,
            text=body,
            parsed=None,
            task=task.value,
            model=usage.model,
        )
        self.usage.add(usage)
        return StructuredResult(value=value, usage=usage, notes=notes)

    async def text(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        context: CallContext | None = None,
    ) -> TextResult:
        """Replay the fixture's free-form ``text`` answer."""
        self.calls.append(task)
        self.last_prompt = prompt
        self._raise_transport_failure(task)
        mode = self.failure_for(task)
        body = self.fixture(task).get("text")
        if mode is FailureMode.EMPTY_RESPONSE or not isinstance(body, str) or not body.strip():
            raise ModelInvalidOutput(
                task=task.value, model=self.model_for(task), reason="empty_response"
            )
        usage = self._usage(task, prompt, None)
        self.usage.add(usage)
        return TextResult(text=body.strip(), usage=usage)

    # ----------------------------------------------------------------- details
    def _raise_transport_failure(self, task: GeminiTask) -> None:
        """Raise the simulated provider-level failures, if one is configured."""
        mode = self.failure_for(task)
        model = self.model_for(task)
        if mode is FailureMode.TIMEOUT:
            raise ModelTimeout(
                task=task.value, model=model, timeout_s=self._settings.gemini_timeout_s
            )
        if mode is FailureMode.RATE_LIMITED:
            raise ModelRateLimited(task=task.value, model=model, retry_after_s=1)
        if mode is FailureMode.UNAVAILABLE:
            raise ModelUnavailable(task=task.value, model=model, status=503, transient=True)

    def _response_body(self, task: GeminiTask) -> tuple[str | None, dict[str, Any] | None]:
        """The raw body to decode, plus the case it came from."""
        mode = self.failure_for(task)
        if mode is FailureMode.EMPTY_RESPONSE:
            return self._case_body(task, mode.value, fallback_raw=""), None
        if mode is FailureMode.MALFORMED_JSON:
            return self._case_body(task, mode.value, fallback_raw=_FALLBACK_MALFORMED), None
        if mode is FailureMode.SCHEMA_VIOLATION:
            body = self._case_body(
                task,
                mode.value,
                fallback_raw=json.dumps(_FALLBACK_SCHEMA_VIOLATION),
            )
            return body, None
        name = mode.value if mode is FailureMode.INJECTED_INSTRUCTIONS else None
        name = name or self.cases.get(task, DEFAULT_CASE)
        case = self.case(task, name)
        if case is None:
            msg = f"fixture for {task.value!r} has no case {name!r}"
            raise FixtureNotFound(msg)
        return _body_of(case), case

    def _case_body(self, task: GeminiTask, name: str, *, fallback_raw: str) -> str | None:
        """A failure case from the fixture, or a synthetic one.

        Fixtures are free to record a realistic malformed body; when they do not,
        the synthetic fallback keeps the failure mode usable for every task.
        """
        case = self.case(task, name)
        if case is None:
            return fallback_raw
        return _body_of(case)

    def _usage(
        self, task: GeminiTask, prompt: PromptSpec, case: dict[str, Any] | None
    ) -> ModelUsage:
        """Build the usage record from the fixture's recorded token counts."""
        recorded = {**self.fixture(task).get("usage", {}), **(case or {}).get("usage", {})}
        prompt_tokens = int(recorded.get("prompt_tokens", 0) or 0)
        output_tokens = int(recorded.get("output_tokens", 0) or 0)
        thought_tokens = int(recorded.get("thought_tokens", 0) or 0)
        total = int(recorded.get("total_tokens", 0) or 0) or (
            prompt_tokens + output_tokens + thought_tokens
        )
        return ModelUsage(
            model=self.model_for(task),
            prompt=prompt.ref,
            source="cache",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thought_tokens=thought_tokens,
            total_tokens=total,
            latency_ms=self.latency_ms,
            attempts=1,
            estimated_cost_usd=self._settings.estimate_cost_usd(
                prompt_tokens, output_tokens + thought_tokens
            ),
            truncated=False,
        )


def _body_of(case: Mapping[str, Any]) -> str | None:
    """Render one fixture case as the raw response body.

    ``raw`` wins over ``response`` so a case can record a body that is not valid
    JSON at all, which is the only way to reproduce a malformed response
    faithfully.
    """
    if "raw" in case:
        raw = case["raw"]
        return raw if isinstance(raw, str) else None
    if "response" in case:
        return json.dumps(case["response"], ensure_ascii=False)
    return None
