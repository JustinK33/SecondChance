"""Model-call envelopes and telemetry.

Every call through :class:`~app.schemas.contracts.GeminiPort` returns a
:class:`GeminiResult`, which carries the validated value *and* the cost of
getting it. Threading usage through the return type rather than a side channel
means per-workflow token and cost accounting is a by-product of normal control
flow and cannot quietly stop working.

``source`` is the other half of the design. The configured model is not
currently reachable with this project's API key (see ``docs/TOOLCHAIN.md``), so
the AI layer must always have a deterministic fallback. Recording which path
produced a value keeps that honest: a benchmark or a demo can never present
fallback output as model output.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import NonEmptyStr

__all__ = [
    "GeminiResult",
    "ModelCallSource",
    "ModelUsage",
    "PromptRef",
]

#: Where a value actually came from.
#:
#: ``live``     - a successful model response.
#: ``cache``    - a previously validated response for an identical request.
#: ``fallback`` - the deterministic implementation, because the model was
#:                unavailable, timed out, or returned output that failed
#:                validation.
ModelCallSource = Literal["live", "cache", "fallback"]


class PromptRef(BaseModel):
    """Identifies the exact prompt that produced an output.

    Prompts are versioned data, not inline strings, so a quality regression can
    be attributed to a specific template revision.
    """

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr = Field(description='Template name, e.g. "extract_profile".')
    version: NonEmptyStr = Field(description='Template version, e.g. "v3".')


class ModelUsage(BaseModel):
    """Cost and latency of one logical model call, including its retries.

    Token counts come from ``response.usage_metadata``. Gemini bills thinking
    tokens as output, so ``estimated_cost_usd`` is computed from
    ``output_tokens + thought_tokens``.

    Contains no user content and is therefore the one AI-related object that is
    safe to log in full.
    """

    model_config = ConfigDict(extra="forbid")

    model: NonEmptyStr = Field(description="Resolved model id actually used.")
    prompt: PromptRef | None = None
    source: ModelCallSource = "fallback"
    prompt_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    thought_tokens: int = Field(default=0, ge=0, description="Billed as output on thinking models.")
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0, description="Wall clock, including retry backoff.")
    attempts: int = Field(default=1, ge=1, description="Total attempts, so retries = attempts - 1.")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    truncated: bool = Field(
        default=False,
        description="Response hit MAX_TOKENS. Truncated structured output must never be trusted.",
    )

    @property
    def retries(self) -> int:
        return self.attempts - 1

    @property
    def billable_output_tokens(self) -> int:
        return self.output_tokens + self.thought_tokens

    @classmethod
    def for_fallback(cls, model: str, prompt: PromptRef | None = None) -> Self:
        """Usage record for a deterministic result: no tokens, no cost."""
        return cls(model=model, prompt=prompt, source="fallback")


class GeminiResult[T](BaseModel):
    """A validated model output together with what it cost to obtain."""

    model_config = ConfigDict(extra="forbid")

    value: T
    usage: ModelUsage

    @property
    def is_fallback(self) -> bool:
        """Whether this value came from the deterministic path, not the model."""
        return self.usage.source == "fallback"

    @model_validator(mode="after")
    def _truncated_output_is_not_a_result(self) -> Self:
        if self.usage.truncated:
            msg = (
                "a truncated model response cannot be returned as a result; "
                "raise InvalidModelOutputError or fall back instead"
            )
            raise ValueError(msg)
        return self
