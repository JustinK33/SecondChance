"""The result object a user actually reads: a program paired with its verdict.

:class:`ProgramMatch` is where the deterministic engine and the language model
meet, and it is deliberately shaped so they cannot be confused:

* ``evaluation`` is the engine's output and the only source of ``status``.
* ``explanation`` is model-written prose and carries no status of its own.
* A model validator asserts ``status is evaluation.status``, so the rule
  "Gemini may never change a match status" is enforced by the type system
  rather than by convention.
"""

from __future__ import annotations

import uuid
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import ConfidenceScore, NonEmptyStr, PlainText
from app.schemas.eligibility import EligibilityEvaluation
from app.schemas.enums import MatchStatus
from app.schemas.program import ProgramSource, ProgramSummary

__all__ = ["MatchExplanation", "ProgramMatch"]


class MatchExplanation(BaseModel):
    """Plain-language wording for a status that has already been decided.

    The generator is given the status and the evidence and asked only to word
    them. ``generated_by`` records which path produced the text, so a run that
    fell back to deterministic templates (no model access, a timeout, or output
    that failed validation) is visibly different from one that did not.
    """

    model_config = ConfigDict(extra="forbid")

    headline: PlainText = Field(
        description="One short sentence. Must not contain 'eligible' or 'ineligible'."
    )
    why_it_may_fit: list[PlainText] = Field(
        default_factory=list, description="Reasons drawn only from evidence with outcome TRUE."
    )
    what_we_still_need: list[PlainText] = Field(
        default_factory=list, description="One line per missing field, phrased as a question."
    )
    caveats: list[PlainText] = Field(
        default_factory=list,
        description="Limitations copied from the evaluation. Never softened or dropped.",
    )
    generated_by: Literal["gemini", "deterministic_fallback"] = "deterministic_fallback"
    prompt_version: NonEmptyStr | None = Field(
        default=None, description="Prompt template version, for reproducibility."
    )


class ProgramMatch(BaseModel):
    """One program as presented to the user, with its verdict and reasoning."""

    model_config = ConfigDict(extra="forbid")

    program: ProgramSummary
    status: MatchStatus = Field(description="Mirrors evaluation.status; validated to be identical.")
    score: ConfidenceScore = Field(default=0.0, description="Ranking aid only, in [0,1].")
    evaluation: EligibilityEvaluation
    explanation: MatchExplanation | None = Field(
        default=None,
        description=(
            "``None`` while explanations are still being generated, or if generation failed."
        ),
    )
    sources: list[ProgramSource] = Field(
        default_factory=list, description="Sources cited by this match's evidence."
    )

    @model_validator(mode="after")
    def _status_is_the_engine_status(self) -> Self:
        """The structural guarantee that language generation cannot move a status."""
        if self.status is not self.evaluation.status:
            msg = (
                f"match status {self.status.value} does not match the rule engine's "
                f"{self.evaluation.status.value}; only the rule engine may set a status"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _program_ids_agree(self) -> Self:
        if self.program.id != self.evaluation.program_id:
            msg = "evaluation.program_id does not match program.id"
            raise ValueError(msg)
        return self

    @property
    def program_id(self) -> uuid.UUID:
        return self.program.id
