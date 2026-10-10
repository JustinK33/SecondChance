"""Source-grounded assistant schemas.

The assistant answers questions like "what does household income mean for this
program?". It must never become a general chatbot that sounds authoritative
about benefits, so the response type forces the separation the product promises:

* :class:`FactStatement` requires at least one source id. A sentence presented
  as fact cannot exist without a citation.
* ``interpretation`` holds everything else - helpful reasoning, explicitly
  labelled as the assistant's reading rather than as official guidance.
* An answer with no citations is representable, because refusing to answer is
  sometimes correct, but ``is_grounded`` is then forced to ``False`` and
  ``ungrounded_notice`` is populated automatically. The UI has no way to render
  an ungrounded answer as if it were sourced.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas.common import HttpUrlStr, PlainText, SourceId
from app.schemas.enums import ConfidenceLevel
from app.schemas.program import ProgramSource

__all__ = [
    "UNGROUNDED_NOTICE",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "CitationVerificationReport",
    "FactStatement",
    "RejectedCitation",
]

#: Shown verbatim whenever an answer carries no citations.
UNGROUNDED_NOTICE = (
    "We could not find this in an official source, so treat the answer as general "
    "information and confirm it with the program before relying on it."
)

MAX_CHAT_MESSAGE_CHARS = 1000
MAX_HISTORY_TURNS = 8


class ChatMessage(BaseModel):
    """One prior turn, replayed by the client.

    History is client-supplied and therefore untrusted: the prompt builder must
    treat it as data and never as instructions.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS),
    ]


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/assistant/chat``."""

    model_config = ConfigDict(extra="forbid")

    message: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=2, max_length=MAX_CHAT_MESSAGE_CHARS),
    ]
    program_id: uuid.UUID | None = Field(
        default=None, description="Scopes retrieval to one program's sources."
    )
    run_id: uuid.UUID | None = Field(
        default=None,
        description=(
            "Scopes the answer to a run the caller owns. Ownership is checked "
            "server-side; the id in the body is never trusted as authorisation."
        ),
    )
    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class Citation(BaseModel):
    """A pointer into an official source document."""

    model_config = ConfigDict(extra="forbid")

    source_id: SourceId
    program_id: uuid.UUID | None = None
    url: HttpUrlStr
    title: PlainText
    excerpt: PlainText = Field(
        description="Verbatim passage supporting the claim. Must appear in the cited source."
    )


class FactStatement(BaseModel):
    """A sentence presented as fact. Cannot exist without a citation."""

    model_config = ConfigDict(extra="forbid")

    text: PlainText
    source_ids: list[SourceId] = Field(
        min_length=1, description="At least one. A fact with no source is an interpretation."
    )


class ChatResponse(BaseModel):
    """Body of ``POST /api/v1/assistant/chat``."""

    model_config = ConfigDict(extra="forbid")

    answer: PlainText = Field(description="The full answer, in plain language.")
    facts: list[FactStatement] = Field(
        default_factory=list, description="Cited claims. Each one is traceable to a source."
    )
    interpretation: list[PlainText] = Field(
        default_factory=list,
        description="Uncited reasoning, rendered under an explicit 'our reading' heading.",
    )
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    caveats: list[PlainText] = Field(default_factory=list)
    suggested_next_steps: list[PlainText] = Field(default_factory=list, max_length=5)
    is_grounded: bool = Field(
        default=False, description="Computed, not supplied: true only when citations exist."
    )
    ungrounded_notice: str | None = Field(
        default=None, description="Computed. Populated exactly when is_grounded is false."
    )

    @model_validator(mode="after")
    def _enforce_grounding(self) -> Self:
        """Derive the grounding flags so a caller cannot misreport them."""
        grounded = bool(self.citations)
        if self.is_grounded != grounded:
            object.__setattr__(self, "is_grounded", grounded)
        if grounded:
            if self.ungrounded_notice is not None:
                object.__setattr__(self, "ungrounded_notice", None)
        elif not self.ungrounded_notice:
            object.__setattr__(self, "ungrounded_notice", UNGROUNDED_NOTICE)
        return self

    @model_validator(mode="after")
    def _facts_cite_declared_sources(self) -> Self:
        """Every id a fact cites must appear in ``citations``.

        Without this a response could claim support from a source the user is
        never shown, which is indistinguishable from making it up.
        """
        declared = {citation.source_id for citation in self.citations}
        for fact in self.facts:
            missing = sorted(set(fact.source_ids) - declared)
            if missing:
                msg = f"fact cites source ids absent from citations: {missing}"
                raise ValueError(msg)
        return self


class RejectedCitation(BaseModel):
    """A citation that failed verification, with the reason."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    reason: Literal[
        "unknown_source",
        "excerpt_not_found",
        "url_mismatch",
        "stale_source",
        "program_mismatch",
    ]
    detail: str | None = Field(default=None, max_length=300)


class CitationVerificationReport(BaseModel):
    """Result of checking generated citations against the real catalog.

    Verification is deterministic string and identity matching against stored
    sources - no model is involved in deciding whether a citation is real.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool = Field(description="True only when nothing was rejected and no claim is unsupported.")
    verified: list[Citation] = Field(default_factory=list)
    rejected: list[RejectedCitation] = Field(default_factory=list)
    unsupported_claims: list[PlainText] = Field(
        default_factory=list,
        description="Sentences presented as fact that no verified source supports.",
    )
    checked_sources: list[ProgramSource] = Field(
        default_factory=list, description="The source records the check ran against."
    )

    @model_validator(mode="after")
    def _ok_is_consistent(self) -> Self:
        expected = not self.rejected and not self.unsupported_claims
        if self.ok != expected:
            msg = "ok must be true exactly when there are no rejections and no unsupported claims"
            raise ValueError(msg)
        return self

    @property
    def unsupported_claim_rate(self) -> float:
        """Share of claims that no source supports. Reported in METRICS.md."""
        total = len(self.verified) + len(self.rejected) + len(self.unsupported_claims)
        if total == 0:
            return 0.0
        return len(self.unsupported_claims) / total
