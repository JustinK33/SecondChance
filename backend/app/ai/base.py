"""The shared half of every :class:`~app.schemas.contracts.GeminiPort`.

Each of the five port methods is the same four steps: build a versioned prompt,
run one structured call, project the response onto the contract type, drop
whatever the projection refused. Only the middle step differs between
implementations - the live client talks to Google, the fake replays a fixture -
so the other three live here and both implementations inherit them.

This is what makes the fake *faithful* rather than merely convenient. A test
that exercises ``FakeGeminiClient.answer_question`` exercises the same prompt
builder, the same response schema, the same decoder and the same grounding gate
that production runs, and it raises the same error classes. A fake that
reimplemented the projection would drift, and the drift would be invisible
precisely in the paths - injected instructions, unresolvable citations - that
the fake exists to test.

Subclasses implement :meth:`BaseGeminiPort.structured` and
:meth:`BaseGeminiPort.text`.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from app.ai.errors import ModelInvalidOutput
from app.ai.prompts import action_plan, chat, clarification, explanation, extraction
from app.ai.prompts.base import PromptSpec
from app.ai.prompts.context import catalog_urls
from app.ai.schemas import (
    ActionPlanResponse,
    ChatAnswerResponse,
    ClarificationResponse,
    ExplanationResponse,
    ExtractionResponse,
    GeminiTask,
)
from app.core.logging import get_logger
from app.schemas.ai import GeminiResult, ModelUsage
from app.schemas.assistant import ChatRequest, ChatResponse
from app.schemas.common import slugify
from app.schemas.eligibility import EligibilityEvaluation
from app.schemas.matching import MatchExplanation, ProgramMatch
from app.schemas.plan import ActionPlanDraft
from app.schemas.profile import (
    PROFILE_FIELD_NAMES,
    ClarificationQuestionSet,
    ExtractedProfile,
    ProfileDraft,
    UserProfile,
)
from app.schemas.program import AssistanceProgram, ProgramSource, RetrievedProgram

__all__ = [
    "BaseGeminiPort",
    "CallContext",
    "StructuredResult",
    "TextResult",
    "question_id_for_field",
]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CallContext:
    """Correlation ids for one model call.

    Ids only. Anything that could identify a person belongs nowhere near a
    model call's metadata.
    """

    request_id: str | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredResult[M: BaseModel]:
    """A validated structured response, what it cost, and any parse notes."""

    value: M
    usage: ModelUsage
    notes: list[str]


@dataclass(frozen=True, slots=True)
class TextResult:
    """A free-form response and what it cost."""

    text: str
    usage: ModelUsage


def question_id_for_field(field: str) -> str:
    """Deterministic clarification question id, derived from the field name.

    ``app.schemas.common.clarification_question_id`` needs a ``run_id``, which
    :class:`~app.schemas.contracts.GeminiPort` does not pass, so ids are
    field-scoped here and are unique within one question set. A caller that
    persists a set should rewrite them with the run-scoped helper.
    """
    return f"ask-{slugify(field)}"


class BaseGeminiPort(abc.ABC):
    """Prompt building and response projection, shared by every implementation."""

    # ------------------------------------------------------------- primitives
    @abc.abstractmethod
    async def structured[M: BaseModel](
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        schema: type[M],
        context: CallContext | None = None,
    ) -> StructuredResult[M]:
        """Run one structured call and return the validated response."""

    @abc.abstractmethod
    async def text(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        context: CallContext | None = None,
    ) -> TextResult:
        """Run one free-form call and return its text."""

    # ---------------------------------------------------------------- logging
    def log_projection(
        self, task: GeminiTask, dropped: Sequence[str], notes: Sequence[str] = ()
    ) -> None:
        """Record what the projection refused to keep.

        Reason strings come from a fixed vocabulary in :mod:`app.ai.schemas`,
        never from model text, so this line carries no user content.
        """
        if not dropped and not notes:
            return
        log.info(
            "gemini.projection",
            task=task.value,
            dropped=len(dropped),
            reasons=sorted(set(dropped)),
            notes=sorted(set(notes)),
        )

    # ------------------------------------------------------------ GeminiPort
    async def extract_profile(
        self,
        narrative: str,
        *,
        prefilled: ProfileDraft | None = None,
        request_id: str | None = None,
    ) -> GeminiResult[ExtractedProfile]:
        """Extract structured profile fields from a free-text narrative."""
        task = GeminiTask.EXTRACT_PROFILE
        spec = extraction.build(narrative, prefilled=prefilled)
        result = await self.structured(
            task, spec, ExtractionResponse, CallContext(request_id=request_id)
        )
        value, dropped = result.value.to_extracted_profile()
        self.log_projection(task, dropped, result.notes)
        return GeminiResult(value=value, usage=result.usage)

    async def propose_clarifications(
        self,
        profile: UserProfile,
        *,
        missing_fields: Sequence[str],
        candidates: Sequence[RetrievedProgram] = (),
        max_questions: int = 3,
        request_id: str | None = None,
    ) -> GeminiResult[ClarificationQuestionSet]:
        """Propose questions whose answers could change a result."""
        task = GeminiTask.PROPOSE_CLARIFICATIONS
        asked_about = [name for name in missing_fields if name in PROFILE_FIELD_NAMES]
        spec = clarification.build(
            profile,
            missing_fields=asked_about,
            candidates=candidates,
            max_questions=max_questions,
        )
        result = await self.structured(
            task, spec, ClarificationResponse, CallContext(request_id=request_id)
        )
        value, dropped = result.value.to_question_set(
            id_for=question_id_for_field,
            allowed_fields=asked_about,
            allowed_slugs=[candidate.program.slug for candidate in candidates],
            max_questions=max_questions,
        )
        self.log_projection(task, dropped, result.notes)
        return GeminiResult(value=value, usage=result.usage)

    async def explain_match(
        self,
        profile: UserProfile,
        program: AssistanceProgram,
        evaluation: EligibilityEvaluation,
        *,
        request_id: str | None = None,
    ) -> GeminiResult[MatchExplanation]:
        """Word an already-decided status in plain language.

        The decided status is an input and is never returned: the result is a
        :class:`~app.schemas.matching.MatchExplanation`, which has no status
        field. Prose containing the banned vocabulary is rejected whole rather
        than scrubbed, because a half-corrected explanation is less trustworthy
        than falling back to deterministic text.
        """
        task = GeminiTask.EXPLAIN_MATCH
        spec = explanation.build(profile, program, evaluation)
        result = await self.structured(
            task, spec, ExplanationResponse, CallContext(request_id=request_id)
        )
        if (banned := result.value.banned_term()) is not None:
            log.warning(
                "gemini.banned_vocabulary",
                task=task.value,
                model=result.usage.model,
                term=banned,
                prompt_version=explanation.VERSION,
            )
            raise ModelInvalidOutput(
                task=task.value, model=result.usage.model, reason="banned_vocabulary"
            )
        try:
            value = result.value.to_match_explanation(
                prompt_version=explanation.VERSION,
                required_caveats=evaluation.limitations,
            )
        except ValueError as exc:
            raise ModelInvalidOutput(
                task=task.value,
                model=result.usage.model,
                reason="unusable_explanation",
                cause=exc,
            ) from exc
        self.log_projection(task, (), result.notes)
        return GeminiResult(value=value, usage=result.usage)

    async def draft_action_plan(
        self,
        profile: UserProfile,
        matches: Sequence[ProgramMatch],
        *,
        request_id: str | None = None,
    ) -> GeminiResult[ActionPlanDraft]:
        """Draft next steps for a decided set of matches."""
        task = GeminiTask.DRAFT_ACTION_PLAN
        spec = action_plan.build(profile, matches)
        result = await self.structured(
            task, spec, ActionPlanResponse, CallContext(request_id=request_id)
        )
        try:
            value, dropped = result.value.to_draft(
                allowed_slugs=[match.program.slug for match in matches],
                allowed_urls=catalog_urls(matches),
            )
        except ValueError as exc:
            raise ModelInvalidOutput(
                task=task.value, model=result.usage.model, reason="unusable_plan", cause=exc
            ) from exc
        self.log_projection(task, dropped, result.notes)
        return GeminiResult(value=value, usage=result.usage)

    async def answer_question(
        self,
        request: ChatRequest,
        *,
        sources: Sequence[ProgramSource],
        request_id: str | None = None,
    ) -> GeminiResult[ChatResponse]:
        """Answer a follow-up question from the supplied sources only."""
        task = GeminiTask.ANSWER_QUESTION
        spec = chat.build(request, sources=sources)
        result = await self.structured(
            task, spec, ChatAnswerResponse, CallContext(request_id=request_id)
        )
        if (banned := result.value.banned_term()) is not None:
            log.warning(
                "gemini.banned_vocabulary",
                task=task.value,
                model=result.usage.model,
                term=banned,
                prompt_version=chat.VERSION,
            )
            raise ModelInvalidOutput(
                task=task.value, model=result.usage.model, reason="banned_vocabulary"
            )
        value, dropped = result.value.to_chat_response(known_sources=sources)
        self.log_projection(task, dropped, result.notes)
        return GeminiResult(value=value, usage=result.usage)
