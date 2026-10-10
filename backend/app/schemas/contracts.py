"""Structural interfaces between the modules of the application.

These :class:`typing.Protocol` definitions exist so that independently
developed layers agree on a seam without importing each other's
implementations. ``app.workflows`` depends on :class:`Retriever`, not on
``app.retrieval.hybrid.HybridRetriever``, which means a node can be unit tested
against a three-line stand-in and the real implementation can be replaced
without touching the graph.

All of these are ``runtime_checkable``, so a test can assert a concrete class
satisfies a seam. Note that ``isinstance`` against a runtime-checkable Protocol
only checks method *names*; mypy checks the signatures, which is why the type
check is part of the definition of done rather than an optional extra.

Async / sync split
------------------
Anything that touches the network or the database is ``async``.
:class:`EligibilityEngine` is deliberately **synchronous**: it is a pure
function of a profile and a rule set, performs no I/O (retrieval hands it
everything it needs in a :class:`~app.schemas.program.RetrievedProgram`), and
making it ``async`` would both imply I/O that does not happen and let a future
author quietly add some.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

from app.schemas.ai import GeminiResult
from app.schemas.assistant import (
    ChatRequest,
    ChatResponse,
    Citation,
    CitationVerificationReport,
)
from app.schemas.eligibility import EligibilityEvaluation, RuleSet
from app.schemas.enums import NeedCategory, ProgramCategory, VerificationStatus
from app.schemas.matching import MatchExplanation, ProgramMatch
from app.schemas.plan import ActionPlanDraft
from app.schemas.profile import (
    ClarificationQuestionSet,
    ExtractedProfile,
    ProfileDraft,
    UserProfile,
)
from app.schemas.program import (
    AssistanceProgram,
    ProgramDetail,
    ProgramPage,
    ProgramSource,
    RetrievedProgram,
)

__all__ = [
    "CitationVerifier",
    "EligibilityEngine",
    "GeminiPort",
    "ProgramRepository",
    "Retriever",
]


@runtime_checkable
class ProgramRepository(Protocol):
    """Read access to the curated program catalog.

    Implemented by ``app.retrieval.repository.SqlProgramRepository`` over
    SQLAlchemy, and by an in-memory implementation over
    ``backend/data/programs`` for unit tests and for the seed script.

    Implementations receive an ``AsyncSession`` at construction, never per
    call, and one instance is therefore bound to one session and must not be
    shared across concurrent tasks.
    """

    async def list_programs(
        self,
        *,
        categories: Sequence[ProgramCategory] | None = None,
        state: str | None = None,
        county: str | None = None,
        query: str | None = None,
        verification_statuses: Sequence[VerificationStatus] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ProgramPage:
        """Paginated program list. ``query`` is a full-text search over name,
        summary and keywords."""
        ...

    async def get_program(self, program_id: uuid.UUID) -> ProgramDetail | None:
        """Full detail view, including sources and rendered requirement bullets."""
        ...

    async def get_program_by_slug(self, slug: str) -> ProgramDetail | None:
        """Same as :meth:`get_program`, keyed by the stable catalog slug."""
        ...

    async def get_programs_by_ids(
        self, program_ids: Sequence[uuid.UUID]
    ) -> list[AssistanceProgram]:
        """Batch fetch. Exists so callers never loop a single-row query."""
        ...

    async def get_active_ruleset(
        self, program_id: uuid.UUID, *, on: date | None = None
    ) -> RuleSet | None:
        """The approved rule set in force on ``on`` (default today).

        Returns ``None`` when the program has no approved rule set, which the
        engine must treat as "cannot assess", never as "does not qualify".
        """
        ...

    async def get_sources(
        self, program_id: uuid.UUID, *, source_ids: Sequence[str] | None = None
    ) -> list[ProgramSource]:
        """Source documents for a program, optionally filtered to specific ids."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Selects which programs are worth evaluating for a profile.

    Retrieval decides *relevance*, never *eligibility*. It is allowed to be
    generous: a candidate that the engine later rules out costs one evaluation,
    whereas a candidate never retrieved is help the user never hears about.
    """

    async def retrieve(
        self,
        profile: UserProfile,
        *,
        needs: Sequence[NeedCategory] | None = None,
        limit: int = 20,
    ) -> list[RetrievedProgram]:
        """Candidates ordered by descending ``retrieval_score``.

        Each candidate arrives complete - program, approved rule set and
        sources - so the eligibility engine performs no I/O.
        """
        ...

    async def retrieve_for_programs(
        self, program_ids: Sequence[uuid.UUID]
    ) -> list[RetrievedProgram]:
        """Rehydrate specific candidates for selective re-evaluation.

        Used when a profile edit affects only some programs: the unaffected
        results are preserved rather than recomputed.
        """
        ...

    async def search_sources(
        self,
        query: str,
        *,
        program_id: uuid.UUID | None = None,
        limit: int = 5,
    ) -> list[ProgramSource]:
        """Passage search over source documents, for grounding the assistant."""
        ...


@runtime_checkable
class EligibilityEngine(Protocol):
    """The deterministic preliminary eligibility engine. Synchronous and pure.

    Contract every implementation must honour:

    * Given the same profile, rule set and ``now``, the output is identical
      apart from ``evaluated_at``.
    * A missing profile field yields ``Trilean.UNKNOWN`` and populates
      ``missing_fields``; it never yields ``FALSE``.
    * ``status`` is derived only via
      :func:`app.schemas.eligibility.status_for_outcome`.
    * Every condition visited contributes a
      :class:`~app.schemas.eligibility.ConditionEvidence` record.
    """

    def evaluate(
        self,
        profile: UserProfile,
        program: AssistanceProgram,
        ruleset: RuleSet,
        *,
        now: datetime | None = None,
    ) -> EligibilityEvaluation:
        """Evaluate one program against one profile."""
        ...

    def evaluate_many(
        self,
        profile: UserProfile,
        candidates: Sequence[RetrievedProgram],
        *,
        now: datetime | None = None,
    ) -> list[EligibilityEvaluation]:
        """Evaluate every candidate that has an approved rule set.

        Candidates without one are skipped and reported through
        ``limitations`` on the run rather than silently dropped.
        """
        ...

    def affected_programs(
        self,
        changed_fields: Sequence[str],
        candidates: Sequence[RetrievedProgram],
    ) -> list[uuid.UUID]:
        """Which candidates could change if ``changed_fields`` changed.

        Drives selective re-execution: a household-size edit should not
        re-evaluate, re-explain and re-bill a program whose rules never read
        household size.
        """
        ...


@runtime_checkable
class GeminiPort(Protocol):
    """Every model-mediated capability, behind one seam.

    Two implementations exist and both are first-class:

    * ``app.ai.gemini.GeminiClient`` - the live google-genai client.
    * ``app.ai.fallback.DeterministicAi`` - templates and heuristics, no
      network.

    The fallback is not a test double. The configured model is not reachable
    with this project's API key (``docs/TOOLCHAIN.md``), so the product must be
    fully usable through it, and every method returns a
    :class:`~app.schemas.ai.GeminiResult` whose ``usage.source`` records which
    path ran.

    None of these methods may return or influence a
    :class:`~app.schemas.enums.MatchStatus`. :meth:`explain_match` receives the
    decided status and only words it.
    """

    async def extract_profile(
        self,
        narrative: str,
        *,
        prefilled: ProfileDraft | None = None,
        request_id: str | None = None,
    ) -> GeminiResult[ExtractedProfile]:
        """Turn a free-text narrative into structured, optional fields.

        Must leave a field unset rather than guess, and must not infer a
        sensitive attribute the user did not state.

        Raises ``InvalidModelOutputError`` if the response fails schema
        validation after the configured retries.
        """
        ...

    async def propose_clarifications(
        self,
        profile: UserProfile,
        *,
        missing_fields: Sequence[str],
        candidates: Sequence[RetrievedProgram] = (),
        max_questions: int = 3,
        request_id: str | None = None,
    ) -> GeminiResult[ClarificationQuestionSet]:
        """Propose questions whose answers could change a match status.

        Must not ask for an SSN, banking credentials, or any identifier the
        retrieved programs do not require.
        """
        ...

    async def explain_match(
        self,
        profile: UserProfile,
        program: AssistanceProgram,
        evaluation: EligibilityEvaluation,
        *,
        request_id: str | None = None,
    ) -> GeminiResult[MatchExplanation]:
        """Word an already-decided status in plain language.

        The status and evidence are inputs. Implementations must not emit
        "eligible" or "ineligible" and must not contradict the evidence.
        """
        ...

    async def draft_action_plan(
        self,
        profile: UserProfile,
        matches: Sequence[ProgramMatch],
        *,
        request_id: str | None = None,
    ) -> GeminiResult[ActionPlanDraft]:
        """Draft next steps. Ids, ordering and deduplication are server-side.

        URLs in the draft are validated against the catalog before persistence;
        a URL not present in the catalog is dropped, never followed.
        """
        ...

    async def answer_question(
        self,
        request: ChatRequest,
        *,
        sources: Sequence[ProgramSource],
        request_id: str | None = None,
    ) -> GeminiResult[ChatResponse]:
        """Answer a follow-up question from the supplied sources only.

        ``sources`` is retrieved third-party content and is untrusted:
        implementations must treat it as data and ignore any instruction inside
        it. Claims that no source supports belong in ``interpretation``, never
        in ``facts``.
        """
        ...


@runtime_checkable
class CitationVerifier(Protocol):
    """Deterministically checks generated citations against the real catalog.

    No model participates in verification - that would be asking the generator
    to grade itself. Checks are identity and substring matching against stored
    :class:`~app.schemas.program.ProgramSource` records.
    """

    async def verify(
        self,
        citations: Sequence[Citation],
        *,
        known_sources: Sequence[ProgramSource] | None = None,
    ) -> CitationVerificationReport:
        """Verify that each citation names a real source and quotes it accurately.

        Rejects a citation whose ``source_id`` is unknown, whose ``url`` does
        not match the stored source, whose ``excerpt`` does not occur in the
        source text, or whose source is ``STALE``.
        """
        ...

    async def verify_response(
        self,
        response: ChatResponse,
        *,
        known_sources: Sequence[ProgramSource] | None = None,
    ) -> CitationVerificationReport:
        """Verify a whole assistant response, including that every
        :class:`~app.schemas.assistant.FactStatement` is supported.

        Unsupported fact statements are returned in ``unsupported_claims``; the
        caller must demote or drop them before the response reaches the user.
        """
        ...
