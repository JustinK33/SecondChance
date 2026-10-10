"""Injected dependencies for the workflow graph.

The graph depends on the ``Protocol`` seams in
:mod:`app.schemas.contracts`, never on concrete classes, so
:func:`~app.workflows.graph.build_graph` takes one of these and a test can swap
in a fixture-replaying model client or an in-memory catalog without touching a
node.

Session discipline
------------------
``ProgramRepository``, ``Retriever`` and ``CitationVerifier`` are each bound to
one ``AsyncSession``, and a session must never be shared between concurrent
tasks. The first fan-out runs retrieval and evidence validation at the same
time, so they cannot share one. That is why this object carries a session
*factory* plus per-seam constructors rather than ready-made instances:
:meth:`WorkflowDeps.with_retriever` and :meth:`WorkflowDeps.with_verifier` open
a scope of their own, which is also what makes the transaction boundary inside
``persist_results`` obvious.

A test that wants no database at all sets ``retriever`` / ``verifier``
directly; those take precedence over the constructors and no session is opened.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import InternalError
from app.db.repositories import SqlProgramRepository
from app.db.session import session_scope
from app.eligibility.engine import MAX_CLARIFICATION_QUESTIONS, RuleEngine
from app.retrieval.citations import SourceCitationVerifier
from app.retrieval.search import HybridRetriever
from app.schemas import (
    ActionPlan,
    ActionPlanDraft,
    CitationVerifier,
    ClarificationAnswer,
    ClarificationQuestion,
    EligibilityEngine,
    EligibilityEvaluation,
    ExtractedProfile,
    GeminiPort,
    ProfileDraft,
    ProgramMatch,
    RetrievedProgram,
    Retriever,
    UserProfile,
)
from app.workflows.plan_builder import build_plan as default_build_plan
from app.workflows.profile_merge import assemble_profile as default_assemble_profile

__all__ = [
    "NavigatorEngine",
    "PlanBuilder",
    "ProfileAssembler",
    "SessionFactory",
    "WorkflowDeps",
]

#: Anything that hands back a fresh ``AsyncSession``. An ``async_sessionmaker``
#: satisfies it, and so does a test factory bound to a rolled-back connection.
SessionFactory = Callable[[], AsyncSession]


@runtime_checkable
class NavigatorEngine(EligibilityEngine, Protocol):
    """The engine surface the graph needs, which is wider than the contract's.

    ``app.schemas.contracts.EligibilityEngine`` declares ``evaluate``,
    ``evaluate_many`` and ``affected_programs``. The graph also needs the two
    methods ``RuleEngine`` really implements: ``reevaluate_affected``, without
    which a clarification answer re-evaluates and re-bills every program, and
    ``missing_field_questions``, which is the only deterministic source of the
    targeted clarification questions.

    Declared here rather than by editing the frozen contract. See the handoff
    note asking for these two to be added to ``EligibilityEngine``, after which
    this Protocol collapses into an alias.
    """

    def reevaluate_affected(
        self,
        previous: Sequence[EligibilityEvaluation],
        profile: UserProfile,
        changed_fields: Sequence[str],
        *,
        candidates: Sequence[RetrievedProgram] = (),
        now: datetime | None = None,
    ) -> list[EligibilityEvaluation]: ...

    def missing_field_questions(
        self,
        evaluations: Sequence[EligibilityEvaluation],
        *,
        run_id: UUID | None = None,
        limit: int = MAX_CLARIFICATION_QUESTIONS,
    ) -> list[ClarificationQuestion]: ...


class ProfileAssembler(Protocol):
    """``app.services.profile_assembly.assemble_profile``'s signature."""

    def __call__(
        self,
        extracted: ExtractedProfile | None,
        *,
        base: UserProfile | None = None,
        prefilled: ProfileDraft | None = None,
        answers: Sequence[ClarificationAnswer] = (),
        min_confidence: float = 0.5,
    ) -> tuple[UserProfile, list[str]]: ...


class PlanBuilder(Protocol):
    """``app.services.plan_service.build_plan``'s signature."""

    def __call__(
        self,
        run_id: UUID,
        matches: Sequence[ProgramMatch],
        draft: ActionPlanDraft,
        *,
        version: int = 1,
        now: datetime | None = None,
    ) -> ActionPlan: ...


def _default_retriever(session: AsyncSession) -> Retriever:
    return HybridRetriever.from_session(session)


def _default_verifier(session: AsyncSession) -> CitationVerifier:
    return SourceCitationVerifier(SqlProgramRepository(session), session=session)


@dataclass(frozen=True, slots=True, eq=False)
class WorkflowDeps:
    """Everything the graph needs from the outside world.

    Frozen, so a compiled graph cannot have its dependencies swapped under it
    at runtime, and hashable-by-identity, which is what lets
    :func:`~app.workflows.graph.get_compiled_graph` cache one compiled instance
    per dependency set.
    """

    #: The model seam. Both the live client and the deterministic
    #: implementation are first-class; nothing here cares which one it has.
    gemini: GeminiPort
    #: The deterministic rule engine. Stateless, so one instance is shared.
    engine: NavigatorEngine = field(default_factory=RuleEngine)

    #: Opens a session per concurrent task. ``None`` means "no database", which
    #: is legal for a graph run that only needs the model and in-memory fakes.
    session_factory: SessionFactory | None = None
    #: Ready-made seams, for tests. Take precedence over the constructors.
    retriever: Retriever | None = None
    verifier: CitationVerifier | None = None
    retriever_for: Callable[[AsyncSession], Retriever] = _default_retriever
    verifier_for: Callable[[AsyncSession], CitationVerifier] = _default_verifier

    #: Policy functions owned by ``app.services`` once that package lands.
    assemble: ProfileAssembler = default_assemble_profile
    build_plan: PlanBuilder = default_build_plan

    settings: Settings | None = None

    #: Extraction confidence below which a value is discarded rather than used.
    min_confidence: float = 0.5
    #: Retrieval breadth. Generous on purpose: a candidate the engine later
    #: rules out costs one evaluation, a candidate never retrieved is help the
    #: user never hears about.
    max_candidates: int = 20
    #: How many matches get model-written prose. The rest keep the engine's
    #: evidence, which is complete without it.
    max_explanations: int = 8
    #: Concurrent explanation calls. Bounded so one run cannot monopolise the
    #: provider's rate limit on behalf of every other user.
    explanation_concurrency: int = 4

    @property
    def persists(self) -> bool:
        """Whether results can be written anywhere."""
        return self.session_factory is not None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A transactional session scoped to the calling task only."""
        if self.session_factory is None:
            msg = "this workflow was built without a session factory"
            raise InternalError(msg)
        async with session_scope(self.session_factory) as session:
            yield session

    @asynccontextmanager
    async def with_retriever(self) -> AsyncIterator[Retriever]:
        """The retriever, in its own session scope when one is needed."""
        if self.retriever is not None:
            yield self.retriever
            return
        async with self.session() as session:
            yield self.retriever_for(session)

    @asynccontextmanager
    async def with_verifier(self) -> AsyncIterator[CitationVerifier]:
        """The citation verifier, in its own session scope when needed."""
        if self.verifier is not None:
            yield self.verifier
            return
        async with self.session() as session:
            yield self.verifier_for(session)
