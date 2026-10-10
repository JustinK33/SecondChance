"""Workflow run schemas: the polling contract between the API and the frontend.

An AI workflow takes tens of seconds, so no HTTP request is held open for it.
``POST /api/v1/workflows`` returns a durable run id immediately and the client
polls ``GET /api/v1/workflows/{id}``.

That makes :class:`WorkflowRunDetail` the most-rendered object in the product,
and it has one hard requirement: **it must be safe and useful at every point in
a run.** Every result field is optional and defaults to empty, so a response
mid-run is a valid partial view rather than an error state. Matches appear as
soon as the engine has produced them, before explanations exist. A failure is a
user-safe :class:`UserFacingError`, never an exception string.

No field here exposes a backend concept. ``phase`` is an internal enum that the
API translates through :data:`PHASE_PROGRESS_MESSAGES`; the user sees "Finding
programs in your area", never ``RETRIEVE_PROGRAMS`` or "LangGraph interrupt".
"""

from __future__ import annotations

import uuid
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.core.errors import ErrorCode
from app.schemas.common import NonEmptyStr, PlainText, UtcDatetime
from app.schemas.enums import MatchStatus, WorkflowPhase, WorkflowStatus
from app.schemas.matching import ProgramMatch
from app.schemas.plan import ActionPlan
from app.schemas.profile import (
    MAX_NARRATIVE_CHARS,
    ClarificationAnswer,
    ClarificationQuestion,
    ProfileDraft,
    UserProfile,
)

__all__ = [
    "PHASE_PROGRESS_MESSAGES",
    "TERMINAL_STATUSES",
    "MatchRequest",
    "UserFacingError",
    "WorkflowCreateRequest",
    "WorkflowCreateResponse",
    "WorkflowResumeRequest",
    "WorkflowRunDetail",
    "WorkflowRunSummary",
    "progress_message_for",
]

#: Statuses from which a run will never advance on its own.
TERMINAL_STATUSES: Final[frozenset[WorkflowStatus]] = frozenset(
    {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
)

#: User-facing copy for each graph phase. The single place technical state is
#: translated into something a person should read, shared by the API and the
#: frontend so the two cannot disagree.
PHASE_PROGRESS_MESSAGES: Final[dict[WorkflowPhase, str]] = {
    WorkflowPhase.PENDING: "Getting started",
    WorkflowPhase.EXTRACT_PROFILE: "Reading what you told us",
    WorkflowPhase.VALIDATE_PROFILE: "Checking what we understood",
    WorkflowPhase.AWAIT_CLARIFICATION: "We need one more detail before continuing",
    WorkflowPhase.RETRIEVE_PROGRAMS: "Finding programs in your area",
    WorkflowPhase.VALIDATE_EVIDENCE: "Checking official sources",
    WorkflowPhase.CATEGORIZE_NEEDS: "Matching programs to what you need",
    WorkflowPhase.EVALUATE_ELIGIBILITY: "Comparing your situation to each program",
    WorkflowPhase.AWAIT_TARGETED_CLARIFICATION: "One question could change your results",
    WorkflowPhase.GENERATE_EXPLANATIONS: "Writing your results in plain language",
    WorkflowPhase.BUILD_ACTION_PLAN: "Building your next steps",
    WorkflowPhase.VERIFY_CITATIONS: "Double-checking every link and source",
    WorkflowPhase.MERGE_OUTPUTS: "Putting your results together",
    WorkflowPhase.PERSIST_RESULTS: "Saving your results",
    WorkflowPhase.DONE: "Your results are ready",
}

#: Ordered phases, used to render a determinate progress bar.
_PHASE_ORDER: Final[tuple[WorkflowPhase, ...]] = (
    WorkflowPhase.PENDING,
    WorkflowPhase.EXTRACT_PROFILE,
    WorkflowPhase.VALIDATE_PROFILE,
    WorkflowPhase.RETRIEVE_PROGRAMS,
    WorkflowPhase.VALIDATE_EVIDENCE,
    WorkflowPhase.CATEGORIZE_NEEDS,
    WorkflowPhase.EVALUATE_ELIGIBILITY,
    WorkflowPhase.GENERATE_EXPLANATIONS,
    WorkflowPhase.BUILD_ACTION_PLAN,
    WorkflowPhase.VERIFY_CITATIONS,
    WorkflowPhase.MERGE_OUTPUTS,
    WorkflowPhase.PERSIST_RESULTS,
    WorkflowPhase.DONE,
)


def progress_message_for(status: WorkflowStatus, phase: WorkflowPhase) -> str:
    """The sentence to show a user for a given run state.

    Terminal statuses override the phase message, because a run that failed
    during ``GENERATE_EXPLANATIONS`` must not keep telling the user we are
    writing their results.
    """
    if status is WorkflowStatus.FAILED:
        return "We could not finish checking your programs"
    if status is WorkflowStatus.CANCELLED:
        return "This check was cancelled"
    if status is WorkflowStatus.COMPLETED:
        return PHASE_PROGRESS_MESSAGES[WorkflowPhase.DONE]
    if status is WorkflowStatus.INTERRUPTED:
        return "We paused this check and can pick up where we left off"
    if status is WorkflowStatus.QUEUED:
        return PHASE_PROGRESS_MESSAGES[WorkflowPhase.PENDING]
    return PHASE_PROGRESS_MESSAGES[phase]


def _progress_fraction(phase: WorkflowPhase) -> float:
    try:
        index = _PHASE_ORDER.index(phase)
    except ValueError:
        # The two AWAIT_* phases are off the happy path; report the position of
        # the node that will run once the user answers.
        pending = {
            WorkflowPhase.AWAIT_CLARIFICATION: WorkflowPhase.VALIDATE_PROFILE,
            WorkflowPhase.AWAIT_TARGETED_CLARIFICATION: WorkflowPhase.EVALUATE_ELIGIBILITY,
        }
        index = _PHASE_ORDER.index(pending[phase])
    return round(index / (len(_PHASE_ORDER) - 1), 3)


class UserFacingError(BaseModel):
    """A failure described in terms a nontechnical person can act on.

    Built from an :class:`~app.core.errors.AppError`. Carries the machine code
    so the client can branch, and never carries driver text, prompts or stack
    frames.
    """

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: PlainText = Field(description="Safe to render verbatim.")
    is_retryable: bool = Field(
        description="Whether resuming or retrying this run could plausibly succeed."
    )
    retry_hint: str | None = Field(
        default=None,
        max_length=200,
        description='What the user can do, e.g. "Try again - your answers are saved".',
    )
    occurred_at: UtcDatetime


class WorkflowCreateRequest(BaseModel):
    """Body of ``POST /api/v1/workflows``.

    ``idempotency_key`` is how a double-tapped submit button or a retried
    request stops producing two runs and two plans. The server stores the key
    per user and returns the original run.
    """

    model_config = ConfigDict(extra="forbid")

    narrative: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=10, max_length=MAX_NARRATIVE_CHARS),
    ]
    prefilled: ProfileDraft | None = None
    idempotency_key: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=64)] | None
    ) = None


class WorkflowCreateResponse(BaseModel):
    """Immediate ``202`` response: the durable id to poll."""

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    status: WorkflowStatus
    phase: WorkflowPhase
    progress_message: NonEmptyStr
    poll_url: NonEmptyStr = Field(description="Relative URL the client should poll.")
    created_at: UtcDatetime
    deduplicated: bool = Field(
        default=False,
        description="True when an idempotency_key matched an existing run and no new run was made.",
    )


class WorkflowRunSummary(BaseModel):
    """List-view shape for a user's run history."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    status: WorkflowStatus
    phase: WorkflowPhase
    progress_message: NonEmptyStr
    created_at: UtcDatetime
    updated_at: UtcDatetime
    completed_at: UtcDatetime | None = None
    match_count: int = Field(default=0, ge=0)
    potential_match_count: int = Field(default=0, ge=0)
    plan_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> Self:
        if self.potential_match_count > self.match_count:
            msg = "potential_match_count cannot exceed match_count"
            raise ValueError(msg)
        return self


class WorkflowRunDetail(BaseModel):
    """Full run view. Safe and useful at every point in the run.

    Partial-result contract:

    * ``matches`` fills in as soon as the engine finishes, before explanations.
    * ``plan`` stays ``None`` until the plan node completes.
    * ``clarification_questions`` is non-empty only while
      ``status is AWAITING_INPUT``.
    * ``error`` is set only on ``FAILED``, and results gathered before the
      failure are still returned, so a late failure degrades rather than
      discards.
    """

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    status: WorkflowStatus
    phase: WorkflowPhase
    progress_message: NonEmptyStr
    progress_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Determinate progress for the UI."
    )

    profile: UserProfile | None = Field(
        default=None, description="What we currently understand, for the 'edit this' affordance."
    )
    clarification_questions: list[ClarificationQuestion] = Field(default_factory=list)
    matches: list[ProgramMatch] = Field(default_factory=list)
    plan: ActionPlan | None = None
    limitations: list[PlainText] = Field(
        default_factory=list,
        description="Run-level caveats, e.g. a stale source or a program with no encoded rules.",
    )
    error: UserFacingError | None = None

    created_at: UtcDatetime
    updated_at: UtcDatetime
    completed_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def _validate_state_consistency(self) -> Self:
        if self.status is WorkflowStatus.AWAITING_INPUT and not self.clarification_questions:
            msg = "AWAITING_INPUT requires at least one clarification question"
            raise ValueError(msg)
        if self.status is not WorkflowStatus.AWAITING_INPUT and self.clarification_questions:
            msg = (
                f"clarification questions must not be returned while status is {self.status.value}"
            )
            raise ValueError(msg)
        if self.status is WorkflowStatus.FAILED and self.error is None:
            msg = "FAILED requires a user-facing error"
            raise ValueError(msg)
        if self.status is not WorkflowStatus.FAILED and self.error is not None:
            msg = f"error must be null while status is {self.status.value}"
            raise ValueError(msg)
        if self.status in TERMINAL_STATUSES and self.completed_at is None:
            msg = f"{self.status.value} requires completed_at"
            raise ValueError(msg)
        return self

    @property
    def potential_matches(self) -> list[ProgramMatch]:
        return [m for m in self.matches if m.status is MatchStatus.POTENTIAL_MATCH]

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_summary(self) -> WorkflowRunSummary:
        return WorkflowRunSummary(
            id=self.id,
            status=self.status,
            phase=self.phase,
            progress_message=self.progress_message,
            created_at=self.created_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            match_count=len(self.matches),
            potential_match_count=len(self.potential_matches),
            plan_id=self.plan.id if self.plan else None,
        )

    @classmethod
    def progress_for(cls, status: WorkflowStatus, phase: WorkflowPhase) -> tuple[str, float]:
        """Message and fraction for a run state. Used by the API serialiser."""
        return progress_message_for(status, phase), _progress_fraction(phase)


class WorkflowResumeRequest(BaseModel):
    """Body of ``POST /api/v1/workflows/{id}/resume``.

    An empty ``answers`` list is valid and means "continue with what you have":
    the run proceeds and the unanswered fields stay unknown, leaving affected
    programs at ``MORE_INFO_NEEDED``.
    """

    model_config = ConfigDict(extra="forbid")

    answers: list[ClarificationAnswer] = Field(default_factory=list, max_length=10)
    idempotency_key: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=64)] | None
    ) = None

    @model_validator(mode="after")
    def _unique_question_ids(self) -> Self:
        ids = [answer.question_id for answer in self.answers]
        if len(set(ids)) != len(ids):
            msg = "each question may be answered at most once"
            raise ValueError(msg)
        return self


class MatchRequest(BaseModel):
    """Body of ``POST /api/v1/matches``: re-evaluate after a profile edit.

    Starts a new run that reuses the source run's profile with ``changes``
    applied, and re-evaluates only the programs whose rules actually read a
    changed field (see
    :meth:`app.schemas.contracts.EligibilityEngine.affected_programs`).
    Unaffected matches and their explanations are copied forward, so editing
    household size does not re-bill every program's explanation.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID = Field(description="The run whose profile is being edited.")
    changes: ProfileDraft = Field(description="Fields the user changed. ``None`` means unchanged.")
    idempotency_key: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=8, max_length=64)] | None
    ) = None

    @model_validator(mode="after")
    def _require_a_change(self) -> Self:
        if not self.changes.set_fields():
            msg = "changes must contain at least one field"
            raise ValueError(msg)
        return self
