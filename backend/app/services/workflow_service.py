"""Workflow runs: create, poll, resume, cancel and selectively re-evaluate.

This module is the boundary between an HTTP request and a background run. It
owns three things that are easy to get wrong.

**Idempotency.** ``workflow_runs`` has ``UNIQUE (user_id, idempotency_key)``,
and :meth:`WorkflowRunRepository.create` inserts with
``ON CONFLICT DO NOTHING``, so a replayed key returns the *original* run rather
than creating a second one. The uniqueness is enforced by the database, not by
a read-then-write in Python, because two concurrent double-taps would both read
"no existing run" and both insert. A key reused with a **different** narrative
is a 409: returning the first run's results for a different question would be
answering something the user did not ask.

**Polling that is useful at every moment.** :func:`get_run` assembles a
``WorkflowRunDetail`` from the row, the stored profile, the matches written so
far and the newest plan version. Every result field is optional, so a response
mid-run is a valid partial view: matches appear as soon as the engine has
produced them, before any prose exists.

**Selective re-evaluation.** :func:`reevaluate` is deliberately deterministic
and makes no model call at all. It takes the source run's profile, applies the
user's edits, asks the engine which programs could possibly read a changed
field, re-runs only those, and copies every unaffected match forward with the
explanation it already had. Editing household size must not re-bill twenty
explanations, and the twenty sentences the user already read must not change
underneath them.

Ownership is checked on every call, and a run belonging to another user is
reported as 404 rather than 403 - the API does not confirm that an id it will
not serve actually exists.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Final

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError, WorkflowFailedError
from app.core.logging import get_logger
from app.db import models
from app.db.repositories import (
    AuditAction,
    AuditRepository,
    MatchRepository,
    PlanRepository,
    ProfileRepository,
    SqlProgramRepository,
    WorkflowRunRepository,
    hash_narrative,
)
from app.eligibility.engine import RuleEngine
from app.retrieval.search import HybridRetriever
from app.schemas import (
    ClarificationQuestion,
    MatchRequest,
    Page,
    ProgramMatch,
    RetrievedProgram,
    UserFacingError,
    UserProfile,
    WorkflowCreateRequest,
    WorkflowCreateResponse,
    WorkflowPhase,
    WorkflowResumeRequest,
    WorkflowRunDetail,
    WorkflowRunSummary,
    WorkflowStatus,
    utcnow,
)
from app.schemas.workflow import TERMINAL_STATUSES
from app.services.plan_service import draft_from_plan, save_plan_version
from app.services.profile_assembly import assemble_profile
from app.services.workflow_runner import WorkflowRunner, get_runner
from app.workflows.assembly import build_matches

__all__ = [
    "cancel_run",
    "create_run",
    "get_run",
    "list_runs",
    "poll_url_for",
    "reevaluate",
    "resume_run",
]

_log = get_logger(__name__)

#: Statuses that can be resumed by answering questions. Anything else is a 409:
#: the run is either finished or still working.
_RESUMABLE: Final[frozenset[WorkflowStatus]] = frozenset({WorkflowStatus.AWAITING_INPUT})


def _is_continuable(status: WorkflowStatus, error: dict[str, Any] | None) -> bool:
    """Whether a run can be picked up where it stopped, without answers.

    ``INTERRUPTED`` always can: it means a process died or a deploy happened,
    and the last completed superstep is checkpointed.

    ``FAILED`` can when the recorded failure is retryable - a timeout, a
    provider outage. ``docs/INTERFACES.md`` says resume on anything other than
    ``AWAITING_INPUT`` is a 409, and this narrows that to "anything that is not
    resumable", because the alternative is telling a user whose run timed out
    through no fault of their own to start again from the beginning when the
    checkpoint of their work is right there. Recorded as a contract delta in
    the handoff notes.
    """
    if status is WorkflowStatus.INTERRUPTED:
        return True
    if status is not WorkflowStatus.FAILED or not error:
        return False
    return bool(error.get("is_retryable"))


#: The phase a resumed run is about to enter, by the phase it paused in. A user
#: who has just answered must not keep seeing "one question could change your
#: results" while the graph does the work their answer unblocked.
_PHASE_AFTER_PAUSE: Final[dict[WorkflowPhase, WorkflowPhase]] = {
    WorkflowPhase.AWAIT_CLARIFICATION: WorkflowPhase.VALIDATE_PROFILE,
    WorkflowPhase.AWAIT_TARGETED_CLARIFICATION: WorkflowPhase.EVALUATE_ELIGIBILITY,
}

#: Said when a re-evaluation reuses the wording from the previous run.
_CARRIED_FORWARD = (
    "Your next steps were carried over from your previous check and were not rewritten."
)


def poll_url_for(run_id: uuid.UUID) -> str:
    """The relative URL a client should poll for a run."""
    return f"/api/v1/workflows/{run_id}"


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------
async def create_run(
    session: AsyncSession,
    user_id: str,
    request: WorkflowCreateRequest,
    *,
    request_id: str,
    runner: WorkflowRunner | None = None,
) -> WorkflowCreateResponse:
    """Insert the run row, schedule the graph, and return immediately.

    The row is committed before the background task is scheduled. That order
    matters: a task that started before the commit could finish and try to
    update a row that does not exist yet, and a client polling the id it was
    just handed would get a 404.
    """
    runs = WorkflowRunRepository(session)
    row, created = await runs.create(
        user_id,
        narrative=request.narrative,
        idempotency_key=request.idempotency_key,
    )

    if not created:
        if row.narrative_hash != hash_narrative(request.narrative):
            raise ConflictError(
                "That request id was already used for a different description. "
                "Start a new check instead.",
                detail={"field": "idempotency_key"},
            )
        _log.info("workflow.create.deduplicated", run_id=str(row.id))
        return _create_response(row, deduplicated=True)

    await AuditRepository(session).record(
        user_id,
        AuditAction.RUN_CREATED,
        resource_type="workflow_run",
        resource_id=row.id,
        run_id=row.id,
        request_id=request_id,
        metadata={"has_prefilled": request.prefilled is not None},
    )
    await session.commit()

    (runner or get_runner()).schedule_run(
        run_id=row.id,
        user_id=user_id,
        request_id=request_id,
        narrative=request.narrative,
        prefilled=request.prefilled,
    )
    _log.info("workflow.create.scheduled", run_id=str(row.id))
    return _create_response(row, deduplicated=False)


def _create_response(row: models.WorkflowRun, *, deduplicated: bool) -> WorkflowCreateResponse:
    status = WorkflowStatus(row.status)
    phase = WorkflowPhase(row.phase)
    message, _fraction = WorkflowRunDetail.progress_for(status, phase)
    return WorkflowCreateResponse(
        run_id=row.id,
        status=status,
        phase=phase,
        progress_message=message,
        poll_url=poll_url_for(row.id),
        created_at=row.created_at,
        deduplicated=deduplicated,
    )


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------
async def get_run(session: AsyncSession, user_id: str, run_id: uuid.UUID) -> WorkflowRunDetail:
    """The full run view, safe and useful at any point in the run."""
    row = await WorkflowRunRepository(session).get(user_id, run_id)
    if row is None:
        raise NotFoundError("We could not find that check.")
    return await _detail_for(session, user_id, row)


async def list_runs(
    session: AsyncSession, user_id: str, *, limit: int = 20, offset: int = 0
) -> Page[WorkflowRunSummary]:
    """A page of the caller's run history, newest first."""
    return await WorkflowRunRepository(session).list_summaries(user_id, limit=limit, offset=offset)


async def _detail_for(
    session: AsyncSession, user_id: str, row: models.WorkflowRun
) -> WorkflowRunDetail:
    """Assemble a ``WorkflowRunDetail`` from the row and its children.

    The status is normalised against what can actually be rendered, because
    ``WorkflowRunDetail`` validates its own consistency and a row that says
    ``AWAITING_INPUT`` with an unreadable payload would otherwise make polling
    raise instead of answering.
    """
    status = WorkflowStatus(row.status)
    phase = WorkflowPhase(row.phase)

    questions: list[ClarificationQuestion] = []
    if status is WorkflowStatus.AWAITING_INPUT:
        questions = _questions_from(row.clarification_payload)
        if not questions:
            # Nothing to ask means the run cannot be waiting on an answer. It
            # is resumable from its checkpoint, so say that instead.
            _log.warning("workflow.awaiting_without_questions", run_id=str(row.id))
            status = WorkflowStatus.INTERRUPTED

    error: UserFacingError | None = None
    if status is WorkflowStatus.FAILED:
        error = _error_from(row.error)

    matches = await MatchRepository(session).list_for_run(user_id, row.id)
    plan = await PlanRepository(session).get_latest_for_run(user_id, row.id)
    profile = await _profile_for(session, user_id, row)

    message, fraction = WorkflowRunDetail.progress_for(status, phase)
    completed_at = row.completed_at
    if status in TERMINAL_STATUSES and completed_at is None:
        completed_at = row.updated_at

    return WorkflowRunDetail(
        id=row.id,
        status=status,
        phase=phase,
        progress_message=message,
        progress_fraction=fraction,
        profile=profile,
        clarification_questions=questions,
        matches=matches,
        plan=plan,
        limitations=list(row.limitations or ()),
        error=error,
        created_at=row.created_at,
        updated_at=row.updated_at,
        completed_at=completed_at,
    )


def _questions_from(payload: dict[str, Any] | None) -> list[ClarificationQuestion]:
    """Validate the stored clarification payload into questions.

    An entry that no longer validates is dropped rather than raising: the other
    two questions are still answerable, and a schema change must not strand a
    paused run.
    """
    if not payload:
        return []
    raw = payload.get("questions")
    if not isinstance(raw, list):
        return []
    questions: list[ClarificationQuestion] = []
    for entry in raw:
        try:
            questions.append(ClarificationQuestion.model_validate(entry))
        except ValueError:
            _log.warning("workflow.clarification.question_dropped")
    return questions


def _error_from(payload: dict[str, Any] | None) -> UserFacingError:
    """The stored user-facing error, or a vetted generic one."""
    if payload:
        try:
            return UserFacingError.model_validate(payload)
        except ValueError:
            _log.warning("workflow.error_payload_invalid")

    failure = WorkflowFailedError()
    return UserFacingError(
        code=failure.code,
        message=failure.message,
        is_retryable=failure.is_retryable,
        retry_hint="You can run this again without re-entering anything.",
        occurred_at=utcnow(),
    )


async def _profile_for(
    session: AsyncSession, user_id: str, row: models.WorkflowRun
) -> UserProfile | None:
    """The profile this run was evaluated against, if one was stored."""
    profiles = ProfileRepository(session)
    if row.profile_id is not None:
        snapshot = await profiles.get_by_id(user_id, row.profile_id)
        if snapshot is not None:
            return snapshot.profile
    return None


# ---------------------------------------------------------------------------
# resume and cancel
# ---------------------------------------------------------------------------
async def resume_run(
    session: AsyncSession,
    user_id: str,
    run_id: uuid.UUID,
    request: WorkflowResumeRequest,
    *,
    request_id: str,
    runner: WorkflowRunner | None = None,
) -> WorkflowRunDetail:
    """Answer a paused run's questions and let it continue.

    An empty ``answers`` list is a real answer - "continue with what you have" -
    and leaves the unanswered fields unknown, which keeps the programs they
    block at ``MORE_INFO_NEEDED`` rather than discouraging anyone.
    """
    runs = WorkflowRunRepository(session)
    row = await runs.get(user_id, run_id)
    if row is None:
        raise NotFoundError("We could not find that check.")

    status = WorkflowStatus(row.status)
    if _is_continuable(status, row.error):
        # Stopped without asking anything - a restart, a deploy, or a timeout
        # that the checkpoint survived. Replaying the uncommitted superstep is
        # the right move and needs no answers, so the answers in the body are
        # deliberately ignored rather than rejected.
        await runs.mark_running(user_id, run_id, phase=WorkflowPhase(row.phase))
        await session.commit()
        (runner or get_runner()).schedule_continue(
            run_id=run_id, user_id=user_id, request_id=request_id
        )
        refreshed = await runs.get(user_id, run_id)
        return await _detail_for(session, user_id, refreshed or row)

    if status not in _RESUMABLE:
        raise ConflictError(
            "This check is not waiting for an answer right now.",
            detail={"status": status.value},
        )

    paused_phase = WorkflowPhase(row.phase)
    next_phase = _PHASE_AFTER_PAUSE.get(paused_phase, WorkflowPhase.VALIDATE_PROFILE)
    await runs.mark_running(user_id, run_id, phase=next_phase)
    await AuditRepository(session).record(
        user_id,
        AuditAction.RUN_RESUMED,
        resource_type="workflow_run",
        resource_id=run_id,
        run_id=run_id,
        request_id=request_id,
        metadata={
            "answer_count": len(request.answers),
            "skipped_count": sum(1 for answer in request.answers if answer.skipped),
        },
    )
    await session.commit()

    (runner or get_runner()).schedule_resume(
        run_id=run_id,
        user_id=user_id,
        request_id=request_id,
        answers=request.answers,
        entry_phase=next_phase,
    )
    _log.info("workflow.resume.scheduled", run_id=str(run_id), answers=len(request.answers))
    refreshed = await runs.get(user_id, run_id)
    return await _detail_for(session, user_id, refreshed or row)


async def cancel_run(
    session: AsyncSession,
    user_id: str,
    run_id: uuid.UUID,
    *,
    request_id: str | None = None,
    runner: WorkflowRunner | None = None,
) -> WorkflowRunDetail:
    """Stop a run the user no longer wants."""
    runs = WorkflowRunRepository(session)
    row = await runs.get(user_id, run_id)
    if row is None:
        raise NotFoundError("We could not find that check.")
    status = WorkflowStatus(row.status)
    if status in TERMINAL_STATUSES:
        raise ConflictError("This check has already finished.", detail={"status": status.value})

    # Status first, then the cancellation: a task being cancelled cannot
    # reliably finish a database write, so the truthful status has to be
    # recorded by the request that asked for it.
    await runs.mark_cancelled(user_id, run_id)
    await AuditRepository(session).record(
        user_id,
        AuditAction.RUN_CANCELLED,
        resource_type="workflow_run",
        resource_id=run_id,
        run_id=run_id,
        request_id=request_id,
    )
    await session.commit()
    (runner or get_runner()).cancel(run_id)

    refreshed = await runs.get(user_id, run_id)
    return await _detail_for(session, user_id, refreshed or row)


# ---------------------------------------------------------------------------
# selective re-evaluation
# ---------------------------------------------------------------------------
async def reevaluate(
    session: AsyncSession,
    user_id: str,
    request: MatchRequest,
    *,
    request_id: str,
) -> WorkflowCreateResponse:
    """Re-evaluate a profile edit deterministically, with no model call.

    Returns a run that is already ``COMPLETED``, so the client's first poll
    gets results. That is the point of this endpoint: a profile edit must feel
    instant, and nothing it needs is model-generated. The rule engine decides
    every status, exactly as it does in the full workflow; the only thing
    carried over from the source run is language the user has already read.
    """
    runs = WorkflowRunRepository(session)
    source = await runs.get(user_id, request.run_id)
    if source is None:
        raise NotFoundError("We could not find the check you are editing.")

    base = await _profile_for(session, user_id, source)
    if base is None:
        raise ConflictError(
            "We do not have the information from that check any more. Start a new one.",
            detail={"field": "run_id"},
        )

    profile, changed_fields = assemble_profile(None, base=base, prefilled=request.changes)
    if not changed_fields:
        raise ConflictError(
            "Nothing changed, so there is nothing to re-check.", detail={"field": "changes"}
        )

    previous = await MatchRepository(session).list_for_run(user_id, request.run_id)
    if not previous:
        # Nothing to re-evaluate selectively. The source run has not produced
        # results yet, so the honest answer is "finish that one first" rather
        # than a new run with an empty result set.
        raise ConflictError(
            "That check has not finished yet, so there is nothing to re-check. "
            "Finish it first, or start a new one.",
            detail={"field": "run_id", "status": WorkflowStatus(source.status).value},
        )
    engine = RuleEngine()
    retriever = HybridRetriever(SqlProgramRepository(session))

    program_ids = [match.program.id for match in previous]
    candidates = await retriever.retrieve_for_programs(program_ids)
    affected = set(engine.affected_programs(changed_fields, candidates))

    # The narrative is never stored, only hashed, so a re-evaluation run gets a
    # synthetic, non-personal marker naming the run it came from. That keeps
    # ``narrative_hash`` meaningful for idempotency without inventing text the
    # user never wrote.
    row, created = await runs.create(
        user_id,
        narrative=f"reevaluation-of-{request.run_id}",
        idempotency_key=request.idempotency_key,
        status=WorkflowStatus.RUNNING,
        phase=WorkflowPhase.EVALUATE_ELIGIBILITY,
    )
    if not created:
        _log.info("workflow.reevaluate.deduplicated", run_id=str(row.id))
        return _create_response(row, deduplicated=True)

    matches = _reevaluated_matches(
        engine=engine,
        profile=profile,
        candidates=candidates,
        previous=previous,
        affected=affected,
        changed_fields=changed_fields,
    )

    profile_row = await ProfileRepository(session).save_new_version(
        user_id, profile, changed_fields=changed_fields
    )
    await runs.attach_profile(user_id, row.id, profile_row.id)
    await MatchRepository(session).upsert_many(user_id, row.id, matches)

    source_plan = await PlanRepository(session).get_latest_for_run(user_id, request.run_id)
    limitations: list[str] = []
    if source_plan is not None:
        await save_plan_version(
            session,
            user_id,
            row.id,
            matches,
            draft_from_plan(source_plan, matches),
            request_id=request_id,
        )
        limitations.append(_CARRIED_FORWARD)

    await runs.add_limitations(user_id, row.id, limitations)
    await runs.mark_completed(user_id, row.id)
    await AuditRepository(session).record(
        user_id,
        AuditAction.MATCHES_PERSISTED,
        resource_type="workflow_run",
        resource_id=row.id,
        run_id=row.id,
        request_id=request_id,
        metadata={
            "source_run_id": str(request.run_id),
            "changed_field_count": len(changed_fields),
            "reevaluated": len(affected),
            "carried_forward": len(matches) - len(affected),
        },
    )
    await session.commit()

    _log.info(
        "workflow.reevaluated",
        run_id=str(row.id),
        reevaluated=len(affected),
        carried_forward=max(0, len(matches) - len(affected)),
        changed_fields=len(changed_fields),
    )
    refreshed = await runs.get(user_id, row.id)
    return _create_response(refreshed or row, deduplicated=False)


def _reevaluated_matches(
    *,
    engine: RuleEngine,
    profile: UserProfile,
    candidates: Sequence[RetrievedProgram],
    previous: Sequence[ProgramMatch],
    affected: set[uuid.UUID],
    changed_fields: Sequence[str],
) -> list[ProgramMatch]:
    """Re-run the engine for affected programs; carry the rest forward.

    The carried-forward half keeps its explanation. The re-evaluated half loses
    its prose deliberately: the status may have moved, and prose written for
    the old status would now be wrong. The engine's own evidence is complete
    without it, and re-wording it would mean a model call this endpoint
    promises not to make.
    """
    evaluations = engine.reevaluate_affected(
        [match.evaluation for match in previous],
        profile,
        list(changed_fields),
        candidates=candidates,
    )
    explanations = {
        match.evaluation.program_slug: match.explanation
        for match in previous
        if match.explanation is not None and match.program.id not in affected
    }
    return build_matches(evaluations, candidates, explanations=explanations)
