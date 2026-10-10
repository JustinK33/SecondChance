"""Workflow run persistence: the durable identity of one analysis.

The run row is created before the graph starts and outlives it, which is what
lets ``POST /api/v1/workflows`` return in milliseconds and the client poll for
the result.

**The narrative never lands here.** ``create`` takes a narrative and hashes it
on the spot; the text itself goes into the LangGraph checkpoint for the run and
nowhere else. The hash is what makes idempotency checkable - a key reused with
a *different* body is a conflict rather than a duplicate - without the most
sensitive field in the product being readable from a table that support
tooling, reporting and future joins will all touch.

Idempotency is enforced by ``UNIQUE (user_id, idempotency_key)``, not by
check-then-insert: two taps on a submit button arrive concurrently, so the
check has to be the constraint. :meth:`WorkflowRunRepository.create` therefore
attempts the insert and resolves the conflict, rather than looking first.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.logging import get_logger
from app.db import models
from app.db.repositories.base import (
    UserScopedRepository,
    count_of,
    dedupe,
    normalise_page,
)
from app.schemas import (
    MatchStatus,
    Page,
    UserFacingError,
    WorkflowPhase,
    WorkflowRunSummary,
    WorkflowStatus,
    progress_message_for,
    utcnow,
)

__all__ = ["WorkflowRunRepository", "hash_narrative"]

_log = get_logger(__name__)


def hash_narrative(narrative: str) -> str:
    """SHA-256 of a submitted narrative, as 64 lowercase hex characters.

    Whitespace-normalised first, so that a resubmission differing only in a
    trailing newline is recognised as the same request. Not a privacy control
    on its own - a short narrative is guessable - but the point is that the
    stored value is not *readable*, and nothing in the product ever needs to
    reverse it.
    """
    return hashlib.sha256(" ".join(narrative.split()).encode("utf-8")).hexdigest()


class WorkflowRunRepository(UserScopedRepository):
    """Create, read and advance workflow runs for one user."""

    __slots__ = ()

    async def create(
        self,
        user_id: str,
        *,
        narrative: str,
        idempotency_key: str | None = None,
        profile_id: uuid.UUID | None = None,
        status: WorkflowStatus = WorkflowStatus.QUEUED,
        phase: WorkflowPhase = WorkflowPhase.PENDING,
    ) -> tuple[models.WorkflowRun, bool]:
        """Create a run, or return the existing one for a repeated key.

        :returns: ``(run, created)``. ``created`` is ``False`` when an
            ``idempotency_key`` matched an existing run, which is what the API
            reports as ``deduplicated: true``.

        A key reused with a *different* narrative is not deduplicated: the
        returned run's ``narrative_hash`` will differ from the one computed
        here, and the caller is expected to turn that into a ``409 conflict``
        rather than quietly returning results for the wrong question.
        """
        run_id = uuid.uuid4()
        narrative_hash = hash_narrative(narrative)
        values: dict[str, Any] = {
            "id": run_id,
            "user_id": user_id,
            "status": status.value,
            "phase": phase.value,
            "idempotency_key": idempotency_key,
            # The checkpointer keys threads by ``str(run_id)``; stored so the
            # deletion flow does not have to re-derive the convention.
            "thread_id": str(run_id),
            "narrative_hash": narrative_hash,
            "profile_id": profile_id,
        }

        if idempotency_key is None:
            row = models.WorkflowRun(**values)
            self.session.add(row)
            await self.session.flush()
            return row, True

        statement = (
            pg_insert(models.WorkflowRun)
            .values(**values)
            .on_conflict_do_nothing(
                index_elements=[
                    models.WorkflowRun.user_id,
                    models.WorkflowRun.idempotency_key,
                ]
            )
            .returning(models.WorkflowRun.id)
        )
        inserted_id = (await self.session.execute(statement)).scalar_one_or_none()
        if inserted_id is None:
            existing = await self.get_by_idempotency_key(user_id, idempotency_key)
            if existing is None:  # pragma: no cover - only if the row vanished mid-statement
                msg = "idempotency key conflicted but the original run could not be read back"
                raise RuntimeError(msg)
            _log.info("workflow.run.deduplicated", run_id=str(existing.id))
            return existing, False
        row = await self.session.get_one(models.WorkflowRun, inserted_id)
        return row, True

    async def get(self, user_id: str, run_id: uuid.UUID) -> models.WorkflowRun | None:
        """One run, scoped to its owner. ``None`` for somebody else's run.

        Returning ``None`` rather than raising is deliberate: the API turns it
        into a 404, so a probe cannot distinguish "does not exist" from
        "exists and is not yours".
        """
        statement = select(models.WorkflowRun).where(
            models.WorkflowRun.id == run_id, models.WorkflowRun.user_id == user_id
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def get_by_thread_id(self, user_id: str, thread_id: str) -> models.WorkflowRun | None:
        statement = select(models.WorkflowRun).where(
            models.WorkflowRun.thread_id == thread_id,
            models.WorkflowRun.user_id == user_id,
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def get_by_idempotency_key(self, user_id: str, key: str) -> models.WorkflowRun | None:
        statement = select(models.WorkflowRun).where(
            models.WorkflowRun.user_id == user_id,
            models.WorkflowRun.idempotency_key == key,
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def thread_ids_for_user(self, user_id: str) -> list[str]:
        """Every checkpoint thread belonging to this user.

        Used by the account-deletion flow: the checkpoint tables have no
        ``user_id`` of their own, so this is the only link between a user and
        the rows holding their narrative.
        """
        statement = select(models.WorkflowRun.thread_id).where(
            models.WorkflowRun.user_id == user_id
        )
        return list((await self.session.scalars(statement)).all())

    async def list_summaries(
        self, user_id: str, *, limit: int = 20, offset: int = 0
    ) -> Page[WorkflowRunSummary]:
        """A page of the user's run history, newest first.

        The match counts and the plan id are aggregated in SQL rather than by
        loading each run's children, because this is the history screen and N+1
        queries there are felt immediately.
        """
        limit, offset = normalise_page(limit, offset)
        runs = models.WorkflowRun

        match_counts = (
            select(
                models.ProgramMatch.run_id.label("run_id"),
                func.count().label("match_count"),
                func.count()
                .filter(models.ProgramMatch.status == MatchStatus.POTENTIAL_MATCH.value)
                .label("potential_match_count"),
            )
            .where(models.ProgramMatch.user_id == user_id)
            .group_by(models.ProgramMatch.run_id)
            .subquery()
        )
        latest_plan = (
            select(
                models.ActionPlan.run_id.label("run_id"),
                func.max(models.ActionPlan.version).label("version"),
            )
            .where(models.ActionPlan.user_id == user_id)
            .group_by(models.ActionPlan.run_id)
            .subquery()
        )
        plans = models.ActionPlan.__table__.alias("latest_plan_row")

        base = select(runs).where(runs.user_id == user_id)
        total = await count_of(self.session, base)

        statement = (
            select(
                runs,
                func.coalesce(match_counts.c.match_count, 0),
                func.coalesce(match_counts.c.potential_match_count, 0),
                plans.c.id,
            )
            .outerjoin(match_counts, match_counts.c.run_id == runs.id)
            .outerjoin(latest_plan, latest_plan.c.run_id == runs.id)
            .outerjoin(
                plans,
                (plans.c.run_id == latest_plan.c.run_id)
                & (plans.c.version == latest_plan.c.version),
            )
            .where(runs.user_id == user_id)
            .order_by(runs.created_at.desc(), runs.id.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = (await self.session.execute(statement)).all()
        items = [
            WorkflowRunSummary(
                id=run.id,
                status=WorkflowStatus(run.status),
                phase=WorkflowPhase(run.phase),
                progress_message=progress_message_for(
                    WorkflowStatus(run.status), WorkflowPhase(run.phase)
                ),
                created_at=run.created_at,
                updated_at=run.updated_at,
                completed_at=run.completed_at,
                match_count=match_count,
                potential_match_count=potential_count,
                plan_id=plan_id,
            )
            for run, match_count, potential_count, plan_id in rows
        ]
        return Page(items=items, total=total, limit=limit, offset=offset)

    # ------------------------------------------------------------ transitions
    async def mark_running(
        self, user_id: str, run_id: uuid.UUID, *, phase: WorkflowPhase
    ) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.status = WorkflowStatus.RUNNING.value
        row.phase = phase.value
        if row.started_at is None:
            row.started_at = utcnow()
        row.clarification_payload = None
        row.error = None
        await self.session.flush()
        return row

    async def set_phase(
        self, user_id: str, run_id: uuid.UUID, phase: WorkflowPhase
    ) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.phase = phase.value
        await self.session.flush()
        return row

    async def mark_awaiting_input(
        self,
        user_id: str,
        run_id: uuid.UUID,
        *,
        phase: WorkflowPhase,
        clarification_payload: dict[str, Any],
    ) -> models.WorkflowRun | None:
        """Park a run on a clarification question set.

        The payload is stored so a poll can render the questions without
        reading the checkpoint. It holds questions and option labels - editorial
        text - and never the user's answers.
        """
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.status = WorkflowStatus.AWAITING_INPUT.value
        row.phase = phase.value
        row.clarification_payload = clarification_payload
        await self.session.flush()
        return row

    async def mark_completed(
        self,
        user_id: str,
        run_id: uuid.UUID,
        *,
        limitations: Sequence[str] = (),
        model_usage: Sequence[dict[str, Any]] = (),
    ) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.status = WorkflowStatus.COMPLETED.value
        row.phase = WorkflowPhase.DONE.value
        row.completed_at = utcnow()
        row.clarification_payload = None
        row.error = None
        row.limitations = dedupe([*row.limitations, *limitations])
        if model_usage:
            row.model_usage = [*row.model_usage, *model_usage]
        await self.session.flush()
        return row

    async def mark_failed(
        self,
        user_id: str,
        run_id: uuid.UUID,
        error: UserFacingError,
        *,
        limitations: Sequence[str] = (),
    ) -> models.WorkflowRun | None:
        """Record a failure as user-safe structured data.

        ``error`` is already a :class:`~app.schemas.workflow.UserFacingError`,
        so what reaches the column is a code and plain-language copy - never a
        driver message, a prompt or a stack frame.
        """
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.status = WorkflowStatus.FAILED.value
        row.completed_at = utcnow()
        row.clarification_payload = None
        row.error = error.model_dump(mode="json")
        row.limitations = dedupe([*row.limitations, *limitations])
        await self.session.flush()
        _log.info("workflow.run.failed", run_id=str(run_id), error_code=error.code.value)
        return row

    async def mark_cancelled(self, user_id: str, run_id: uuid.UUID) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.status = WorkflowStatus.CANCELLED.value
        row.completed_at = utcnow()
        row.clarification_payload = None
        await self.session.flush()
        return row

    async def attach_profile(
        self, user_id: str, run_id: uuid.UUID, profile_id: uuid.UUID
    ) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None:
            return None
        row.profile_id = profile_id
        await self.session.flush()
        return row

    async def add_model_usage(
        self, user_id: str, run_id: uuid.UUID, usage: Sequence[dict[str, Any]]
    ) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None or not usage:
            return row
        row.model_usage = [*row.model_usage, *usage]
        await self.session.flush()
        return row

    async def add_limitations(
        self, user_id: str, run_id: uuid.UUID, limitations: Sequence[str]
    ) -> models.WorkflowRun | None:
        row = await self.get(user_id, run_id)
        if row is None or not limitations:
            return row
        row.limitations = dedupe([*row.limitations, *limitations])
        await self.session.flush()
        return row
