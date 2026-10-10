"""The background execution model for workflow runs.

``POST /api/v1/workflows`` must return in milliseconds. A run takes tens of
seconds, so the HTTP request inserts the ``workflow_runs`` row and hands the
graph to this runner, which owns everything about executing it.

The design, and why each piece is there
---------------------------------------
**Durability comes from two places, not one.** The row in ``workflow_runs`` is
what the user polls; the LangGraph checkpoint is what makes the work resumable.
They are updated in that order and never in one transaction, because they live
in different systems. The reconciliation step below is what closes the gap.

**One task per run, one session per task.** Each run executes in a tracked
``asyncio.Task`` that opens its *own* ``session_scope()``. An ``AsyncSession``
owns a single DBAPI connection and is not safe for concurrent use, and the
graph fans out, so sharing one would interleave protocol traffic and surface as
``InterfaceError: another operation is in progress`` - or, worse, as one user's
rows arriving in another user's result set.

**Admission control, not just a timeout.** A semaphore sized by
``MAX_CONCURRENT_WORKFLOWS`` bounds how many runs execute at once; the rest sit
queued with status ``QUEUED``, which is an honest thing to show a user. Without
the bound, a burst would open a session per run and exhaust the pool, taking
down the endpoints that were working fine.

**Phases are persisted as the graph advances.** The runner consumes
``graph.astream(..., stream_mode="updates")`` and writes each new phase to the
row, so polling shows real progress rather than jumping from ``PENDING`` to
``DONE``. The stream is also where per-node durations come from.

**Shutdown is graceful and honest.** On lifespan shutdown, in-flight runs are
marked ``INTERRUPTED`` *before* their tasks are cancelled, then the
cancellations are awaited. The checkpoint for the last completed superstep
survives, so the user resumes having re-typed nothing.

**Startup reconciles the liars.** A process that died mid-run left rows saying
``RUNNING`` with nobody running them. Those would poll forever, so startup
rewrites them to ``INTERRUPTED``, which is both true and resumable.

**A failure never leaks.** What reaches the row is a
:class:`~app.schemas.workflow.UserFacingError`: a stable code and vetted
plain-language copy. Never a stack trace, never a driver message, never a
provider error string.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final

from langgraph.types import Command
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.errors import InternalError, UpstreamTimeoutError
from app.core.logging import bound_request_context, get_logger, hash_user_id
from app.core.metrics import MetricsRegistry, get_metrics
from app.db import models
from app.db.repositories import (
    AuditAction,
    AuditRepository,
    ProfileRepository,
    WorkflowRunRepository,
)
from app.db.session import get_session_factory, session_scope
from app.schemas import (
    ClarificationAnswer,
    GeminiPort,
    ProfileDraft,
    UserFacingError,
    UserProfile,
    WorkflowPhase,
    WorkflowStatus,
    utcnow,
)
from app.services.ai_port import provide_ai_port
from app.workflows.checkpointer import get_checkpointer, thread_config
from app.workflows.deps import WorkflowDeps
from app.workflows.graph import get_compiled_graph, pending_interrupt
from app.workflows.nodes.base import log_failure, user_facing
from app.workflows.state import NavigatorState, build_initial_state

__all__ = [
    "RunnerSnapshot",
    "WorkflowRunner",
    "get_runner",
    "reset_runner",
    "set_runner",
]

_log = get_logger(__name__)

#: ``WorkflowPhase`` for each kind of pause, keyed by the ``kind`` the
#: clarification nodes put in their interrupt payload. ``interrupt()`` stops a
#: node *before* it returns, so the phase it would have written is never
#: committed - the pause has to be named from the payload instead.
_PHASE_BY_INTERRUPT: Final[dict[str, WorkflowPhase]] = {
    "clarification": WorkflowPhase.AWAIT_CLARIFICATION,
    "targeted_clarification": WorkflowPhase.AWAIT_TARGETED_CLARIFICATION,
}

#: Statuses a run can be reconciled out of on startup: it claimed to be
#: executing, and this process knows it is not.
_ORPHANABLE: Final[tuple[str, ...]] = (
    WorkflowStatus.RUNNING.value,
    WorkflowStatus.QUEUED.value,
)

#: How long shutdown waits for a cancelled task to unwind before giving up on
#: it. Tasks only have status writes left to do at that point.
_SHUTDOWN_GRACE_S: Final[float] = 10.0


@dataclass(frozen=True, slots=True)
class RunnerSnapshot:
    """Operational counters, for ``/internal/metrics`` and the benchmarks."""

    #: Runs holding a concurrency slot and actually executing.
    in_flight: int
    #: Tracked tasks, which is ``in_flight`` plus those still queued.
    tracked: int
    #: Runs admitted to the process but waiting for a slot.
    queued: int
    #: ``MAX_CONCURRENT_WORKFLOWS``.
    capacity: int
    scheduled: int
    completed: int
    awaiting_input: int
    failed: int
    interrupted: int
    rejected: int


class WorkflowRunner:
    """Executes workflow runs in tracked background tasks."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        gemini: GeminiPort | None = None,
        deps: WorkflowDeps | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._session_factory = session_factory
        self._gemini = gemini
        self._deps = deps
        self._metrics = metrics if metrics is not None else get_metrics()

        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_workflows)
        self._tasks: dict[uuid.UUID, asyncio.Task[None]] = {}
        self._graph: Any | None = None
        self._graph_lock = asyncio.Lock()
        self._closing = False
        self._queued = 0
        self._executing = 0
        self._counters = dict.fromkeys(
            ("scheduled", "completed", "awaiting_input", "failed", "interrupted", "rejected"), 0
        )

    # ------------------------------------------------------------- lifecycle
    async def startup(self) -> None:
        """Prepare the checkpointer and reconcile runs stranded by a restart."""
        self._closing = False
        handle = await get_checkpointer(self._settings)
        _log.info(
            "runner.started",
            checkpointer=handle.kind,
            persistent=handle.is_persistent,
            capacity=self._settings.max_concurrent_workflows,
            timeout_s=self._settings.workflow_timeout_s,
        )
        await self.reconcile_orphans()
        self._publish_gauges()

    async def shutdown(self, *, grace_s: float = _SHUTDOWN_GRACE_S) -> None:
        """Stop accepting work, record in-flight runs, then cancel them.

        The status write happens before the cancellation on purpose. A task
        that is being cancelled cannot reliably finish a database round trip -
        every ``await`` inside it re-raises ``CancelledError`` - so the only
        place a truthful status can be written is here.
        """
        self._closing = True
        in_flight = {run_id: task for run_id, task in self._tasks.items() if not task.done()}
        if in_flight:
            _log.info("runner.shutdown.interrupting", count=len(in_flight))
            await self._mark_interrupted(list(in_flight))
        for task in in_flight.values():
            task.cancel()
        if in_flight:
            await asyncio.wait(list(in_flight.values()), timeout=grace_s)
            self._counters["interrupted"] += len(in_flight)
        self._tasks.clear()
        self._publish_gauges()
        _log.info("runner.shutdown.complete", interrupted=len(in_flight))

    async def reconcile_orphans(self) -> int:
        """Rewrite runs left ``RUNNING`` or ``QUEUED`` by a dead process.

        A row claiming to be running with no task behind it is a run that will
        never finish and a poll that will never resolve. ``INTERRUPTED`` is the
        true status and it is resumable from the last checkpointed superstep,
        so the user loses at most one step of work and nothing they typed.

        Only rows untouched for twice the workflow budget are reconciled. That
        staleness window is what makes this safe with more than one web worker:
        a run genuinely executing in another process writes a phase every
        superstep, so it can never look stale, and a run whose process died
        always will.
        """
        stale_after = timedelta(seconds=self._settings.workflow_timeout_s * 2)
        cutoff = utcnow() - stale_after
        async with self._scope() as session:
            statement = (
                update(models.WorkflowRun)
                .where(
                    models.WorkflowRun.status.in_(_ORPHANABLE),
                    models.WorkflowRun.updated_at < cutoff,
                )
                .values(status=WorkflowStatus.INTERRUPTED.value, clarification_payload=None)
            )
            result = await session.execute(statement)
            count = int(result.rowcount or 0)
        if count:
            _log.warning("runner.reconciled_orphans", count=count)
        return count

    # ------------------------------------------------------------- scheduling
    def schedule_run(
        self,
        *,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        narrative: str,
        prefilled: ProfileDraft | None = None,
        profile: UserProfile | None = None,
    ) -> None:
        """Start a fresh run in the background. Returns immediately."""
        state = build_initial_state(
            run_id=run_id,
            user_id=user_id,
            request_id=request_id,
            narrative=narrative,
            prefilled=prefilled,
            profile=profile,
        )
        self._spawn(
            run_id,
            user_id,
            request_id,
            payload=state,
            kind="start",
            entry_phase=WorkflowPhase.EXTRACT_PROFILE,
        )

    def schedule_resume(
        self,
        *,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        answers: Sequence[ClarificationAnswer] = (),
        entry_phase: WorkflowPhase = WorkflowPhase.VALIDATE_PROFILE,
    ) -> None:
        """Resume a paused run with the user's answers. Returns immediately.

        The state is not passed again: it comes from the checkpoint for
        ``thread_id == str(run_id)``, which is why a different process can
        finish a run it did not start.

        ``entry_phase`` is the phase the resumed run is *about to* enter, not
        the one it paused in. A user who has just answered must not be shown
        "one question could change your results" again while the graph works.
        """
        payload: Command[Any] = Command(
            resume={"answers": [answer.model_dump(mode="json") for answer in answers]}
        )
        self._spawn(
            run_id,
            user_id,
            request_id,
            payload=payload,
            kind="resume",
            entry_phase=entry_phase,
        )

    def schedule_continue(self, *, run_id: uuid.UUID, user_id: str, request_id: str) -> None:
        """Pick up a run that stopped without asking anything.

        Replays the superstep that never committed, which is what makes an
        ``INTERRUPTED`` run - a crash, a deploy - cost one step rather than the
        whole run.
        """
        self._spawn(
            run_id,
            user_id,
            request_id,
            payload=None,
            kind="continue",
            entry_phase=WorkflowPhase.VALIDATE_PROFILE,
        )

    def cancel(self, run_id: uuid.UUID) -> bool:
        """Cancel an in-flight run. The caller owns the status write."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        _log.info("runner.cancelled", run_id=str(run_id))
        return True

    def is_running(self, run_id: uuid.UUID) -> bool:
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    async def drain(self, *, timeout_s: float | None = None) -> bool:
        """Wait until nothing is in flight. For scripts, tests and benchmarks."""
        pending = [task for task in self._tasks.values() if not task.done()]
        if not pending:
            return True
        done, _ = await asyncio.wait(pending, timeout=timeout_s)
        return len(done) == len(pending)

    def snapshot(self) -> RunnerSnapshot:
        return RunnerSnapshot(
            in_flight=self._executing,
            tracked=sum(1 for task in self._tasks.values() if not task.done()),
            queued=self._queued,
            capacity=self._settings.max_concurrent_workflows,
            scheduled=self._counters["scheduled"],
            completed=self._counters["completed"],
            awaiting_input=self._counters["awaiting_input"],
            failed=self._counters["failed"],
            interrupted=self._counters["interrupted"],
            rejected=self._counters["rejected"],
        )

    # -------------------------------------------------------------- internals
    def _spawn(
        self,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        *,
        payload: Any,
        kind: str,
        entry_phase: WorkflowPhase,
    ) -> None:
        if self._closing:
            self._counters["rejected"] += 1
            msg = "the workflow runner is shutting down"
            raise InternalError(msg)
        existing = self._tasks.get(run_id)
        if existing is not None and not existing.done():
            # Two tasks on one ``thread_id`` would interleave writes to the
            # same checkpoint. The run is already executing; saying so is
            # better than racing it.
            _log.info("runner.already_running", run_id=str(run_id))
            return

        task = asyncio.create_task(
            self._execute(
                run_id,
                user_id,
                request_id,
                payload=payload,
                kind=kind,
                entry_phase=entry_phase,
            ),
            name=f"workflow-{kind}-{run_id}",
        )
        self._tasks[run_id] = task
        self._counters["scheduled"] += 1
        task.add_done_callback(lambda finished: self._finalise(run_id, finished))
        self._publish_gauges()

    def _finalise(self, run_id: uuid.UUID, task: asyncio.Task[None]) -> None:
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)
        self._publish_gauges()
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:  # pragma: no cover - _execute catches everything
            _log.error("runner.task.crashed", run_id=str(run_id), error_type=type(error).__name__)

    async def _execute(
        self,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        *,
        payload: Any,
        kind: str,
        entry_phase: WorkflowPhase,
    ) -> None:
        """One run, from admission to a persisted terminal status."""
        started = time.perf_counter()
        self._queued += 1
        admitted = False
        try:
            async with self._semaphore:
                self._queued -= 1
                admitted = True
                self._executing += 1
                self._publish_gauges()
                if self._closing:
                    raise asyncio.CancelledError
                with bound_request_context(request_id, hash_user_id(user_id)):
                    await self._run_guarded(
                        run_id,
                        user_id,
                        request_id,
                        payload=payload,
                        kind=kind,
                        entry_phase=entry_phase,
                        started=started,
                    )
        except asyncio.CancelledError:
            # The status is written by whoever asked for the cancellation -
            # ``shutdown`` or ``workflow_service.cancel_run`` - because a task
            # being cancelled cannot finish a database round trip.
            _log.info("runner.run.cancelled", run_id=str(run_id), kind=kind)
            raise
        finally:
            if admitted:
                self._executing -= 1
            else:
                # Cancelled while still queued for a slot.
                self._queued -= 1
            self._publish_gauges()

    async def _run_guarded(
        self,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        *,
        payload: Any,
        kind: str,
        entry_phase: WorkflowPhase,
        started: float,
    ) -> None:
        await self._transition(run_id, user_id, entry_phase, running=True)
        status = WorkflowStatus.FAILED
        try:
            async with asyncio.timeout(self._settings.workflow_timeout_s):
                status = await self._drive(run_id, user_id, request_id, payload=payload)
        except asyncio.CancelledError:
            # Recorded as cancelled, not failed. The run did what it was asked
            # to do, and a cancellation in the failure histogram would make an
            # ordinary user action look like an incident.
            status = WorkflowStatus.CANCELLED
            raise
        except TimeoutError as exc:
            # The checkpoint of the last completed superstep survives, so the
            # run stays resumable with nothing re-entered.
            self._metrics.increment_timeout(operation="workflow")
            await self._fail(run_id, user_id, user_facing(UpstreamTimeoutError(cause=exc)))
            status = WorkflowStatus.FAILED
        except Exception as exc:
            # ``user_facing`` flattens anything unrecognised into the generic
            # internal error, which is right for the user and useless for an
            # operator - so the class and the traceback are logged here, where
            # the redacting pipeline can scrub them, before that happens.
            log_failure("workflow", exc, recoverable=False)
            _log.error(
                "runner.run.unhandled",
                run_id=str(run_id),
                kind=kind,
                error_type=type(exc).__name__,
                exc_info=exc,
            )
            await self._fail(run_id, user_id, user_facing(exc))
            status = WorkflowStatus.FAILED
        finally:
            duration_s = time.perf_counter() - started
            self._metrics.observe_workflow(status=status.value, duration_s=duration_s)
            _log.info(
                "runner.run.finished",
                run_id=str(run_id),
                kind=kind,
                status=status.value,
                duration_ms=round(duration_s * 1000),
            )

    async def _drive(
        self,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        *,
        payload: Any,
    ) -> WorkflowStatus:
        """Stream the graph, persisting phases, then record the outcome."""
        graph = await self._ensure_graph()
        config = thread_config(run_id)
        interrupt: dict[str, Any] | None = None
        last_phase: WorkflowPhase | None = None
        node_started = time.perf_counter()

        async for chunk in graph.astream(payload, config=config, stream_mode="updates"):
            if not isinstance(chunk, Mapping):  # pragma: no cover - defensive
                continue
            now = time.perf_counter()
            if "__interrupt__" in chunk:
                interrupt = pending_interrupt(dict(chunk)) or interrupt
                node_started = now
                continue
            for node, node_update in chunk.items():
                self._metrics.observe_workflow_node(
                    node=str(node), duration_s=max(0.0, now - node_started)
                )
                phase = node_update.get("phase") if isinstance(node_update, Mapping) else None
                if isinstance(phase, WorkflowPhase) and phase is not last_phase:
                    last_phase = phase
                    await self._transition(run_id, user_id, phase, running=False)
            node_started = now

        snapshot = await graph.aget_state(config)
        state: NavigatorState = dict(snapshot.values)  # type: ignore[assignment]
        if interrupt is None:
            pending = tuple(getattr(snapshot, "interrupts", ()) or ())
            if pending:
                interrupt = pending_interrupt({"__interrupt__": pending})

        return await self._record_outcome(
            run_id, user_id, request_id, state=state, interrupt=interrupt
        )

    async def _record_outcome(
        self,
        run_id: uuid.UUID,
        user_id: str,
        request_id: str,
        *,
        state: NavigatorState,
        interrupt: dict[str, Any] | None,
    ) -> WorkflowStatus:
        error = state.get("error")
        profile = state.get("profile")
        changed_fields = list(state.get("changed_fields") or ())

        async with self._scope() as session:
            runs = WorkflowRunRepository(session)
            if profile is not None:
                await self._attach_profile(session, user_id, run_id, profile, changed_fields)

            if isinstance(error, UserFacingError):
                await runs.mark_failed(user_id, run_id, error)
                await self._audit(session, user_id, run_id, request_id, AuditAction.RUN_FAILED)
                self._counters["failed"] += 1
                return WorkflowStatus.FAILED

            if interrupt is not None:
                phase = _PHASE_BY_INTERRUPT.get(
                    str(interrupt.get("kind")), WorkflowPhase.AWAIT_CLARIFICATION
                )
                await runs.mark_awaiting_input(
                    user_id, run_id, phase=phase, clarification_payload=interrupt
                )
                self._counters["awaiting_input"] += 1
                _log.info(
                    "runner.run.awaiting_input",
                    run_id=str(run_id),
                    phase=phase.value,
                    questions=len(interrupt.get("questions") or ()),
                )
                return WorkflowStatus.AWAITING_INPUT

            # ``persist_results`` already wrote the matches, the plan, the
            # model usage and the limitations inside its own transaction, so
            # this only closes the run out.
            await runs.mark_completed(user_id, run_id)
            await self._audit(session, user_id, run_id, request_id, AuditAction.RUN_COMPLETED)
            self._counters["completed"] += 1
        return WorkflowStatus.COMPLETED

    async def _attach_profile(
        self,
        session: AsyncSession,
        user_id: str,
        run_id: uuid.UUID,
        profile: UserProfile,
        changed_fields: Sequence[str],
    ) -> None:
        """Store the run's profile as a version and point the run at it.

        A new version is written only when the profile actually differs from
        the newest stored one. Without that check, every poll-and-resume cycle
        would append an identical version and the history screen would be
        unreadable.
        """
        profiles = ProfileRepository(session)
        latest = await profiles.get_latest(user_id)
        if latest is not None and latest.profile == profile:
            row = await profiles.get_latest_row(user_id)
            if row is not None:
                await WorkflowRunRepository(session).attach_profile(user_id, run_id, row.id)
            return
        row = await profiles.save_new_version(user_id, profile, changed_fields=changed_fields)
        await WorkflowRunRepository(session).attach_profile(user_id, run_id, row.id)

    async def _transition(
        self,
        run_id: uuid.UUID,
        user_id: str,
        phase: WorkflowPhase,
        *,
        running: bool,
    ) -> None:
        """Persist a phase (and, on entry, the ``RUNNING`` status)."""
        try:
            async with self._scope() as session:
                runs = WorkflowRunRepository(session)
                if running:
                    await runs.mark_running(user_id, run_id, phase=phase)
                else:
                    await runs.set_phase(user_id, run_id, phase)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A lost progress write must not fail a run that is otherwise
            # fine: the terminal status write is the one that matters, and the
            # user sees a slightly stale progress line rather than an error.
            _log.warning(
                "runner.transition.failed",
                run_id=str(run_id),
                phase=phase.value,
                error_type=type(exc).__name__,
            )

    async def _fail(self, run_id: uuid.UUID, user_id: str, error: UserFacingError) -> None:
        self._counters["failed"] += 1
        self._metrics.increment_error(code=error.code.value, route="workflow")
        try:
            async with self._scope() as session:
                await WorkflowRunRepository(session).mark_failed(user_id, run_id, error)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.error(
                "runner.fail_write_failed", run_id=str(run_id), error_type=type(exc).__name__
            )

    async def _mark_interrupted(self, run_ids: Sequence[uuid.UUID]) -> None:
        if not run_ids:
            return
        try:
            async with self._scope() as session:
                statement = (
                    update(models.WorkflowRun)
                    .where(models.WorkflowRun.id.in_(list(run_ids)))
                    .values(status=WorkflowStatus.INTERRUPTED.value)
                )
                await session.execute(statement)
        except Exception as exc:  # pragma: no cover - shutdown best effort
            _log.error("runner.interrupt_write_failed", error_type=type(exc).__name__)

    async def _audit(
        self,
        session: AsyncSession,
        user_id: str,
        run_id: uuid.UUID,
        request_id: str,
        action: AuditAction,
    ) -> None:
        await AuditRepository(session).record(
            user_id,
            action,
            resource_type="workflow_run",
            resource_id=run_id,
            run_id=run_id,
            request_id=request_id,
        )

    async def _ensure_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        async with self._graph_lock:
            if self._graph is None:
                handle = await get_checkpointer(self._settings)
                self._graph = get_compiled_graph(self._workflow_deps(), handle.saver)
            return self._graph

    def _workflow_deps(self) -> WorkflowDeps:
        if self._deps is not None:
            return self._deps
        self._deps = WorkflowDeps(
            gemini=self._gemini if self._gemini is not None else provide_ai_port(self._settings),
            session_factory=self._factory(),
            settings=self._settings,
        )
        return self._deps

    def _factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is not None:
            return self._session_factory
        return get_session_factory(self._settings)

    def _scope(self) -> Any:
        return session_scope(self._factory())

    def _publish_gauges(self) -> None:
        self._metrics.set_workflows_in_flight(self._executing)
        self._metrics.set_gauge(
            "workflows_queued",
            float(self._queued),
            help_text="Workflow runs admitted to the process but waiting for a concurrency slot.",
        )


# ---------------------------------------------------------------------------
# process-wide instance
# ---------------------------------------------------------------------------
_runner: WorkflowRunner | None = None


def get_runner(settings: Settings | None = None) -> WorkflowRunner:
    """The process-wide runner, created on first use.

    One per process, because the semaphore that bounds concurrency and the set
    of tracked tasks only mean anything if they are shared.
    """
    global _runner
    if _runner is None:
        _runner = WorkflowRunner(settings or get_settings())
    return _runner


def set_runner(runner: WorkflowRunner | None) -> None:
    """Install a runner. Used by the lifespan and by tests."""
    global _runner
    _runner = runner


def reset_runner() -> None:
    """Drop the process-wide runner without shutting it down."""
    global _runner
    _runner = None
