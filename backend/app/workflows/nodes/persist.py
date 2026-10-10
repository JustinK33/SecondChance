"""``persist_results``: one transaction, one session, one task.

Everything this node writes belongs to one unit of work - the matches, the plan
and its items, the run's telemetry and its caveats - so it writes them inside a
single :meth:`~app.workflows.deps.WorkflowDeps.session` scope that commits once
at the end. A half-written result is worse than none: a user polling the run
would see three of their eight programs and no plan, with no way to tell that
from the real answer.

The session is opened here and closed here. It is never the session another
node used, because this node can run while nothing else does but the rule
("one session per task, never shared") is only safe if it is unconditional.

Idempotency is a property of the repositories rather than a flag this node
sets. ``MatchRepository.upsert_many`` conflicts on ``(run_id, program_id)`` and
``PlanRepository.save`` on ``(run_id, version)`` and on each item's id, which is
``uuid5(run_id, dedup_key)``. So a resumed run that reaches this node twice
converges on the same rows and a user's completed steps survive a regeneration.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.db.repositories import MatchRepository, PlanRepository, WorkflowRunRepository
from app.schemas import WorkflowPhase
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import entering, run_id_of
from app.workflows.state import NavigatorState, StateUpdate

__all__ = ["NODE", "persist_results"]

NODE = "persist_results"

_log = get_logger(__name__)

_NOT_PERSISTED = "These results were not saved, so they will not be here if you come back later."


async def persist_results(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Write the run's results through the repositories, in one transaction.

    Failure here is not recoverable and is deliberately not caught: the results
    exist in the checkpoint, so raising lets the run be marked failed and then
    resumed, whereas swallowing the error would report success for data that
    does not exist.
    """
    update = entering(NODE)
    run_id = run_id_of(state)
    user_id = state["user_id"]
    matches = list(state.get("matches") or ())
    plan = state.get("plan")
    usage = list(state.get("model_usage") or ())
    limitations = list(state.get("limitations") or ())

    if not deps.persists:
        # A graph built without a database - a benchmark, or a node-level test.
        # Saying so is better than silently discarding the results.
        update["persisted"] = {"matches": 0, "plan_items": 0, "model_calls": 0}
        update["limitations"] = [_NOT_PERSISTED]
        return update

    async with deps.session() as session:
        written = await MatchRepository(session).upsert_many(user_id, run_id, matches)
        plan_items = 0
        if plan is not None:
            saved = await PlanRepository(session).save(user_id, plan)
            plan_items = len(plan.items)
            _log.info("workflow.persist.plan", plan_id=str(saved.id), items=plan_items)
        runs = WorkflowRunRepository(session)
        if usage:
            await runs.add_model_usage(
                user_id, run_id, [item.model_dump(mode="json") for item in usage]
            )
        if limitations:
            await runs.add_limitations(user_id, run_id, limitations)
        await runs.set_phase(user_id, run_id, WorkflowPhase.PERSIST_RESULTS)

    _log.info(
        "workflow.persist.done",
        node=NODE,
        run_id=str(run_id),
        matches=written,
        plan_items=plan_items,
        model_calls=len(usage),
        limitations=len(limitations),
    )
    update["persisted"] = {
        "matches": written,
        "plan_items": plan_items,
        "model_calls": len(usage),
    }
    return update
