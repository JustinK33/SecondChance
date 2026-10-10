"""Action plans: reads, completion toggling and versioning on regeneration.

Three things worth naming, because each one is a decision rather than
plumbing.

**Versioning, not overwriting.** A regenerated plan is a new ``version`` row
for the same run, so "what did you tell me last Tuesday" stays answerable.
``PlanRepository.save`` upserts on ``(run_id, version)``, and the caller asks
for the next version through :func:`next_plan_version`.

**Completion survives regeneration.** An item's primary key is
``action_item_uuid(run_id, dedup_key)``, identical across versions of one run,
and the repository deliberately leaves ``is_complete`` out of its conflict
update. The generator has no opinion about whether a user has done something,
so a regeneration must never un-tick a step somebody completed.

**Ownership is checked in SQL, not in Python.** Every read and write here goes
through a ``user_id``-scoped query and a miss returns ``None``, which this
module turns into a 404. A plan or an item belonging to someone else is
indistinguishable from one that does not exist, so the API cannot be used to
confirm that an id is real.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.repositories import AuditAction, AuditRepository, PlanRepository
from app.schemas import (
    ActionItem,
    ActionItemDraft,
    ActionItemPatch,
    ActionPlan,
    ActionPlanDraft,
    Page,
    ProgramMatch,
)
from app.schemas.enums import ActionItemKind
from app.workflows.plan_builder import build_plan

__all__ = [
    "build_plan",
    "draft_from_plan",
    "get_plan",
    "get_plan_for_run",
    "list_plans",
    "next_plan_version",
    "save_plan_version",
    "set_item_complete",
]

_log = get_logger(__name__)


async def get_plan(session: AsyncSession, user_id: str, plan_id: uuid.UUID) -> ActionPlan:
    """One plan the caller owns. Raises ``NotFoundError`` otherwise."""
    plan = await PlanRepository(session).get(user_id, plan_id)
    if plan is None:
        raise NotFoundError("We could not find that plan.")
    return plan


async def get_plan_for_run(
    session: AsyncSession, user_id: str, run_id: uuid.UUID
) -> ActionPlan | None:
    """The newest plan version for a run, or ``None`` if it has none yet."""
    return await PlanRepository(session).get_latest_for_run(user_id, run_id)


async def list_plans(
    session: AsyncSession, user_id: str, *, limit: int = 20, offset: int = 0
) -> Page[ActionPlan]:
    """A page of the caller's plans, newest first."""
    return await PlanRepository(session).list_for_user(user_id, limit=limit, offset=offset)


async def set_item_complete(
    session: AsyncSession,
    user_id: str,
    item_id: uuid.UUID,
    patch: ActionItemPatch,
    *,
    request_id: str | None = None,
) -> ActionItem:
    """Tick or untick one step, enforcing ownership.

    Commits: the user tapped a checkbox and expects it to still be ticked after
    a refresh.
    """
    repository = PlanRepository(session)
    item = await repository.set_item_complete(user_id, item_id, is_complete=patch.is_complete)
    if item is None:
        raise NotFoundError("We could not find that step.")
    await AuditRepository(session).record(
        user_id,
        AuditAction.ACTION_ITEM_UPDATED,
        resource_type="action_item",
        resource_id=item_id,
        request_id=request_id,
        metadata={"is_complete": patch.is_complete, "kind": item.kind.value},
    )
    await session.commit()
    _log.info("plan.item.updated", item_id=str(item_id), is_complete=patch.is_complete)
    return item


async def next_plan_version(session: AsyncSession, user_id: str, run_id: uuid.UUID) -> int:
    """The version number a new plan for ``run_id`` should take."""
    return await PlanRepository(session).next_version(user_id, run_id)


async def save_plan_version(
    session: AsyncSession,
    user_id: str,
    run_id: uuid.UUID,
    matches: Sequence[ProgramMatch],
    draft: ActionPlanDraft,
    *,
    request_id: str | None = None,
) -> ActionPlan:
    """Build and store the next plan version for a run.

    Does not commit: a plan is normally written in the same transaction as the
    matches it was built from, and the caller owns that boundary.
    """
    repository = PlanRepository(session)
    version = await repository.next_version(user_id, run_id)
    plan = build_plan(run_id, matches, draft, version=version)
    await repository.save(user_id, plan)
    await AuditRepository(session).record(
        user_id,
        AuditAction.PLAN_CREATED,
        resource_type="action_plan",
        resource_id=plan.id,
        run_id=run_id,
        request_id=request_id,
        metadata={"version": version, "item_count": len(plan.items)},
    )
    _log.info("plan.version.saved", run_id=str(run_id), version=version, item_count=len(plan.items))
    return plan


def draft_from_plan(
    plan: ActionPlan,
    matches: Sequence[ProgramMatch],
) -> ActionPlanDraft:
    """Recover a draft from a stored plan, for a deterministic regeneration.

    Used by the selective re-evaluation path, which must not make a model call:
    the wording a user already read is carried forward rather than rewritten,
    and only the set of programs it may point at changes. Any item whose
    program is no longer in ``matches`` is dropped, because a plan may only
    point at programs the run it belongs to actually assessed.

    The canonical subject is read back out of ``dedup_key``, which is
    ``"{kind}:{canonical_subject}"`` by construction, so ``make_dedup_key``
    reproduces the same key, the rebuilt item keeps its identity within the run
    and a shared document stays shared.
    """
    slug_by_program = {match.program.id: match.program.slug for match in matches}
    items: list[ActionItemDraft] = []
    for item in sorted(plan.items, key=lambda entry: entry.sort_order):
        slug = slug_by_program.get(item.program_id)
        if slug is None:
            continue
        items.append(
            ActionItemDraft(
                program_slug=slug,
                kind=item.kind,
                subject=_subject_of(item.dedup_key, item.kind),
                title=item.title,
                detail=item.detail,
                url=item.url,
            )
        )
    return ActionPlanDraft(title=plan.title, summary=plan.summary, items=items)


def _subject_of(dedup_key: str, kind: ActionItemKind) -> str:
    """The canonical subject encoded in a ``dedup_key``."""
    prefix = f"{kind.value.lower()}:"
    if dedup_key.startswith(prefix):
        remainder = dedup_key[len(prefix) :]
        if remainder:
            return remainder
    return dedup_key
