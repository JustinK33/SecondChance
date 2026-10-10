"""Action plan and action item persistence.

Plans are versioned, items are deduplicated, and both of those are database
constraints rather than conventions:

* ``UNIQUE (run_id, version)`` means :meth:`PlanRepository.save` has to be told
  which version it is writing, and regenerating a plan writes N+1 rather than
  overwriting N. A step the user already ticked off is never rewritten
  underneath them.
* ``UNIQUE (plan_id, dedup_key)`` means two programs that both ask for "proof
  of address" cannot produce two items, even if the generator worded them
  differently - the key is built from the canonical subject by
  :func:`app.schemas.plan.make_dedup_key`, not from the title.

Item ids are ``uuid5(run_id, dedup_key)``, so the same requirement has the same
id however many times a plan is regenerated for a run. That is what lets the
frontend carry completion state across a regeneration and what makes
:meth:`PlanRepository.save` safe to retry.

It also forces one design decision worth stating plainly. An item id is derived
from the *run*, not the plan, so a single requirement cannot exist in two plan
versions of the same run at once - and it must not, or ``PATCH
/api/v1/action-items/{id}`` would have two rows to choose from. So saving
version N+1 moves the existing item rows onto the new plan, carrying their
completion state with them, while version N keeps its own title, summary and
shared-document grouping as the historical record of what the user was told.
The practical case this serves is the common one: a profile edit starts a *new
run* (see ``MatchRequest``), with new item ids and no interaction with the old
plan at all. Plan versions within one run come from regenerating that run's
plan, where the items genuinely are the same items.

Only completion is mutable. ``set_item_complete`` writes ``is_complete`` and
``completed_at`` and nothing else, so a user cannot rewrite an item's title or
point its URL somewhere else.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.db import models
from app.db.repositories.base import (
    UserScopedRepository,
    count_of,
    normalise_page,
)
from app.schemas import (
    ActionItem,
    ActionItemKind,
    ActionPlan,
    Page,
    SharedDocument,
    utcnow,
)

__all__ = ["PlanRepository"]

_log = get_logger(__name__)


def _to_item(row: models.ActionItem) -> ActionItem:
    return ActionItem(
        id=row.id,
        program_id=row.program_id,
        program_name=row.program_name,
        kind=ActionItemKind(row.kind),
        title=row.title,
        detail=row.detail,
        url=row.url,
        sort_order=row.sort_order,
        is_complete=row.is_complete,
        completed_at=row.completed_at,
        dedup_key=row.dedup_key,
    )


def _to_plan(row: models.ActionPlan) -> ActionPlan:
    return ActionPlan(
        id=row.id,
        run_id=row.run_id,
        version=row.version,
        title=row.title,
        summary=row.summary,
        items=[_to_item(item) for item in row.items],
        shared_documents=[SharedDocument.model_validate(doc) for doc in row.shared_documents],
        created_at=row.created_at,
    )


class PlanRepository(UserScopedRepository):
    """Read and write action plans for one user."""

    __slots__ = ()

    async def next_version(self, user_id: str, run_id: uuid.UUID) -> int:
        """The version number a new plan for ``run_id`` should take."""
        current = await self.session.scalar(
            select(func.coalesce(func.max(models.ActionPlan.version), 0)).where(
                models.ActionPlan.run_id == run_id,
                models.ActionPlan.user_id == user_id,
            )
        )
        return int(current or 0) + 1

    async def save(self, user_id: str, plan: ActionPlan) -> models.ActionPlan:
        """Persist a plan and its items.

        Idempotent: re-saving the same plan object - a retried request, a
        replayed workflow step - converges on the same rows rather than
        failing, because the plan id and every item id are derived from the run
        and the deduplication key rather than generated.

        Completion state is preserved across a re-save. The generator has no
        opinion about whether a user has done a step, so ``is_complete`` and
        ``completed_at`` are deliberately left out of the conflict update.
        """
        plans = models.ActionPlan
        plan_values = {
            "id": plan.id,
            "user_id": user_id,
            "run_id": plan.run_id,
            "version": plan.version,
            "title": plan.title,
            "summary": plan.summary,
            "shared_documents": [doc.model_dump(mode="json") for doc in plan.shared_documents],
        }
        statement = (
            pg_insert(plans)
            .values(**plan_values)
            .on_conflict_do_update(
                index_elements=[plans.run_id, plans.version],
                set_={
                    "title": plan_values["title"],
                    "summary": plan_values["summary"],
                    "shared_documents": plan_values["shared_documents"],
                    "updated_at": func.now(),
                },
            )
            .returning(plans.id)
        )
        plan_id = (await self.session.execute(statement)).scalar_one()

        if plan.items:
            items = models.ActionItem
            rows = [
                {
                    "id": item.id,
                    "plan_id": plan_id,
                    "user_id": user_id,
                    "program_id": item.program_id,
                    "program_name": item.program_name,
                    "kind": item.kind.value,
                    "title": item.title,
                    "detail": item.detail,
                    "url": item.url,
                    "sort_order": item.sort_order,
                    "is_complete": item.is_complete,
                    "completed_at": item.completed_at,
                    "dedup_key": item.dedup_key,
                }
                for item in plan.items
            ]
            item_statement = pg_insert(items).values(rows)
            # Conflict on the item's own primary key, which is
            # ``uuid5(run_id, dedup_key)`` and therefore identical across plan
            # versions of the same run. See the note on versioning in the
            # module docstring: the row is re-pointed at the new plan rather
            # than duplicated, which is what keeps ``PATCH
            # /api/v1/action-items/{id}`` unambiguous.
            #
            # ``is_complete`` and ``completed_at`` are deliberately absent from
            # the update set. The generator has no opinion about whether a user
            # has done something, so a regeneration must never un-tick a step.
            item_statement = item_statement.on_conflict_do_update(
                index_elements=[items.id],
                set_={
                    name: item_statement.excluded[name]
                    for name in (
                        "plan_id",
                        "program_id",
                        "program_name",
                        "kind",
                        "title",
                        "detail",
                        "url",
                        "sort_order",
                        "dedup_key",
                    )
                }
                | {"updated_at": func.now()},
            )
            await self.session.execute(item_statement)

        await self.session.flush()
        row = await self.session.get_one(
            plans, plan_id, options=[selectinload(plans.items)], populate_existing=True
        )
        _log.debug(
            "plan.saved",
            plan_id=str(plan_id),
            run_id=str(plan.run_id),
            version=plan.version,
            item_count=len(plan.items),
        )
        return row

    async def get(self, user_id: str, plan_id: uuid.UUID) -> ActionPlan | None:
        """One plan by id, scoped to its owner."""
        statement = (
            select(models.ActionPlan)
            .where(models.ActionPlan.id == plan_id, models.ActionPlan.user_id == user_id)
            .options(selectinload(models.ActionPlan.items))
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return _to_plan(row) if row else None

    async def get_latest_for_run(self, user_id: str, run_id: uuid.UUID) -> ActionPlan | None:
        statement = (
            select(models.ActionPlan)
            .where(models.ActionPlan.run_id == run_id, models.ActionPlan.user_id == user_id)
            .order_by(models.ActionPlan.version.desc())
            .limit(1)
            .options(selectinload(models.ActionPlan.items))
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return _to_plan(row) if row else None

    async def list_for_user(
        self, user_id: str, *, limit: int = 20, offset: int = 0
    ) -> Page[ActionPlan]:
        limit, offset = normalise_page(limit, offset)
        base = select(models.ActionPlan).where(models.ActionPlan.user_id == user_id)
        total = await count_of(self.session, base)
        statement = (
            base.order_by(models.ActionPlan.created_at.desc(), models.ActionPlan.id.desc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(models.ActionPlan.items))
        )
        rows = (await self.session.scalars(statement)).all()
        return Page(items=[_to_plan(row) for row in rows], total=total, limit=limit, offset=offset)

    async def get_item(self, user_id: str, item_id: uuid.UUID) -> ActionItem | None:
        statement = select(models.ActionItem).where(
            models.ActionItem.id == item_id, models.ActionItem.user_id == user_id
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return _to_item(row) if row else None

    async def set_item_complete(
        self, user_id: str, item_id: uuid.UUID, *, is_complete: bool
    ) -> ActionItem | None:
        """Tick or untick one step.

        ``completed_at`` is set and cleared together with the flag, because the
        table's check constraint requires the two to agree - a "complete" item
        with no timestamp is not a state this product allows.
        """
        statement = select(models.ActionItem).where(
            models.ActionItem.id == item_id, models.ActionItem.user_id == user_id
        )
        row = (await self.session.scalars(statement)).one_or_none()
        if row is None:
            return None
        row.is_complete = is_complete
        row.completed_at = utcnow() if is_complete else None
        await self.session.flush()
        return _to_item(row)

    async def list_items_for_plan(
        self, user_id: str, plan_id: uuid.UUID
    ) -> Sequence[models.ActionItem]:
        statement = (
            select(models.ActionItem)
            .where(models.ActionItem.plan_id == plan_id, models.ActionItem.user_id == user_id)
            .order_by(models.ActionItem.sort_order.asc())
        )
        return list((await self.session.scalars(statement)).all())
