"""Turning a model-drafted plan into the persisted :class:`ActionPlan`.

``docs/INTERFACES.md`` assigns this to ``app.services.plan_service.build_plan``,
which has not landed yet, so the signature there is implemented here verbatim
and injected through :class:`~app.workflows.deps.WorkflowDeps`. See the handoff
note in the task report.

Why the server owns identity
----------------------------
The model proposes ``kind``, ``subject``, ``title``, ``detail`` and a URL.
Everything that has consequences is computed here:

* ``dedup_key`` comes from :func:`~app.schemas.plan.make_dedup_key`, so the
  model cannot influence what collapses with what.
* ``id`` is ``action_item_uuid(run_id, dedup_key)``, so regenerating a plan for
  the same run produces the same ids and a user's completed steps survive.
* a URL survives only if the catalog publishes it for that program. An
  invented link is dropped, never followed.

The ``(plan_id, dedup_key)`` unique constraint means one row per key per plan,
so ``DOCUMENT`` drafts from several programs collapse into a single errand -
which is the whole point of the plan. ``shared_documents`` is computed from the
*uncollapsed* drafts with :func:`~app.schemas.plan.group_shared_documents`, so
the "four programs want this" view and the collapsed item list are derived from
one grouping and cannot disagree.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Final

from app.schemas import (
    ActionItem,
    ActionItemDraft,
    ActionItemKind,
    ActionPlan,
    ActionPlanDraft,
    ProgramMatch,
    ProgramSummary,
    SharedDocument,
    action_item_uuid,
    group_shared_documents,
    make_dedup_key,
    utcnow,
)

__all__ = ["KIND_ORDER", "build_plan", "plan_uuid"]

#: Display order for the plan. Documents first because they are the errands
#: that unblock everything else, and because one trip collects several of them.
KIND_ORDER: Final[tuple[ActionItemKind, ...]] = (
    ActionItemKind.DOCUMENT,
    ActionItemKind.CONFIRM_INFO,
    ActionItemKind.APPLY,
    ActionItemKind.CONTACT,
    ActionItemKind.FOLLOW_UP,
    ActionItemKind.LEARN,
)

_PLAN_NAMESPACE: Final[uuid.UUID] = uuid.UUID("3f6c0b7a-6d1e-5f4b-8a2c-9e0d1f2a3b4c")


def plan_uuid(run_id: uuid.UUID, version: int) -> uuid.UUID:
    """Deterministic plan id, so re-saving one run's plan converges."""
    return uuid.uuid5(_PLAN_NAMESPACE, f"{run_id}:{version}")


def _allowed_urls(summary: ProgramSummary) -> set[str]:
    urls = {summary.official_url}
    if summary.application_url:
        urls.add(summary.application_url)
    return urls


def _item(
    *,
    run_id: uuid.UUID,
    summary: ProgramSummary,
    draft: ActionItemDraft,
    dedup_key: str,
    sort_order: int,
) -> ActionItem:
    url = draft.url if draft.url in _allowed_urls(summary) else None
    if url is None and draft.kind is ActionItemKind.APPLY:
        url = summary.application_url or summary.official_url
    return ActionItem(
        id=action_item_uuid(run_id, dedup_key),
        program_id=summary.id,
        program_name=summary.official_name,
        kind=draft.kind,
        title=draft.title,
        detail=draft.detail,
        url=url,
        sort_order=sort_order,
        is_complete=False,
        completed_at=None,
        dedup_key=dedup_key,
    )


def build_plan(
    run_id: uuid.UUID,
    matches: Sequence[ProgramMatch],
    draft: ActionPlanDraft,
    *,
    version: int = 1,
    now: datetime | None = None,
) -> ActionPlan:
    """Assemble the persisted plan. Pure and deterministic.

    Drafts naming a program that is not in ``matches`` are dropped: the plan may
    only ever point at programs this run actually assessed.
    """
    by_slug = {match.program.slug: match.program for match in matches}

    # One provisional item per draft, keyed canonically, so that two programs
    # asking for the same document are visible as two before they collapse.
    provisional: list[tuple[ActionItemDraft, ProgramSummary, str]] = []
    for item_draft in draft.items:
        summary = by_slug.get(item_draft.program_slug)
        if summary is None:
            continue
        provisional.append(
            (item_draft, summary, make_dedup_key(item_draft.kind, item_draft.subject))
        )

    shared: list[SharedDocument] = group_shared_documents(
        [
            _item(
                run_id=run_id,
                summary=summary,
                draft=item_draft,
                dedup_key=key,
                sort_order=index,
            )
            for index, (item_draft, summary, key) in enumerate(provisional)
        ]
    )

    # Collapse. A DOCUMENT key is shared on purpose - one errand for several
    # programs - so the first occurrence wins. Any other kind is per-program by
    # nature, so a repeated key is disambiguated rather than dropped: two
    # programs both needing an application mean two applications.
    chosen: dict[str, tuple[ActionItemDraft, ProgramSummary]] = {}
    for item_draft, summary, key in provisional:
        if key not in chosen:
            chosen[key] = (item_draft, summary)
            continue
        if item_draft.kind is ActionItemKind.DOCUMENT:
            continue
        scoped = make_dedup_key(item_draft.kind, f"{item_draft.subject} {summary.slug}")
        chosen.setdefault(scoped, (item_draft, summary))

    def order(entry: tuple[str, tuple[ActionItemDraft, ProgramSummary]]) -> tuple[int, int, str]:
        key, (item_draft, _program) = entry
        kind_rank = (
            KIND_ORDER.index(item_draft.kind) if item_draft.kind in KIND_ORDER else len(KIND_ORDER)
        )
        shared_rank = 0 if any(document.dedup_key == key for document in shared) else 1
        return (kind_rank, shared_rank, key)

    items = [
        _item(
            run_id=run_id,
            summary=summary,
            draft=item_draft,
            dedup_key=key,
            sort_order=index,
        )
        for index, (key, (item_draft, summary)) in enumerate(sorted(chosen.items(), key=order))
    ]

    return ActionPlan(
        id=plan_uuid(run_id, version),
        run_id=run_id,
        version=version,
        title=draft.title,
        summary=draft.summary,
        items=items,
        shared_documents=shared,
        created_at=now or utcnow(),
    )
