"""Action plan schemas, and the deduplication rule for shared documents.

The plan is the product's actual output. A user who has been told they might
qualify for five programs does not need five separate instruction sets that each
begin "gather proof of address" - they need one list of documents and five
applications.

The dedup key
-------------
``dedup_key`` is ``"{kind}:{slugify(subject)}"``:

* ``kind`` is the :class:`~app.schemas.enums.ActionItemKind`, lower-cased. Items
  of different kinds never collapse, so "bring your lease to the office"
  (``CONTACT``) stays separate from "find your lease" (``DOCUMENT``).
* ``subject`` is the canonical requirement, not the item's title. Titles are
  model-written and vary; the subject is first normalised through
  :data:`DOCUMENT_SUBJECT_ALIASES` so "driver's license", "photo ID" and
  "state ID" all reduce to ``photo-id``, then slugified.

Only ``DOCUMENT`` items are grouped into ``shared_documents``. ``APPLY``,
``CONTACT`` and ``FOLLOW_UP`` items are per-program by nature: two programs
sharing an application portal still need two applications.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import HttpUrlStr, NonEmptyStr, PlainText, UtcDatetime, slugify
from app.schemas.enums import ActionItemKind

__all__ = [
    "DOCUMENT_SUBJECT_ALIASES",
    "ActionItem",
    "ActionItemDraft",
    "ActionItemPatch",
    "ActionPlan",
    "ActionPlanDraft",
    "SharedDocument",
    "group_shared_documents",
    "make_dedup_key",
]

#: Canonical subject for each commonly requested document, keyed by a slugified
#: variant. Without this, deduplication fails exactly where it matters most:
#: programs describe the same document in different words.
DOCUMENT_SUBJECT_ALIASES: Final[dict[str, str]] = {
    "photo-id": "photo-id",
    "government-issued-photo-id": "photo-id",
    "drivers-license": "photo-id",
    "driver-s-license": "photo-id",
    "state-id": "photo-id",
    "id-card": "photo-id",
    "proof-of-identity": "photo-id",
    "proof-of-address": "proof-of-address",
    "proof-of-nj-address": "proof-of-address",
    "proof-of-residency": "proof-of-address",
    "proof-of-new-jersey-residency": "proof-of-address",
    "utility-bill": "proof-of-address",
    "lease": "lease-or-mortgage",
    "lease-agreement": "lease-or-mortgage",
    "rental-agreement": "lease-or-mortgage",
    "mortgage-statement": "lease-or-mortgage",
    "proof-of-income": "proof-of-income",
    "pay-stubs": "proof-of-income",
    "paystubs": "proof-of-income",
    "recent-pay-stubs": "proof-of-income",
    "income-verification": "proof-of-income",
    "tax-return": "tax-return",
    "federal-tax-return": "tax-return",
    "form-1040": "tax-return",
    "w-2": "tax-return",
    "social-security-award-letter": "benefit-award-letter",
    "benefit-award-letter": "benefit-award-letter",
    "unemployment-benefit-letter": "benefit-award-letter",
    "birth-certificate": "birth-certificate",
    "proof-of-enrollment": "proof-of-enrollment",
    "enrollment-verification": "proof-of-enrollment",
    "class-schedule": "proof-of-enrollment",
    "student-id": "proof-of-enrollment",
    "utility-shutoff-notice": "utility-shutoff-notice",
    "shut-off-notice": "utility-shutoff-notice",
    "disconnection-notice": "utility-shutoff-notice",
    "bank-statement": "bank-statement",
    "medical-bills": "medical-bills",
    "proof-of-pregnancy": "proof-of-pregnancy",
}


def make_dedup_key(kind: ActionItemKind, subject: str) -> str:
    """Build the collapse key for an action item.

    ``subject`` must be the canonical requirement ("proof of address"), not the
    item's display title. The result is stable across runs and across model
    wordings, which is what makes :func:`group_shared_documents` and the
    deterministic action-item ids work.
    """
    slug = slugify(subject)
    canonical = DOCUMENT_SUBJECT_ALIASES.get(slug, slug)
    return f"{kind.value.lower()}:{canonical}"


class ActionItemDraft(BaseModel):
    """A model-proposed action item, before the server assigns identity.

    Ids, ``dedup_key`` and ordering are computed server-side from ``kind`` and
    ``subject``, so the model cannot influence deduplication or create
    colliding identifiers.
    """

    model_config = ConfigDict(extra="forbid")

    program_slug: NonEmptyStr
    kind: ActionItemKind
    subject: NonEmptyStr = Field(
        description='Canonical requirement for deduplication, e.g. "proof of address".'
    )
    title: PlainText = Field(description="Imperative, one line, e.g. 'Find a recent utility bill'.")
    detail: PlainText = Field(description="What to do and why, in plain language.")
    url: HttpUrlStr | None = Field(
        default=None,
        description="Must be an official program URL. Validated against the catalog before use.",
    )


class ActionItem(BaseModel):
    """One concrete step, owned by one user."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID = Field(description="uuid5 of (run_id, dedup_key); see common.action_item_uuid.")
    program_id: uuid.UUID
    program_name: NonEmptyStr = Field(description="Denormalised for display; never a link target.")
    kind: ActionItemKind
    title: PlainText
    detail: PlainText
    url: HttpUrlStr | None = None
    sort_order: int = Field(ge=0, description="Ascending display order within the plan.")
    is_complete: bool = False
    completed_at: UtcDatetime | None = None
    dedup_key: NonEmptyStr = Field(description="See the module docstring for the derivation rule.")

    @model_validator(mode="after")
    def _completion_is_consistent(self) -> Self:
        if self.is_complete and self.completed_at is None:
            msg = "a completed action item must record completed_at"
            raise ValueError(msg)
        if not self.is_complete and self.completed_at is not None:
            msg = "completed_at must be null while is_complete is false"
            raise ValueError(msg)
        return self


class ActionItemPatch(BaseModel):
    """Body of ``PATCH /api/v1/action-items/{id}``.

    Only completion is editable. Titles, URLs and deduplication come from the
    plan generator, so a user cannot rewrite an item into a link somewhere else.
    """

    model_config = ConfigDict(extra="forbid")

    is_complete: bool


class SharedDocument(BaseModel):
    """One document requirement that several programs ask for."""

    model_config = ConfigDict(extra="forbid")

    dedup_key: NonEmptyStr
    title: PlainText
    detail: PlainText
    program_ids: list[uuid.UUID] = Field(min_length=1)
    program_names: list[NonEmptyStr] = Field(min_length=1)

    @model_validator(mode="after")
    def _parallel_lists(self) -> Self:
        if len(self.program_ids) != len(self.program_names):
            msg = "program_ids and program_names must be the same length"
            raise ValueError(msg)
        return self


class ActionPlanDraft(BaseModel):
    """Model-proposed plan, before the server assigns identity and deduplicates."""

    model_config = ConfigDict(extra="forbid")

    title: PlainText
    summary: PlainText = Field(description="Two or three sentences on what to do first.")
    items: list[ActionItemDraft] = Field(default_factory=list, max_length=60)


class ActionPlan(BaseModel):
    """The persisted plan for one workflow run.

    Versioned rather than mutated: editing a profile and re-running produces
    version N+1, so a user can see what changed and a completed step is never
    silently rewritten underneath them.
    """

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    run_id: uuid.UUID
    version: int = Field(ge=1)
    title: PlainText
    summary: PlainText
    items: list[ActionItem] = Field(default_factory=list)
    shared_documents: list[SharedDocument] = Field(default_factory=list)
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _unique_items(self) -> Self:
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            msg = "action item ids must be unique within a plan"
            raise ValueError(msg)
        return self

    @property
    def completed_count(self) -> int:
        return sum(1 for item in self.items if item.is_complete)

    @property
    def progress_fraction(self) -> float:
        if not self.items:
            return 0.0
        return self.completed_count / len(self.items)


def group_shared_documents(items: list[ActionItem]) -> list[SharedDocument]:
    """Collapse ``DOCUMENT`` items that more than one program asks for.

    Grouping is by ``dedup_key``, so it relies entirely on the canonical subject
    rather than on model wording. An item required by a single program is left
    alone: promoting it to a "shared" document would overstate what the user has
    to gather. Programs are listed in first-seen order, which keeps the output
    deterministic.
    """
    buckets: OrderedDict[str, list[ActionItem]] = OrderedDict()
    for item in items:
        if item.kind is not ActionItemKind.DOCUMENT:
            continue
        buckets.setdefault(item.dedup_key, []).append(item)

    shared: list[SharedDocument] = []
    for dedup_key, grouped in buckets.items():
        program_ids = list(dict.fromkeys(item.program_id for item in grouped))
        if len(program_ids) < 2:
            continue
        by_id = {item.program_id: item.program_name for item in grouped}
        shared.append(
            SharedDocument(
                dedup_key=dedup_key,
                title=grouped[0].title,
                detail=grouped[0].detail,
                program_ids=program_ids,
                program_names=[by_id[program_id] for program_id in program_ids],
            )
        )
    return shared
