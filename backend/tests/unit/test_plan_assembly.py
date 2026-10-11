"""``build_plan``: the step that turns a drafted plan into the real one.

This is the product's output, so the assertions are about what a person ends up
holding in their hand:

* one errand for a document several programs ask for, with every program that
  wants it named - and the *same* document asked for under three different
  words still collapses, because the canonical subject is what keys it;
* a step that belongs to one program stays that program's step, even when two
  programs phrase it identically. Two applications are two applications;
* identity is computed server-side, so a regenerated plan reuses the same item
  ids and a step somebody already completed is not resurrected;
* a link the model invented is dropped rather than followed, and a program the
  run never assessed cannot appear in the plan at all.

``app.services.plan_service.build_plan`` is re-exported from
``app.workflows.plan_builder``; both names are asserted here so the contract in
``docs/INTERFACES.md`` stays true whichever one a caller imports.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from app.schemas import (
    ActionItem,
    ActionItemDraft,
    ActionItemKind,
    ActionPlan,
    ActionPlanDraft,
    AssistanceProgram,
    ProgramMatch,
    action_item_uuid,
    make_dedup_key,
)
from app.services import plan_service
from app.workflows.plan_builder import KIND_ORDER, build_plan, plan_uuid
from tests.conftest import make_match, make_program

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
RUN_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")

GROCERY = make_program("example-grocery-help")
UTILITY = make_program("example-utility-help")


def draft(*items: ActionItemDraft, title: str = "Your next steps") -> ActionPlanDraft:
    return ActionPlanDraft(
        title=title,
        summary="Gather the documents once, then send the applications.",
        items=list(items),
    )


def document(
    program: AssistanceProgram, subject: str, *, title: str | None = None
) -> ActionItemDraft:
    return ActionItemDraft(
        program_slug=program.slug,
        kind=ActionItemKind.DOCUMENT,
        subject=subject,
        title=title or f"Find something that shows your {subject}",
        detail="A recent bill, a lease or a piece of official mail all work.",
    )


def apply_step(program: AssistanceProgram, *, url: str | None = None) -> ActionItemDraft:
    return ActionItemDraft(
        program_slug=program.slug,
        kind=ActionItemKind.APPLY,
        subject="application",
        title="Send the application",
        detail="Fill in the form and submit it with the documents you gathered.",
        url=url,
    )


def matches(*programs: AssistanceProgram) -> list[ProgramMatch]:
    return [make_match(program) for program in programs]


def build(
    plan_draft: ActionPlanDraft,
    *programs: AssistanceProgram,
    version: int = 1,
    run_id: uuid.UUID = RUN_ID,
) -> ActionPlan:
    return build_plan(run_id, matches(*programs), plan_draft, version=version, now=NOW)


def titles_by_kind(plan: ActionPlan, kind: ActionItemKind) -> list[str]:
    return [item.title for item in plan.items if item.kind is kind]


# ---------------------------------------------------------------------------
# shared documents collapse
# ---------------------------------------------------------------------------
class TestSharedDocumentsCollapse:
    def test_one_document_two_programs_is_one_errand(self) -> None:
        plan = build(
            draft(document(GROCERY, "proof of address"), document(UTILITY, "proof of address")),
            GROCERY,
            UTILITY,
        )

        documents = [item for item in plan.items if item.kind is ActionItemKind.DOCUMENT]
        assert len(documents) == 1, "the user should make one trip, not two"
        assert len(plan.shared_documents) == 1
        shared = plan.shared_documents[0]
        assert shared.dedup_key == make_dedup_key(ActionItemKind.DOCUMENT, "proof of address")
        assert shared.program_ids == [GROCERY.id, UTILITY.id]
        assert shared.program_names == [GROCERY.official_name, UTILITY.official_name]

    def test_the_collapsed_item_and_the_shared_record_agree(self) -> None:
        """Two views of one grouping; if they disagree the UI contradicts itself."""
        plan = build(
            draft(document(GROCERY, "proof of address"), document(UTILITY, "utility bill")),
            GROCERY,
            UTILITY,
        )

        assert len(plan.shared_documents) == 1
        keys = {item.dedup_key for item in plan.items}
        assert plan.shared_documents[0].dedup_key in keys

    def test_three_wordings_of_one_document_still_collapse(self) -> None:
        """ "Photo ID", "driver's license" and "state ID" are one errand."""
        third = make_program("example-childcare-help")
        plan = build(
            draft(
                document(GROCERY, "photo ID"),
                document(UTILITY, "driver's license"),
                document(third, "state ID"),
            ),
            GROCERY,
            UTILITY,
            third,
        )

        documents = [item for item in plan.items if item.kind is ActionItemKind.DOCUMENT]
        assert len(documents) == 1
        assert documents[0].dedup_key == "document:photo-id"
        assert plan.shared_documents[0].program_ids == [GROCERY.id, UTILITY.id, third.id]

    def test_a_document_only_one_program_wants_is_not_called_shared(self) -> None:
        plan = build(
            draft(document(GROCERY, "proof of address"), document(GROCERY, "proof of income")),
            GROCERY,
        )

        assert len(plan.items) == 2
        assert plan.shared_documents == [], "overstating the errand list is its own harm"

    def test_shared_documents_are_listed_before_the_program_specific_ones(self) -> None:
        plan = build(
            draft(
                document(GROCERY, "proof of income"),
                document(GROCERY, "proof of address"),
                document(UTILITY, "proof of address"),
            ),
            GROCERY,
            UTILITY,
        )

        first = plan.items[0]
        assert first.dedup_key == plan.shared_documents[0].dedup_key
        assert [item.sort_order for item in plan.items] == list(range(len(plan.items)))

    def test_documents_come_before_applications(self) -> None:
        plan = build(
            draft(apply_step(GROCERY), document(GROCERY, "proof of address")),
            GROCERY,
        )

        kinds = [item.kind for item in plan.items]
        assert kinds == sorted(kinds, key=KIND_ORDER.index)
        assert kinds[0] is ActionItemKind.DOCUMENT


# ---------------------------------------------------------------------------
# program-specific steps do not collapse
# ---------------------------------------------------------------------------
class TestProgramSpecificStepsStaySeparate:
    def test_two_programs_mean_two_applications(self) -> None:
        plan = build(draft(apply_step(GROCERY), apply_step(UTILITY)), GROCERY, UTILITY)

        applications = [item for item in plan.items if item.kind is ActionItemKind.APPLY]
        assert len(applications) == 2, "one application cannot stand in for the other"
        assert {item.program_id for item in applications} == {GROCERY.id, UTILITY.id}
        assert len({item.id for item in applications}) == 2
        assert len({item.dedup_key for item in applications}) == 2
        assert plan.shared_documents == [], "only documents are ever shared"

    @pytest.mark.parametrize(
        "kind",
        [
            ActionItemKind.APPLY,
            ActionItemKind.CONTACT,
            ActionItemKind.FOLLOW_UP,
            ActionItemKind.CONFIRM_INFO,
            ActionItemKind.LEARN,
        ],
    )
    def test_no_non_document_kind_is_ever_collapsed(self, kind: ActionItemKind) -> None:
        """The same subject, the same kind, two programs - still two steps."""
        items = [
            ActionItemDraft(
                program_slug=program.slug,
                kind=kind,
                subject="call the office",
                title="Call the office to confirm",
                detail="Ask whether anything else is needed before you send it.",
            )
            for program in (GROCERY, UTILITY)
        ]
        plan = build(draft(*items), GROCERY, UTILITY)

        assert len([item for item in plan.items if item.kind is kind]) == 2
        assert plan.shared_documents == []

    def test_a_document_and_a_contact_about_it_are_different_errands(self) -> None:
        plan = build(
            draft(
                document(GROCERY, "lease"),
                ActionItemDraft(
                    program_slug=GROCERY.slug,
                    kind=ActionItemKind.CONTACT,
                    subject="lease",
                    title="Ask your landlord for a copy of the lease",
                    detail="They are required to give you one.",
                ),
            ),
            GROCERY,
        )

        assert len(plan.items) == 2
        assert len({item.dedup_key for item in plan.items}) == 2

    def test_every_item_names_the_program_it_belongs_to(self) -> None:
        plan = build(
            draft(apply_step(GROCERY), apply_step(UTILITY), document(GROCERY, "proof of income")),
            GROCERY,
            UTILITY,
        )

        for item in plan.items:
            assert item.program_name, "a step with no program implies an agency we are not"
            assert item.program_id in {GROCERY.id, UTILITY.id}


# ---------------------------------------------------------------------------
# identity, versioning and links
# ---------------------------------------------------------------------------
class TestIdentityIsComputedServerSide:
    def test_regenerating_reuses_the_item_ids(self) -> None:
        """Completion is keyed on the id, so a new version must not reset it."""
        plan_draft = draft(document(GROCERY, "proof of address"), apply_step(GROCERY))
        first = build(plan_draft, GROCERY, version=1)
        second = build(plan_draft, GROCERY, version=2)

        assert {item.id for item in first.items} == {item.id for item in second.items}
        assert first.id != second.id, "a new version is a new plan row"
        assert second.id == plan_uuid(RUN_ID, 2)

    def test_ids_follow_the_published_derivation(self) -> None:
        plan = build(draft(document(GROCERY, "proof of address")), GROCERY)
        item = plan.items[0]
        assert item.id == action_item_uuid(RUN_ID, item.dedup_key)

    def test_two_runs_do_not_share_item_ids(self) -> None:
        plan_draft = draft(document(GROCERY, "proof of address"))
        mine = build(plan_draft, GROCERY)
        theirs = build(plan_draft, GROCERY, run_id=uuid.uuid4())
        assert mine.items[0].id != theirs.items[0].id

    def test_building_twice_is_byte_identical(self) -> None:
        plan_draft = draft(document(GROCERY, "proof of address"), apply_step(UTILITY))
        first = build(plan_draft, GROCERY, UTILITY)
        second = build(plan_draft, GROCERY, UTILITY)
        assert first.model_dump() == second.model_dump()

    def test_a_fresh_plan_has_nothing_ticked(self) -> None:
        plan = build(draft(document(GROCERY, "proof of address")), GROCERY)
        assert all(item.is_complete is False for item in plan.items)
        assert all(item.completed_at is None for item in plan.items)
        assert plan.completed_count == 0
        assert plan.progress_fraction == 0.0


class TestLinksAndPrograms:
    def test_an_invented_link_is_dropped(self) -> None:
        plan = build(
            draft(
                ActionItemDraft(
                    program_slug=GROCERY.slug,
                    kind=ActionItemKind.LEARN,
                    subject="eligibility details",
                    title="Read the programme rules",
                    detail="The official page lists what is required.",
                    url="https://not-the-catalog.example.com/apply-here",
                )
            ),
            GROCERY,
        )

        assert plan.items[0].url is None, "a link nobody published must not be offered"

    def test_an_application_step_falls_back_to_the_official_url(self) -> None:
        plan = build(draft(apply_step(GROCERY, url="https://phishing.example.net/")), GROCERY)
        assert plan.items[0].url == GROCERY.application_url

    def test_a_catalog_link_survives(self) -> None:
        plan = build(draft(apply_step(GROCERY, url=GROCERY.official_url)), GROCERY)
        assert plan.items[0].url in {GROCERY.official_url, GROCERY.application_url}

    def test_a_step_for_a_program_this_run_never_assessed_is_dropped(self) -> None:
        """The plan may only point at programs the run actually evaluated."""
        plan = build(
            draft(document(GROCERY, "proof of address"), document(UTILITY, "proof of address")),
            GROCERY,
        )

        assert len(plan.items) == 1
        assert plan.items[0].program_id == GROCERY.id
        assert plan.shared_documents == [], "a dropped program cannot make a document shared"

    def test_a_plan_with_no_usable_drafts_is_empty_not_invented(self) -> None:
        plan = build(draft(document(UTILITY, "proof of address")), GROCERY)
        assert plan.items == []
        assert plan.shared_documents == []
        assert plan.title == "Your next steps", "the author-written copy is still shown"


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------
def test_the_service_re_exports_the_documented_entry_point() -> None:
    """``docs/INTERFACES.md`` publishes ``plan_service.build_plan``."""
    assert plan_service.build_plan is build_plan
    assert "build_plan" in plan_service.__all__


def test_the_draft_round_trip_preserves_item_identity() -> None:
    """``draft_from_plan`` is what a selective re-evaluation regenerates from.

    Identity is the part that matters for the user: the rebuilt items must keep
    the ids the completed-step rows point at.
    """
    plan = build(
        draft(document(GROCERY, "proof of address"), document(UTILITY, "proof of address")),
        GROCERY,
        UTILITY,
    )
    round_tripped = plan_service.draft_from_plan(plan, matches(GROCERY, UTILITY))
    rebuilt = build_plan(RUN_ID, matches(GROCERY, UTILITY), round_tripped, version=2, now=NOW)

    assert {item.dedup_key for item in rebuilt.items} == {item.dedup_key for item in plan.items}
    assert {item.id for item in rebuilt.items} == {item.id for item in plan.items}


def test_a_dropped_program_takes_its_steps_out_of_the_regenerated_draft() -> None:
    plan = build(draft(apply_step(GROCERY), apply_step(UTILITY)), GROCERY, UTILITY)
    round_tripped = plan_service.draft_from_plan(plan, matches(GROCERY))

    assert [item.program_slug for item in round_tripped.items] == [GROCERY.slug]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "defect in app/services/plan_service.py (not owned by this file): "
        "draft_from_plan reads back the already-collapsed items, so only one "
        "program's draft survives and group_shared_documents can no longer see "
        "that two programs want the document. Version 2 of a plan therefore "
        "drops shared_documents, contradicting the function's own docstring. "
        "Remove this xfail when the service emits one draft per asking program."
    ),
)
def test_a_shared_document_is_still_shared_after_a_regeneration() -> None:
    """ "Four programs need this" must not disappear on version 2."""
    plan = build(
        draft(document(GROCERY, "proof of address"), document(UTILITY, "proof of address")),
        GROCERY,
        UTILITY,
    )
    assert len(plan.shared_documents) == 1

    round_tripped = plan_service.draft_from_plan(plan, matches(GROCERY, UTILITY))
    rebuilt = build_plan(RUN_ID, matches(GROCERY, UTILITY), round_tripped, version=2, now=NOW)

    assert len(rebuilt.shared_documents) == 1
    assert rebuilt.shared_documents[0].program_ids == [GROCERY.id, UTILITY.id]


def test_every_item_kind_the_builder_orders_is_a_real_kind() -> None:
    assert set(KIND_ORDER) <= set(ActionItemKind)
    assert len(set(KIND_ORDER)) == len(KIND_ORDER)


def test_an_action_item_cannot_claim_completion_without_a_timestamp() -> None:
    """The invariant the plan relies on when it reports progress."""
    plan = build(draft(document(GROCERY, "proof of address")), GROCERY)
    item = plan.items[0]
    with pytest.raises(ValueError, match="completed_at"):
        ActionItem.model_validate({**item.model_dump(), "is_complete": True})
