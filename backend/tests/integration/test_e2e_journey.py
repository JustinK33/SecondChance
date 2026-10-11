"""The end-to-end journey, run as a test and asserted a second time.

``scripts/e2e_journey.py`` is the definition of "the MVP works": one fictional
person walks the whole product over HTTP, and every transition is asserted as
it happens. This module runs that script against the test database so the
journey is part of the suite rather than something a human has to remember to
run, and then re-asserts the invariants that matter most against what the
journey observed - because an assertion inside a script is easy to weaken by
accident, and two independent statements of the same property are harder to
break silently.

It also captures the real logging pipeline around the whole run, which makes
this the only place a *full* workflow - background runner, LangGraph nodes,
persistence - is checked for personal content in its logs.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio

from app.core.config import Settings
from app.schemas import MatchStatus, WorkflowStatus
from scripts.e2e_journey import (
    GROCERY_SLUG,
    NARRATIVE,
    UTILITY_SLUG,
    JourneyResult,
    journey_settings,
    run_journey,
)
from tests.security.harness import capture_app_logs

pytestmark = pytest.mark.db

#: Fragments of the fictional narrative that must never reach a log line.
NARRATIVE_FRAGMENTS = (
    "New Brunswick",
    "my two kids",
    "part time at a store",
    "$2,000 a month",
    "electric bill",
)

#: The extracted income, in the two shapes it exists in.
INCOME_FRAGMENTS = ("190000", "210000", "1900.00", "2100.00")


@dataclass(slots=True)
class Journey:
    """One journey run, plus everything the logging pipeline emitted."""

    result: JourneyResult
    logs: str
    trace: str
    settings: Settings


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def journey(migrated_database: str) -> AsyncIterator[Journey]:
    """Run the journey once for this module, under log capture.

    Module-scoped because the journey is the expensive thing in this file and
    every assertion below is a read of what it saw. ``migrated_database``
    guarantees the schema; the journey owns its own engine, seeds its own
    fictional catalog and deletes everything it created.
    """
    settings = journey_settings(database_url=migrated_database)
    trace = io.StringIO()
    with capture_app_logs(settings) as logs:
        result = await run_journey(settings, out=trace)
    # Printed so the step-by-step trace is attached to a failure report.
    print(trace.getvalue())
    yield Journey(result=result, logs=logs.getvalue(), trace=trace.getvalue(), settings=settings)


class TestTheJourneyItself:
    """The script asserts every transition; this checks it really ran them."""

    def test_every_step_ran_and_was_asserted(self, journey: Journey) -> None:
        names = [step.name for step in journey.result.steps]
        assert len(names) == 14, f"the journey changed shape: {names}"
        # The four transitions the product exists for, by position.
        assert "extraction" in names[1]
        assert "pauses" in names[3]
        assert "resumes" in names[4]
        assert "evidence" in names[5]

    def test_the_trace_is_readable_by_a_human(self, journey: Journey) -> None:
        trace = journey.trace
        assert "SecondChance end-to-end journey" in trace
        assert "steps, all asserted" in trace
        for step in journey.result.steps:
            assert step.name in trace
            assert step.duration_ms >= 0


class TestWhatTheJourneySaw:
    """Re-assertions, stated independently of the script's own checks."""

    def test_extraction_kept_a_range_and_invented_nothing(self, journey: Journey) -> None:
        profile = journey.result.intake["profile"]
        assert profile["state"] == "NJ"
        assert profile["household_size"] == 3
        assert profile["income_min_cents"] < profile["income_max_cents"]
        assert profile["age"] is None, "the narrative does not give an age"
        assert "age" in journey.result.intake["unknown_fields"]

    def test_the_run_paused_on_the_blocking_field_and_then_completed(
        self, journey: Journey
    ) -> None:
        paused = journey.result.paused
        assert paused["status"] == WorkflowStatus.AWAITING_INPUT.value
        assert [q["field"] for q in paused["clarification_questions"]] == ["age"]
        assert journey.result.completed["status"] == WorkflowStatus.COMPLETED.value
        assert journey.result.completed["profile"]["age"] == 34

    def test_every_match_is_one_of_exactly_three_statuses(self, journey: Journey) -> None:
        allowed = {status.value for status in MatchStatus}
        matches = journey.result.completed["matches"]
        assert matches
        for match in matches:
            assert match["status"] in allowed
            assert match["status"] == match["evaluation"]["status"], (
                "the status a user reads must be the one the engine decided"
            )
            assert match["evaluation"]["evidence"], "a status without evidence is an opinion"

    def test_the_plan_collapsed_the_shared_document(self, journey: Journey) -> None:
        plan = journey.result.plan
        shared = {doc["dedup_key"]: doc for doc in plan["shared_documents"]}
        assert "document:proof-of-address" in shared
        assert len(shared["document:proof-of-address"]["program_ids"]) == 2
        keys = [item["dedup_key"] for item in plan["items"]]
        assert len(keys) == len(set(keys)), "a collapsed document appears once"

    def test_the_edit_regenerated_one_half_and_preserved_the_other(self, journey: Journey) -> None:
        before = {m["program"]["slug"]: m for m in journey.result.completed["matches"]}
        after = {m["program"]["slug"]: m for m in journey.result.reevaluated["matches"]}
        assert after[UTILITY_SLUG]["explanation"] == before[UTILITY_SLUG]["explanation"]
        assert after[GROCERY_SLUG]["explanation"] is None
        assert journey.result.new_plan["id"] != journey.result.plan["id"]
        assert journey.result.new_plan["run_id"] == str(journey.result.reevaluated_run_id)

    def test_every_citation_resolved_to_a_stored_source(self, journey: Journey) -> None:
        chat = journey.result.chat
        assert chat["citations"], "the journey would have failed, but assert it here too"
        assert chat["is_grounded"] is True
        for citation in chat["citations"]:
            assert citation["source_id"]
            assert citation["url"].startswith("https://example.org/")

    def test_deletion_left_no_user_rows(self, journey: Journey) -> None:
        remaining = journey.result.remaining_rows
        assert remaining["workflow_runs"] == 0
        assert remaining["profiles"] == 0
        assert remaining["program_matches"] == 0
        assert remaining["action_plans"] == 0
        assert remaining["action_items"] == 0
        assert remaining["checkpoints"] == 0, "the narrative lives in the checkpoints"
        # One audit row survives on purpose: the record that the deletion
        # happened. The receipt says so.
        assert remaining["audit_events"] <= 1
        assert any("audit" in line.lower() for line in journey.result.receipt["retained"])

    def test_no_fixture_reply_was_labelled_as_live_model_output(self, journey: Journey) -> None:
        assert journey.result.model_sources
        assert set(journey.result.model_sources) <= {"cache", "fallback"}


class TestTheLogsOfAFullRun:
    """The only place a whole workflow's logging is checked for leakage."""

    def test_nothing_personal_reached_a_log_line(self, journey: Journey) -> None:
        logs = journey.logs
        assert logs, "the capture produced nothing, so this asserts nothing"
        for fragment in NARRATIVE_FRAGMENTS:
            assert fragment not in logs, f"the narrative leaked: {fragment!r}"
        for fragment in INCOME_FRAGMENTS:
            assert fragment not in logs, f"an income figure leaked: {fragment!r}"
        assert NARRATIVE not in logs
        assert journey.result.user_id not in logs, "a raw user id leaked"

    def test_the_user_is_identified_by_a_hash_instead(self, journey: Journey) -> None:
        from app.core.logging import hash_user_id

        assert hash_user_id(journey.result.user_id) in journey.logs, (
            "requests must still be attributable, just not to a raw id"
        )

    def test_the_run_id_is_present_so_a_failure_is_debuggable(self, journey: Journey) -> None:
        assert str(journey.result.run_id) in journey.logs

    def test_no_other_uuid_shaped_subject_appears(self, journey: Journey) -> None:
        """A sanity check on the previous test rather than a new property.

        The raw user id is a UUID, so "no raw user id" is only meaningful if
        the assertion would actually have caught one. This proves the capture
        contains UUIDs at all.
        """
        assert any(
            len(token.strip("\"',=")) == 36 and _is_uuid(token.strip("\"',="))
            for token in journey.logs.replace("\n", " ").split(" ")
        )


def _is_uuid(token: str) -> bool:
    try:
        uuid.UUID(token)
    except ValueError:
        return False
    return True
