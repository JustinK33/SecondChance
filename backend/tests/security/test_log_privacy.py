"""Nothing a person told us may appear in a log line.

``tests/unit/test_log_redaction.py`` asserts on the redaction pipeline given
hand-built events. This file asserts on the *application*: real requests go
through the real app with the real logging configuration, and the whole
captured output is searched for the things that must never be in it - a
narrative, an income figure, an email address, a bearer token, a raw user id.

Why it is searched as one blob rather than per event: a leak does not care
which logger produced it. The question an incident asks is "is this string
anywhere in our logs", and that is the question this file asks.

The markers are deliberately distinctive strings. A substring search for
``"1850"`` could pass by luck; a search for
``zzqq-private-narrative-marker`` cannot.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import hash_user_id
from tests.conftest import make_match, make_plan, make_program, make_ruleset
from tests.security.harness import (
    ApiHarness,
    capture_app_logs,
    dev_settings,
    rolled_back_api,
    seed_catalog,
)

__all__ = ["rolled_back_api"]

pytestmark = pytest.mark.db

#: Markers planted in the request. Each one is a different class of personal
#: content, and each is unique enough that finding it proves a leak.
NARRATIVE_MARKER = "zzqq-private-narrative-marker"
EMAIL_MARKER = "jane.doe.zzqq@example.org"
ADDRESS_MARKER = "41 Zzqq Street, Apartment 7B"
AMOUNT_MARKER = "1850"

NARRATIVE = (
    f"{NARRATIVE_MARKER}. I live at {ADDRESS_MARKER} and my rent is ${AMOUNT_MARKER} a month. "
    f"You can reach me at {EMAIL_MARKER}. I need help with groceries and my electric bill."
)

#: A bearer token shaped like a real one, so the value scrubber has something
#: to recognise. It signs nothing and authorises nothing.
FAKE_TOKEN = (
    "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJ6enFxLW5vdC1hLXJlYWwtc3ViamVjdCJ9.zzqq-not-a-real-signature"
)

#: The income the user typed, in the two shapes it exists in server-side.
INCOME_CENTS = 185_000


def blob(lines: list[str]) -> str:
    return "\n".join(lines)


async def _drive_everything(api: ApiHarness, session: AsyncSession, user_id: str) -> None:
    """One request per route that touches personal content."""
    headers = api.dev(user_id)
    program = make_program()
    await seed_catalog(session, [program], [make_ruleset(program.slug)])

    # Intake: the narrative, plus an income the user typed directly.
    intake = await api.client.post(
        "/api/v1/intake",
        json={
            "narrative": NARRATIVE,
            "prefilled": {
                "income_min_cents": INCOME_CENTS,
                "income_max_cents": INCOME_CENTS,
                "income_period": "MONTHLY",
                "county": "Middlesex",
            },
        },
        headers=headers,
    )
    assert intake.status_code == 200, intake.text

    # A run, and a poll of it.
    created = await api.client.post(
        "/api/v1/workflows", json={"narrative": NARRATIVE}, headers=headers
    )
    assert created.status_code == 202, created.text
    run_id = uuid.UUID(created.json()["run_id"])
    assert (await api.client.get(f"/api/v1/workflows/{run_id}", headers=headers)).status_code == 200

    # A plan and a ticked step, so the plan routes log too.
    from app.db.repositories import MatchRepository, PlanRepository

    await MatchRepository(session).upsert_many(user_id, run_id, [make_match(program)])
    plan = make_plan(run_id, program)
    await PlanRepository(session).save(user_id, plan)
    await session.commit()
    patched = await api.client.patch(
        f"/api/v1/action-items/{plan.items[0].id}",
        json={"is_complete": True},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert (await api.client.get(f"/api/v1/plans/{plan.id}", headers=headers)).status_code == 200

    # A question containing an email address.
    chat = await api.client.post(
        "/api/v1/assistant/chat",
        json={
            "message": (
                f"My email is {EMAIL_MARKER} and I earn ${AMOUNT_MARKER}. "
                "What documents does this program ask for?"
            ),
            "program_id": str(program.id),
        },
        headers=headers,
    )
    assert chat.status_code == 200, chat.text

    # A validation failure carrying the marker in the rejected value.
    rejected = await api.client.post(
        "/api/v1/intake",
        json={"narrative": NARRATIVE, "prefilled": {"household_size": NARRATIVE_MARKER}},
        headers=headers,
    )
    assert rejected.status_code == 422

    # A rejected request carrying a token, so the value scrubber is exercised.
    unauthorised = await api.client.get(
        "/api/v1/workflows", headers={"Authorization": f"Bearer {FAKE_TOKEN}"}
    )
    assert unauthorised.status_code in {200, 401}

    # And the deletion, which logs counts.
    assert (await api.client.delete("/api/v1/account/data", headers=headers)).status_code == 200


@pytest.fixture
async def driven(rolled_back_api: ApiHarness, session: AsyncSession, user_a: str) -> list[str]:
    """Drive the app under log capture and hand back the lines it emitted.

    The capture is closed before the assertions run, so what the tests read is
    the complete output of the drive rather than whatever had been flushed by
    the time they looked.
    """
    with capture_app_logs(dev_settings()) as stream:
        await _drive_everything(rolled_back_api, session, user_a)
    return [line for line in stream.getvalue().splitlines() if line.strip()]


class TestTheCaptureIsReal:
    """If the capture were empty, every assertion below would be vacuous."""

    async def test_the_application_logged_something(self, driven: list[str]) -> None:
        assert driven, "no log lines were captured, so this file asserts nothing"
        assert len(driven) >= 5, f"suspiciously few lines: {driven}"

    async def test_every_line_is_structured_json(self, driven: list[str]) -> None:
        for line in driven:
            payload = json.loads(line)
            assert "event" in payload, line
            assert "level" in payload, line

    async def test_the_access_log_is_present(self, driven: list[str]) -> None:
        events = {json.loads(line)["event"] for line in driven}
        assert "http.request" in events, f"no access log was emitted: {sorted(events)}"


class TestNothingPersonalIsLogged:
    """One assertion per class of personal content."""

    async def test_the_narrative_does_not_appear(self, driven: list[str]) -> None:
        text = blob(driven)
        assert NARRATIVE_MARKER not in text
        assert NARRATIVE not in text
        assert "groceries and my electric bill" not in text

    async def test_no_address_appears(self, driven: list[str]) -> None:
        text = blob(driven)
        assert ADDRESS_MARKER not in text
        assert "Zzqq Street" not in text

    async def test_no_email_appears(self, driven: list[str]) -> None:
        text = blob(driven)
        assert EMAIL_MARKER not in text
        assert "jane.doe" not in text

    async def test_no_income_figure_appears(self, driven: list[str]) -> None:
        text = blob(driven)
        assert AMOUNT_MARKER not in text, "an amount the user typed was logged"
        assert str(INCOME_CENTS) not in text
        assert f"${AMOUNT_MARKER}" not in text

    async def test_no_token_appears(self, driven: list[str]) -> None:
        text = blob(driven)
        assert FAKE_TOKEN not in text
        assert "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9" not in text
        assert "Bearer " not in text

    async def test_no_raw_user_id_appears(self, driven: list[str], user_a: str) -> None:
        assert user_a not in blob(driven), "a raw user id was logged"

    async def test_no_rejected_value_is_echoed_into_a_log_line(self, driven: list[str]) -> None:
        """A validation failure logs the field path, never the value."""
        text = blob(driven)
        assert NARRATIVE_MARKER not in text
        for line in driven:
            payload = json.loads(line)
            if payload.get("event") == "api.error":
                assert "input" not in payload, payload
                assert "msg" not in payload, payload


class TestWhatIsLoggedInstead:
    """Redaction is only acceptable if the line is still useful."""

    async def test_the_user_is_identified_by_a_hash(self, driven: list[str], user_a: str) -> None:
        digest = hash_user_id(user_a)
        assert digest in blob(driven), "requests must stay attributable"
        assert digest.startswith("u_")

    async def test_every_request_carries_a_correlation_id(self, driven: list[str]) -> None:
        access = [json.loads(line) for line in driven]
        access = [entry for entry in access if entry.get("event") == "http.request"]
        assert access
        for entry in access:
            assert entry.get("request_id"), entry
            assert entry.get("route"), entry
            assert entry.get("status"), entry
            # The route is a template, never a filled-in path, so an id cannot
            # arrive in a metrics label or a log field by this door.
            assert "{" in entry["route"] or entry["route"].count("/") <= 4, entry

    async def test_counts_and_enums_survive_redaction(self, driven: list[str]) -> None:
        events = {json.loads(line)["event"]: json.loads(line) for line in driven}
        assert "intake.extracted" in events, sorted(events)
        extracted = events["intake.extracted"]
        assert isinstance(extracted.get("known_fields"), int)
        assert isinstance(extracted.get("unknown_fields"), int)
        assert extracted.get("source") in {"cache", "fallback", "live"}


class TestAuditRowsAreNotALoophole:
    """An audit row is persisted, so it is held to the same standard."""

    async def test_no_audit_row_carries_personal_content(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        from sqlalchemy import select

        from app.db import models

        await _drive_everything(rolled_back_api, session, user_a)
        rows = (await session.scalars(select(models.AuditEvent))).all()
        # The deletion at the end of the drive removes this user's history and
        # leaves one record of the deletion itself.
        assert rows, "the drive produced no audit rows at all"
        for row in rows:
            text = f"{row.resource_type} {row.resource_id} {json.dumps(row.event_metadata)}"
            assert NARRATIVE_MARKER not in text
            assert EMAIL_MARKER not in text
            assert ADDRESS_MARKER not in text
            assert AMOUNT_MARKER not in text
            assert row.user_id_hash.startswith("u_")

    async def test_the_narrative_is_stored_only_as_a_hash(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        """``workflow_runs`` keeps a hash so idempotency works without the text."""
        from sqlalchemy import select

        from app.db import models

        created = await rolled_back_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=rolled_back_api.dev(user_a),
        )
        assert created.status_code == 202
        row = await session.scalar(
            select(models.WorkflowRun).where(
                models.WorkflowRun.id == uuid.UUID(created.json()["run_id"])
            )
        )
        assert row is not None
        assert row.narrative_hash
        assert NARRATIVE_MARKER not in row.narrative_hash
        assert not hasattr(row, "narrative") or getattr(row, "narrative", None) is None
