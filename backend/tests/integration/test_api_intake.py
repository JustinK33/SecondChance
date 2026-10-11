"""``POST /api/v1/intake`` - the cheap half of onboarding.

One model call, no retrieval, no eligibility, no model-written prose. It
exists so a user can correct a misunderstanding before an expensive run is
spent on it, which makes two things worth testing precisely: the response has
to say what was *not* understood as clearly as what was, and a model failure
has to arrive as an honest retryable error rather than as an empty profile
presented as a reading of the narrative.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.fake import FailureMode, FakeGeminiClient
from app.ai.gemini import set_gemini_port_override
from app.ai.schemas import GeminiTask
from app.api.v1.deps import provide_gemini
from app.db import models
from app.schemas import IntakeResponse
from app.schemas.enums import EmploymentStatus, NeedCategory
from app.services.ai_port import DegradingGeminiPort, build_ai_port, reset_ai_port
from tests.security.harness import ApiHarness, dev_settings, rolled_back_api

__all__ = ["rolled_back_api"]

pytestmark = pytest.mark.db

NARRATIVE = (
    "I live in New Brunswick, New Jersey with my two kids. I work part time and make about "
    "$2,000 a month. I need help with groceries and my electric bill."
)


def error_of(response) -> dict:
    body = response.json()
    assert set(body) == {"error"}, body
    return body["error"]


class TestTheHappyPath:
    async def test_the_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/intake", json={"narrative": NARRATIVE}, headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 200, response.text
        intake = IntakeResponse.model_validate(response.json())
        assert intake.profile.state == "NJ"
        assert intake.profile.household_size == 3
        assert intake.extraction_source in {"cache", "fallback", "live"}
        assert len(intake.clarification_questions) <= 5

    async def test_what_was_not_understood_is_reported_rather_than_guessed(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/intake", json={"narrative": NARRATIVE}, headers=rolled_back_api.dev(user_a)
        )
        intake = IntakeResponse.model_validate(response.json())
        assert "age" in intake.unknown_fields
        assert intake.profile.age is None
        assert intake.ambiguities, "an approximate figure is an ambiguity, not a fact"

    async def test_an_approximate_amount_stays_a_range(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        """ "About $2,000" is a range. Collapsing it to a point would turn an
        honest ``MORE_INFO_NEEDED`` into a confident wrong answer."""
        response = await rolled_back_api.client.post(
            "/api/v1/intake", json={"narrative": NARRATIVE}, headers=rolled_back_api.dev(user_a)
        )
        profile = IntakeResponse.model_validate(response.json()).profile
        assert profile.income_min_cents is not None
        assert profile.income_max_cents is not None
        assert profile.income_min_cents < profile.income_max_cents
        assert profile.income_period is not None

    async def test_the_only_side_effect_is_a_profile_version(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        for _ in range(2):
            response = await rolled_back_api.client.post(
                "/api/v1/intake",
                json={"narrative": NARRATIVE},
                headers=rolled_back_api.dev(user_a),
            )
            assert response.status_code == 200
        profiles = await session.scalar(
            select(func.count()).select_from(models.Profile).where(models.Profile.user_id == user_a)
        )
        runs = await session.scalar(
            select(func.count())
            .select_from(models.WorkflowRun)
            .where(models.WorkflowRun.user_id == user_a)
        )
        assert profiles >= 1
        assert runs == 0, "intake must not start a run"
        assert rolled_back_api.runner.scheduled == []

    async def test_no_question_asks_for_something_no_program_needs(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        # The fixture port replays per task, so the sparse reading - one need
        # and everything else unknown - has to be selected explicitly.
        rolled_back_api.gemini.cases[GeminiTask.EXTRACT_PROFILE] = "sparse"
        response = await rolled_back_api.client.post(
            "/api/v1/intake",
            json={"narrative": "Things are tight and I cannot keep food in the house."},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 200
        intake = IntakeResponse.model_validate(response.json())
        assert intake.clarification_questions, "a sparse narrative should produce questions"
        assert [q.field for q in intake.clarification_questions] == [
            "state",
            "household_size",
            "income",
        ], "the questions are the material fields, in a deterministic order"
        banned = (
            "social security",
            "ssn",
            "bank account",
            "routing number",
            "password",
            "immigration",
        )
        for question in intake.clarification_questions:
            text = f"{question.question} {question.why_it_matters}".lower()
            for term in banned:
                assert term not in text, f"a question asked for {term!r}"
            assert question.is_required is False, "every question must be skippable"
            assert question.why_it_matters, "a question must say what it would change"


class TestValidation:
    @pytest.mark.parametrize(
        ("body", "field"),
        [
            ({}, "narrative"),
            ({"narrative": ""}, "narrative"),
            ({"narrative": "too short"}, "narrative"),
            ({"narrative": "x" * 20_000}, "narrative"),
            ({"narrative": NARRATIVE, "prefilled": {"state": "nj!"}}, "prefilled.state"),
            ({"narrative": NARRATIVE, "prefilled": {"age": 200}}, "prefilled.age"),
            ({"narrative": NARRATIVE, "prefilled": {"needs": ["RENT_MONEY"]}}, "prefilled.needs.0"),
            ({"narrative": NARRATIVE, "extra": True}, "extra"),
        ],
    )
    async def test_a_bad_body_is_a_validation_error_naming_only_the_field(
        self, rolled_back_api: ApiHarness, user_a: str, body: dict, field: str
    ) -> None:
        response = await rolled_back_api.client.post(
            "/api/v1/intake", json=body, headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 422, response.text
        error = error_of(response)
        assert error["code"] == "validation_error"
        assert field in error["detail"]["fields"], error["detail"]

    async def test_a_rejected_value_is_never_echoed_back(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        marker = "zzqq-rejected-value-marker"
        response = await rolled_back_api.client.post(
            "/api/v1/intake",
            json={"narrative": NARRATIVE, "prefilled": {"county": marker * 10}},
            headers=rolled_back_api.dev(user_a),
        )
        assert response.status_code == 422
        assert marker not in response.text


class TestModelFailures:
    """A failure must look like a failure, not like an empty reading."""

    @pytest.mark.parametrize(
        ("mode", "status", "code", "retryable"),
        [
            (FailureMode.TIMEOUT, 504, "upstream_timeout", True),
            (FailureMode.RATE_LIMITED, 429, "rate_limited", True),
            (FailureMode.UNAVAILABLE, 503, "upstream_unavailable", True),
            (FailureMode.MALFORMED_JSON, 502, "invalid_model_output", True),
            (FailureMode.SCHEMA_VIOLATION, 502, "invalid_model_output", True),
            (FailureMode.EMPTY_RESPONSE, 502, "invalid_model_output", True),
        ],
    )
    async def test_a_model_failure_maps_to_the_documented_status(
        self,
        rolled_back_api: ApiHarness,
        session: AsyncSession,
        user_a: str,
        mode: FailureMode,
        status: int,
        code: str,
        retryable: bool,
    ) -> None:
        rolled_back_api.gemini.failure_mode = mode
        response = await rolled_back_api.client.post(
            "/api/v1/intake", json={"narrative": NARRATIVE}, headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == status, response.text
        error = error_of(response)
        assert error["code"] == code
        assert error["message"], "a failure must still be renderable to a user"
        assert NARRATIVE not in response.text, "the prompt must not come back in an error"

        saved = await session.scalar(
            select(func.count()).select_from(models.Profile).where(models.Profile.user_id == user_a)
        )
        assert saved == 0, "a failed extraction must not save a profile"

    async def test_a_rate_limit_carries_a_retry_after_header(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        rolled_back_api.gemini.failure_mode = FailureMode.RATE_LIMITED
        response = await rolled_back_api.client.post(
            "/api/v1/intake", json={"narrative": NARRATIVE}, headers=rolled_back_api.dev(user_a)
        )
        assert response.status_code == 429
        assert "Retry-After" in response.headers, "a 429 without one is not actionable"


class TestTheDegradedProductionPath:
    """The gap that let a fabricated profile reach users.

    Every other test in this file injects :class:`~app.ai.fake.FakeGeminiClient`
    as the model port, which is correct for a hermetic test and is exactly why
    this defect survived: nothing exercised the port the *application* builds.
    It built a fixture replayer too, so when the live model answered ``404`` -
    which it does for every call with this project's key today - intake returned
    the recorded fixture's household to whoever had typed, with
    ``extraction_source: "fallback"`` on it.

    These tests therefore use ``build_ai_port``, the function the app itself
    calls, with the live client failing the way it really fails.
    """

    NARRATIVE = (
        "I am a 71 year old retired widow living alone in Camden County and I need help "
        "paying my heating bill this winter."
    )
    OTHER_NARRATIVE = (
        "I live in Essex County, New Jersey with my three kids. I lost my job and I need "
        "help with groceries."
    )

    @contextmanager
    def production_port(self, harness: ApiHarness) -> Iterator[DegradingGeminiPort]:
        """The real port, wired the way ``provide_gemini`` wires it.

        The live leg is a fixture client stuck in ``unavailable``, which raises
        the same :class:`~app.ai.errors.ModelUnavailable` the retired model
        raises. Its *fixtures are never read*: the point is that the live call
        fails and the non-model port answers instead.
        """
        settings = dev_settings(gemini_api_key="test-key-not-a-real-credential")
        set_gemini_port_override(FakeGeminiClient(settings, failure_mode=FailureMode.UNAVAILABLE))
        try:
            port = build_ai_port(settings)
            assert isinstance(port, DegradingGeminiPort)
            assert port.has_live_client, "the live leg must be tried before degrading"
            harness.app.dependency_overrides[provide_gemini] = lambda: port
            yield port
        finally:
            set_gemini_port_override(None)
            reset_ai_port()

    async def test_the_degraded_answer_contains_nothing_the_narrative_did_not_say(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        with self.production_port(rolled_back_api) as port:
            response = await rolled_back_api.client.post(
                "/api/v1/intake",
                json={"narrative": self.NARRATIVE},
                headers=rolled_back_api.dev(user_a),
            )

        assert response.status_code == 200, response.text
        assert port.failures == 1, "the live model must actually have been attempted"
        intake = IntakeResponse.model_validate(response.json())
        assert intake.extraction_source == "fallback"

        profile = intake.profile
        # Stated in the narrative, in the narrative's own words.
        assert profile.county == "Camden"
        assert profile.age == 71
        assert profile.household_size == 1
        assert profile.employment_status is EmploymentStatus.RETIRED
        assert profile.needs == [NeedCategory.UTILITIES]
        # Not stated anywhere in it. Each of these came back populated from the
        # recorded fixture when this path was the fake.
        assert profile.state is None, "the narrative names a county, never a state"
        assert profile.student_status is None
        assert profile.income_min_cents is None
        assert profile.income_max_cents is None
        assert profile.income_period is None
        assert profile.housing_situation is None
        assert profile.flags == {}
        assert intake.ambiguities == []
        assert {"state", "income"} <= set(intake.unknown_fields)

    async def test_two_narratives_do_not_come_back_as_the_same_profile(
        self, rolled_back_api: ApiHarness, user_a: str
    ) -> None:
        """The fixture replayer returned byte-identical bodies here."""
        with self.production_port(rolled_back_api):
            first = await rolled_back_api.client.post(
                "/api/v1/intake",
                json={"narrative": self.NARRATIVE},
                headers=rolled_back_api.dev(user_a),
            )
            second = await rolled_back_api.client.post(
                "/api/v1/intake",
                json={"narrative": self.OTHER_NARRATIVE},
                headers=rolled_back_api.dev(user_a),
            )

        assert first.status_code == second.status_code == 200
        one = IntakeResponse.model_validate(first.json()).profile
        two = IntakeResponse.model_validate(second.json()).profile
        assert one.model_dump_json() != two.model_dump_json()
        assert (one.county, two.county) == ("Camden", "Essex")
        assert (one.household_size, two.household_size) == (1, 4)
        assert two.employment_status is EmploymentStatus.UNEMPLOYED

    async def test_the_degraded_answer_is_saved_and_labelled_as_non_model_output(
        self, rolled_back_api: ApiHarness, session: AsyncSession, user_a: str
    ) -> None:
        with self.production_port(rolled_back_api):
            response = await rolled_back_api.client.post(
                "/api/v1/intake",
                json={"narrative": self.NARRATIVE},
                headers=rolled_back_api.dev(user_a),
            )

        assert response.status_code == 200
        saved = await session.scalar(
            select(func.count()).select_from(models.Profile).where(models.Profile.user_id == user_a)
        )
        assert saved == 1
        audit = await session.scalar(
            select(models.AuditEvent)
            .where(models.AuditEvent.user_id == user_a)
            .order_by(models.AuditEvent.created_at.desc())
            .limit(1)
        )
        assert audit is not None
        assert audit.event_metadata["extraction_source"] == "fallback", (
            "the audit trail has to record that no model wrote this"
        )
