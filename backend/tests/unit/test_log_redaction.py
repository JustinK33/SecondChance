"""Privacy tests for the logging pipeline.

These are the tests that make the product's privacy promise checkable. Every
case here renders through the **real** pipeline installed by
:func:`app.core.logging.configure_logging` - the same processor chain, the same
JSON renderer - into a buffer instead of stdout, and then asserts on the exact
bytes that would have been written to a log sink.

The bar is deliberately crude and absolute: the secret string must not appear
*anywhere* in the emitted line, regardless of which key it arrived under.
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from app.core.config import Settings
from app.core.logging import (
    REDACTED,
    bind_request_context,
    bound_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
    hash_user_id,
    new_request_id,
    redact_event_dict,
)

# Fictional sample data. Nothing here is a real person, address or credential.
NARRATIVE = (
    "I'm a Rutgers student living with my family in New Brunswick. We're "
    "struggling with groceries and the electric bill and I don't know what we "
    "qualify for."
)
EMAIL = "sample.person@example.com"
PHONE = "732-555-0147"
ADDRESS = "17 Fictional Lane, New Brunswick, NJ 08901"
BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJub3QtcmVhbCJ9.ZmFrZS1zaWduYXR1cmU"
RAW_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJub3QtcmVhbCJ9.ZmFrZS1zaWduYXR1cmU"
FAKE_API_KEY = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY0000"
RAW_USER_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
INCOME_CENTS = 245_000
INCOME_TEXT = "$2,450 a month"
DB_URL = "postgresql://secondchance:notarealpassword@localhost:5433/secondchance"

SECRETS = (
    NARRATIVE,
    EMAIL,
    PHONE,
    ADDRESS,
    BEARER,
    RAW_JWT,
    FAKE_API_KEY,
    RAW_USER_ID,
    str(INCOME_CENTS),
    "2,450",
    "notarealpassword",
)


def _settings(**overrides: Any) -> Settings:
    """Settings that do not read the developer's ``.env``."""
    base: dict[str, Any] = {
        "app_env": "local",
        "auth_mode": "dev",
        "log_level": "DEBUG",
        "log_max_text_chars": 200,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class LogCapture:
    """Buffer holding the real JSON lines the pipeline produced."""

    def __init__(self, buffer: io.StringIO) -> None:
        self.buffer = buffer

    @property
    def raw(self) -> str:
        return self.buffer.getvalue()

    def lines(self) -> list[dict[str, Any]]:
        parsed: list[dict[str, Any]] = []
        for line in self.raw.splitlines():
            if line.strip():
                parsed.append(json.loads(line))
        return parsed

    def only(self) -> dict[str, Any]:
        lines = self.lines()
        assert len(lines) == 1, f"expected exactly one log line, got {len(lines)}"
        return lines[0]


@pytest.fixture
def capture() -> Iterator[LogCapture]:
    """Install the real pipeline, writing to a buffer instead of stdout."""
    buffer = io.StringIO()
    configure_logging(_settings(), stream=buffer)
    clear_request_context()
    try:
        yield LogCapture(buffer)
    finally:
        clear_request_context()
        structlog.reset_defaults()
        logging.basicConfig(force=True)


def assert_no_secrets(raw: str) -> None:
    leaked = [secret for secret in SECRETS if secret in raw]
    assert not leaked, f"log output leaked {len(leaked)} secret value(s)"


class TestNothingPersonalIsLogged:
    def test_a_realistic_leaky_log_call_emits_nothing_personal(self, capture: LogCapture) -> None:
        """The event a careless caller actually writes, with everything in it."""
        bind_request_context(request_id="req-abc123", user_id_hash=hash_user_id(RAW_USER_ID))
        logger = get_logger(__name__)
        logger.info(
            "intake received",
            narrative=NARRATIVE,
            income_min_cents=INCOME_CENTS,
            income_max_cents=INCOME_CENTS,
            email=EMAIL,
            phone=PHONE,
            address=ADDRESS,
            authorization=BEARER,
            api_key=FAKE_API_KEY,
            user_id=RAW_USER_ID,
            answers=[{"field": "household_size", "value": 4}],
            program_count=12,
        )
        assert_no_secrets(capture.raw)
        line = capture.only()
        assert line["event"] == "intake received"
        assert line["request_id"] == "req-abc123"
        assert line["user_id_hash"] == hash_user_id(RAW_USER_ID)
        assert line["program_count"] == 12
        for key in (
            "narrative",
            "income_min_cents",
            "income_max_cents",
            "email",
            "phone",
            "address",
            "authorization",
            "api_key",
            "user_id",
            "answers",
        ):
            assert line[key] == REDACTED, f"{key} was not redacted"

    @pytest.mark.parametrize(
        "field",
        [
            "narrative",
            "raw_text",
            "income",
            "income_min_cents",
            "income_max_cents",
            "address",
            "email",
            "phone",
            "token",
            "access_token",
            "authorization",
            "api_key",
            "jwt",
            "password",
            "answers",
            "clarification_answers",
            "profile",
        ],
    )
    def test_every_denylisted_field_is_masked(self, capture: LogCapture, field: str) -> None:
        get_logger(__name__).info("event", **{field: "SENSITIVE-VALUE-SHOULD-NOT-APPEAR"})
        assert "SENSITIVE-VALUE-SHOULD-NOT-APPEAR" not in capture.raw
        assert capture.only()[field] == REDACTED

    @pytest.mark.parametrize(
        "field",
        ["gemini_api_key", "supabase_jwt_secret", "user_narrative", "household_income", "id_token"],
    )
    def test_denylist_fragments_catch_variant_spellings(
        self, capture: LogCapture, field: str
    ) -> None:
        get_logger(__name__).info("event", **{field: "SENSITIVE-VALUE-SHOULD-NOT-APPEAR"})
        assert capture.only()[field] == REDACTED

    def test_nested_structures_are_redacted_recursively(self, capture: LogCapture) -> None:
        """The usual shape of a real leak: a whole state dict passed as context."""
        get_logger(__name__).info(
            "workflow state",
            state={
                "run_id": "run-1",
                "profile": {"narrative": NARRATIVE, "income_min_cents": INCOME_CENTS},
                "candidates": [
                    {"slug": "nj-snap", "profile": {"email": EMAIL}},
                ],
            },
        )
        assert_no_secrets(capture.raw)
        state = capture.only()["state"]
        assert state["run_id"] == "run-1"
        assert state["profile"] == REDACTED
        assert state["candidates"][0]["slug"] == "nj-snap"
        assert state["candidates"][0]["profile"] == REDACTED

    def test_pydantic_models_are_not_dumped_verbatim(self, capture: LogCapture) -> None:
        from app.schemas import IncomePeriod, UserProfile

        profile = UserProfile(
            income_min_cents=INCOME_CENTS,
            income_max_cents=INCOME_CENTS,
            income_period=IncomePeriod.MONTHLY,
        )
        get_logger(__name__).info("evaluating", user_profile=profile)
        assert_no_secrets(capture.raw)
        assert capture.only()["user_profile"] == REDACTED


class TestSecretsAreScrubbedByValue:
    """Second layer: shapes that are sensitive whatever key they arrive under."""

    def test_a_bearer_token_in_an_allowed_field_is_scrubbed(self, capture: LogCapture) -> None:
        get_logger(__name__).warning("auth failed", header_preview=f"Authorization: {BEARER}")
        assert_no_secrets(capture.raw)
        assert REDACTED in capture.only()["header_preview"]

    def test_a_bare_jwt_is_scrubbed(self, capture: LogCapture) -> None:
        get_logger(__name__).warning("token rejected", detail=f"could not verify {RAW_JWT}")
        assert RAW_JWT not in capture.raw

    def test_a_google_api_key_is_scrubbed(self, capture: LogCapture) -> None:
        get_logger(__name__).error("model call failed", upstream_url=f"?key={FAKE_API_KEY}")
        assert FAKE_API_KEY not in capture.raw

    def test_a_database_url_with_a_password_is_scrubbed(self, capture: LogCapture) -> None:
        get_logger(__name__).error("connect failed", target=DB_URL)
        assert "notarealpassword" not in capture.raw

    def test_an_email_in_the_event_message_is_scrubbed(self, capture: LogCapture) -> None:
        get_logger(__name__).info(f"could not notify {EMAIL}")
        assert EMAIL not in capture.raw

    def test_a_money_amount_in_prose_is_scrubbed(self, capture: LogCapture) -> None:
        get_logger(__name__).info("evaluated", note=f"household reported {INCOME_TEXT}")
        assert "2,450" not in capture.raw

    def test_an_exception_message_carrying_a_secret_is_scrubbed(self, capture: LogCapture) -> None:
        """Tracebacks are kept for debuggability, but still scrubbed."""
        try:
            raise RuntimeError(f"could not connect to {DB_URL}")
        except RuntimeError:
            get_logger(__name__).exception("database failure")
        raw = capture.raw
        assert "notarealpassword" not in raw
        assert "RuntimeError" in raw


class TestTruncation:
    def test_long_free_text_in_an_unanticipated_field_is_truncated(
        self, capture: LogCapture
    ) -> None:
        """Third layer: a field nobody thought of cannot dump a document."""
        long_text = "lorem ipsum " * 200
        get_logger(__name__).info("surprising field", some_new_field=long_text)
        value = capture.only()["some_new_field"]
        assert len(value) < len(long_text)
        assert value.endswith("[truncated]")

    def test_the_limit_is_configurable(self) -> None:
        cleaned = redact_event_dict({"event": "x" * 500}, max_text_chars=32)
        assert cleaned["event"].startswith("x" * 32)
        assert len(cleaned["event"]) == 32 + len("...[truncated]")

    def test_tracebacks_are_exempt_from_truncation(self) -> None:
        traceback = "Traceback (most recent call last):\n" + ("  frame\n" * 100)
        cleaned = redact_event_dict({"exception": traceback}, max_text_chars=50)
        assert cleaned["exception"] == traceback

    def test_short_values_are_untouched(self) -> None:
        cleaned = redact_event_dict({"event": "ready", "count": 3}, max_text_chars=200)
        assert cleaned == {"event": "ready", "count": 3}


class TestUserIdentity:
    def test_the_hash_is_stable_and_does_not_contain_the_raw_id(self) -> None:
        first = hash_user_id(RAW_USER_ID)
        assert first == hash_user_id(uuid.UUID(RAW_USER_ID))
        assert RAW_USER_ID not in first
        assert first.startswith("u_")

    def test_different_users_hash_differently(self) -> None:
        assert hash_user_id(RAW_USER_ID) != hash_user_id(str(uuid.uuid4()))

    def test_binding_a_raw_id_by_mistake_still_hashes_it(self, capture: LogCapture) -> None:
        """Defence in depth: the convenient mistake must not be the unsafe one."""
        bind_request_context(request_id="req-1", user_id_hash=RAW_USER_ID)
        get_logger(__name__).info("bound")
        line = capture.only()
        assert RAW_USER_ID not in capture.raw
        assert line["user_id_hash"] == hash_user_id(RAW_USER_ID)

    def test_an_already_hashed_identity_is_not_double_hashed(self, capture: LogCapture) -> None:
        digest = hash_user_id(RAW_USER_ID)
        bind_request_context(request_id="req-1", user_id_hash=digest)
        get_logger(__name__).info("bound")
        assert capture.only()["user_id_hash"] == digest


class TestRequestContext:
    def test_the_request_id_survives_redaction(self, capture: LogCapture) -> None:
        request_id = new_request_id()
        bind_request_context(request_id=request_id)
        get_logger(__name__).info("handled")
        line = capture.only()
        assert line["request_id"] == request_id
        assert "user_id_hash" not in line

    def test_context_is_merged_into_every_later_event(self, capture: LogCapture) -> None:
        bind_request_context(
            request_id="req-7", user_id_hash=hash_user_id(RAW_USER_ID), run_id="run-7"
        )
        logger = get_logger(__name__)
        logger.info("first")
        logger.info("second")
        lines = capture.lines()
        assert len(lines) == 2
        assert all(line["request_id"] == "req-7" for line in lines)
        assert all(line["run_id"] == "run-7" for line in lines)

    def test_clearing_removes_the_context(self, capture: LogCapture) -> None:
        bind_request_context(request_id="req-8")
        clear_request_context()
        get_logger(__name__).info("after clear")
        assert "request_id" not in capture.only()

    def test_the_context_manager_restores_the_previous_context(self, capture: LogCapture) -> None:
        bind_request_context(request_id="outer")
        with bound_request_context(request_id="inner"):
            get_logger(__name__).info("inside")
        get_logger(__name__).info("outside")
        first, second = capture.lines()
        assert first["request_id"] == "inner"
        assert second["request_id"] == "outer"


class TestPipelineShape:
    def test_output_is_one_json_object_per_line(self, capture: LogCapture) -> None:
        logger = get_logger(__name__)
        logger.info("one")
        logger.info("two")
        for line in capture.raw.splitlines():
            assert isinstance(json.loads(line), dict)

    def test_standard_fields_are_present(self, capture: LogCapture) -> None:
        get_logger(__name__).warning("careful")
        line = capture.only()
        assert line["event"] == "careful"
        assert line["level"] == "warning"
        assert line["timestamp"].endswith("Z")

    def test_the_configured_level_filters_events(self) -> None:
        buffer = io.StringIO()
        configure_logging(_settings(log_level="WARNING"), stream=buffer)
        try:
            logger = get_logger("level-test")
            logger.debug("hidden")
            logger.info("hidden")
            logger.warning("shown")
            assert "hidden" not in buffer.getvalue()
            assert "shown" in buffer.getvalue()
        finally:
            structlog.reset_defaults()
            logging.basicConfig(force=True)

    def test_configure_logging_is_idempotent(self, capture: LogCapture) -> None:
        configure_logging(_settings(), stream=capture.buffer)
        configure_logging(_settings(), stream=capture.buffer)
        get_logger(__name__).info("still one line")
        assert len(capture.lines()) == 1
