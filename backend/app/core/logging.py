"""Structured JSON logging with mandatory redaction.

The redaction processor in this module is the product's privacy control, not a
convenience. SecondChance handles income, household details and free-text
narratives from people in financial distress, so the rule is absolute: **no
personal content reaches a log sink.** Observability is built from ids, hashes,
enums, counts and durations.

Three independent layers, so one mistake is not enough to leak anything:

1. **Key denylist** - any event field whose name is in :data:`DENY_KEYS`, or
   contains a fragment in :data:`DENY_KEY_FRAGMENTS`, is replaced with
   ``"[redacted]"``. Applied recursively through nested mappings and sequences,
   because the leak is usually ``extra={"profile": {...}}``.
2. **Value scrubbing** - string values are scanned for shapes that are secret
   or personal no matter which key they arrived under: bearer tokens, JWTs,
   Google API keys, Postgres URLs with credentials, email addresses and money
   amounts.
3. **Truncation** - remaining free text is cut to
   ``settings.log_max_text_chars``, so an unanticipated field cannot dump a
   narrative into the log even if it survives the first two layers.

User identity is logged only as a hash. :func:`hash_user_id` is a plain
truncated SHA-256 of the id: user ids are random UUIDs, which have no
brute-forceable keyspace, so no salt (and therefore no extra secret to manage)
is required. If a raw-looking id is handed to
:func:`bind_request_context`, it is hashed defensively rather than trusted.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Final, TextIO

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, Processor, WrappedLogger

from app.core.config import Settings

__all__ = [
    "DENY_KEYS",
    "DENY_KEY_FRAGMENTS",
    "NO_TRUNCATE_KEYS",
    "REDACTED",
    "USER_HASH_PREFIX",
    "bind_request_context",
    "bound_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "hash_user_id",
    "make_redaction_processor",
    "new_request_id",
    "redact_event_dict",
]

REDACTED: Final[str] = "[redacted]"
TRUNCATION_SUFFIX: Final[str] = "...[truncated]"
USER_HASH_PREFIX: Final[str] = "u_"
_USER_HASH_CHARS: Final[int] = 16
_MAX_DEPTH: Final[int] = 6

#: Exact field names that are never logged. Includes the names the product
#: contract uses for personal data plus the credential names used anywhere in
#: the stack.
DENY_KEYS: Final[frozenset[str]] = frozenset(
    {
        # free text written by the user
        "narrative",
        "raw_text",
        "text",
        "excerpt",
        "message",
        "content",
        "question",
        "answer",
        "source_span",
        "notes",
        # money
        "income",
        "income_cents",
        "income_min_cents",
        "income_max_cents",
        "income_period",
        "monthly_income_cents",
        "annual_income_cents",
        # contact and location
        "address",
        "street",
        "postal_code",
        "zip",
        "zip_code",
        "email",
        "email_address",
        "phone",
        "phone_number",
        # identity
        "user_id",
        "user_uuid",
        "sub",
        "ssn",
        "social_security_number",
        "date_of_birth",
        "dob",
        "full_name",
        "name",
        # credentials
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "authorization",
        "api_key",
        "apikey",
        "jwt",
        "password",
        "secret",
        "service_role_key",
        "anon_key",
        "cookie",
        "set_cookie",
        "credentials",
        "dsn",
        "database_url",
        # aggregates that contain any of the above
        "answers",
        "clarification_answers",
        "profile",
        "extracted_profile",
        "user_profile",
        "prefilled",
        "household_composition",
        "state_update",
        "payload",
        "body",
        "request_body",
        "prompt",
        "contents",
        "history",
    }
)

#: Substrings that make a field name sensitive regardless of its exact spelling,
#: so ``gemini_api_key``, ``supabase_jwt_secret`` and ``user_narrative`` are all
#: caught without being enumerated.
DENY_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "email",
    "income",
    "jwt",
    "narrative",
    "passwd",
    "password",
    "phone",
    "secret",
    "ssn",
    "token",
)

#: Fields exempt from truncation. Tracebacks are long by nature and contain no
#: user data, and cutting them destroys the only signal they carry.
NO_TRUNCATE_KEYS: Final[frozenset[str]] = frozenset({"exception", "stack", "exc_info"})

#: Reserved keys the pipeline itself writes; never redacted or truncated.
_STRUCTURAL_KEYS: Final[frozenset[str]] = frozenset(
    {"event", "level", "logger", "timestamp", "request_id", "user_id_hash"}
)

_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # "Authorization: Bearer <jwt>" / "bearer abc.def"
    re.compile(r"\bbearer\s+[\w\-.~+/=]+", re.IGNORECASE),
    # any JWT-shaped string
    re.compile(r"\beyJ[\w-]{6,}\.[\w-]{4,}\.?[\w-]*"),
    # Google API keys, the shape GEMINI_API_KEY uses
    re.compile(r"\bAIza[\w\-]{10,}"),
    # postgres URL carrying a password
    re.compile(r"\bpostgres(?:ql)?(?:\+\w+)?://[^\s:/@]+:[^\s@]+@\S+", re.IGNORECASE),
    # email addresses
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    # money amounts, e.g. "$2,450" or "$1950.00"
    re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?"),
)


def _is_denied_key(key: str) -> bool:
    lowered = key.lower().lstrip("_")
    if lowered in _STRUCTURAL_KEYS:
        return False
    if lowered in DENY_KEYS:
        return True
    return any(fragment in lowered for fragment in DENY_KEY_FRAGMENTS)


def _scrub_text(value: str) -> str:
    for pattern in _VALUE_PATTERNS:
        value = pattern.sub(REDACTED, value)
    return value


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + TRUNCATION_SUFFIX


def _clean_value(value: Any, *, limit: int, truncate: bool, depth: int) -> Any:
    """Recursively redact, scrub and truncate one value."""
    if depth > _MAX_DEPTH:
        return REDACTED
    if isinstance(value, str):
        cleaned = _scrub_text(value)
        return _truncate(cleaned, limit) if truncate else cleaned
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if _is_denied_key(str(key))
                else _clean_value(item, limit=limit, truncate=truncate, depth=depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            _clean_value(item, limit=limit, truncate=truncate, depth=depth + 1) for item in value
        ]
    if isinstance(value, (bool, int, float, type(None))):
        return value
    # Anything else (enums, UUIDs, pydantic models, exceptions) is rendered as
    # text and then put through the same scrubbing as a string.
    return _clean_value(str(value), limit=limit, truncate=truncate, depth=depth + 1)


def redact_event_dict(event_dict: EventDict, *, max_text_chars: int) -> EventDict:
    """Return a redacted copy of one structlog event dictionary.

    Exposed separately from the processor so tests and ad-hoc tooling can assert
    on the redaction rules without configuring the global pipeline.
    """
    cleaned: EventDict = {}
    for key, value in event_dict.items():
        name = str(key)
        if _is_denied_key(name):
            cleaned[name] = REDACTED
            continue
        truncate = name.lower() not in NO_TRUNCATE_KEYS
        cleaned[name] = _clean_value(value, limit=max_text_chars, truncate=truncate, depth=0)
    return cleaned


def make_redaction_processor(max_text_chars: int) -> Processor:
    """Build the redaction processor bound to a truncation limit."""

    def redaction_processor(
        logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        return redact_event_dict(event_dict, max_text_chars=max_text_chars)

    return redaction_processor


def hash_user_id(user_id: str | uuid.UUID) -> str:
    """Return the stable, non-reversible log identity for a user.

    The same user always produces the same value, so runs can be correlated,
    while the raw id - which is also the Supabase ``sub`` claim and the row key
    for that user's data - never appears in a log line.
    """
    digest = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
    return f"{USER_HASH_PREFIX}{digest[:_USER_HASH_CHARS]}"


def new_request_id() -> str:
    """Generate a correlation id for one inbound request or workflow run."""
    return uuid.uuid4().hex


def _ensure_hashed(user_id_hash: str) -> str:
    """Accept an already-hashed identity, hash anything else.

    Defence in depth: a caller that passes a raw id by mistake gets it hashed
    instead of logged.
    """
    if user_id_hash.startswith(USER_HASH_PREFIX):
        return user_id_hash
    return hash_user_id(user_id_hash)


def bind_request_context(
    request_id: str,
    user_id_hash: str | None = None,
    **extra: str | int | float | bool | None,
) -> None:
    """Bind correlation context for the current task onto every later log line.

    ``user_id_hash`` must be the output of :func:`hash_user_id`. Raw ids are
    hashed before binding. ``extra`` is for low-cardinality, non-personal
    dimensions such as ``run_id`` or ``route``; it still goes through redaction.
    """
    context: dict[str, Any] = {"request_id": request_id}
    if user_id_hash is not None:
        context["user_id_hash"] = _ensure_hashed(user_id_hash)
    context.update(extra)
    structlog.contextvars.bind_contextvars(**context)


def clear_request_context() -> None:
    """Drop all bound correlation context for the current task."""
    structlog.contextvars.clear_contextvars()


def bound_request_context(
    request_id: str,
    user_id_hash: str | None = None,
    **extra: str | int | float | bool | None,
) -> Any:
    """Context-manager form of :func:`bind_request_context`.

    Returns ``structlog.contextvars.bound_contextvars``, which restores the
    previous context on exit - the safe choice inside LangGraph nodes and
    background jobs, where a task may already carry a parent context.
    """
    context: dict[str, Any] = {"request_id": request_id}
    if user_id_hash is not None:
        context["user_id_hash"] = _ensure_hashed(user_id_hash)
    context.update(extra)
    return structlog.contextvars.bound_contextvars(**context)


def configure_logging(settings: Settings, *, stream: TextIO | None = None) -> None:
    """Install the global JSON logging pipeline. Idempotent.

    Also routes the standard library's ``logging`` through the same stream so
    uvicorn, SQLAlchemy and Alembic output stays machine-readable. Those loggers
    are *not* redacted by structlog, so their levels are kept at WARNING and
    they must never be used for application events.

    ``stream`` defaults to ``sys.stdout``, which is what the application always
    wants - a container collects stdout. It is injectable so that tests can
    assert on the bytes the real pipeline produces without monkeypatching
    ``sys.stdout``, which is unreliable under a test runner that manages the
    stream itself.
    """
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Redaction runs last, after every other processor has contributed its
        # fields, so nothing can be added to the event after it is cleaned.
        make_redaction_processor(settings.log_max_text_chars),
        structlog.processors.JSONRenderer(),
    ]

    sink = stream if stream is not None else sys.stdout
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(file=sink),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sink,
        level=logging.WARNING,
        force=True,
    )
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> FilteringBoundLogger:
    """Return a bound logger for ``name``, conventionally ``__name__``."""
    logger: FilteringBoundLogger = structlog.get_logger(name)
    return logger
