"""Primitive types and helpers shared by every schema module.

Conventions fixed here and relied on everywhere else:

* **Money is integer cents.** Floats are never used for money. ``Cents`` is a
  non-negative ``int``.
* **URLs are validated ``str``.** Validation happens on the way in, but the
  value stays a plain string so it round-trips identically through JSON,
  Postgres and the generated TypeScript types.
* **Datetimes are timezone-aware UTC.** Naive values are rejected rather than
  guessed at, and aware values in another offset are converted.
* **Entity ids are UUIDs, derived deterministically where possible.** A
  program's id is ``uuid5(PROGRAM_NAMESPACE, slug)``, so the seed script, the
  test fixtures and the database all agree without a lookup table.
"""

from __future__ import annotations

import unicodedata
import uuid
from datetime import UTC, datetime
from typing import Annotated, Final
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

__all__ = [
    "PROGRAM_NAMESPACE",
    "Cents",
    "ConfidenceScore",
    "CountyName",
    "HttpUrlStr",
    "NonEmptyStr",
    "Page",
    "PlainText",
    "Slug",
    "SourceId",
    "StateCode",
    "UtcDatetime",
    "action_item_uuid",
    "clarification_question_id",
    "program_uuid",
    "slugify",
    "utcnow",
]

#: Fixed UUIDv5 namespace for SecondChance program slugs. Never change it: the
#: value is baked into seeded rows, fixtures and saved action plans.
PROGRAM_NAMESPACE: Final[uuid.UUID] = uuid.UUID("6f1b9e1c-2a4d-5b8e-9c3f-7d0a1b2c3d4e")

_ALLOWED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})
_MAX_URL_LENGTH: Final[int] = 2048


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        msg = (
            "naive datetime rejected: supply a timezone-aware value (use app.schemas.common.utcnow)"
        )
        raise ValueError(msg)
    return value.astimezone(UTC)


def _validate_http_url(value: str) -> str:
    """Accept only absolute ``http``/``https`` URLs.

    Also a security control: official source and application URLs are fetched
    by the ingestion layer, so rejecting ``file:``, ``gopher:`` and
    scheme-relative values here closes the easiest SSRF door.
    """
    candidate = value.strip()
    if len(candidate) > _MAX_URL_LENGTH:
        msg = f"URL longer than {_MAX_URL_LENGTH} characters"
        raise ValueError(msg)
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        msg = f"URL scheme must be http or https, got {parts.scheme!r}"
        raise ValueError(msg)
    if not parts.netloc:
        msg = "URL is missing a host"
        raise ValueError(msg)
    return candidate


def _validate_state_code(value: str) -> str:
    """Require a two-letter USPS state code.

    Checked with an ``AfterValidator`` rather than a ``pattern`` constraint
    because pydantic applies ``pattern`` before ``to_upper``, which would make
    the perfectly reasonable input ``"nj"`` a validation error.
    """
    if len(value) != 2 or not value.isalpha():
        msg = f"expected a two-letter state code, got {value!r}"
        raise ValueError(msg)
    return value


#: US state or territory as a two-letter USPS code, upper-cased on input.
StateCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_upper=True),
    AfterValidator(_validate_state_code),
]

#: County name as the state prints it, for example ``Middlesex``. A trailing
#: "County" is stripped so "Middlesex" and "Middlesex County" compare equal.
CountyName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=80),
    AfterValidator(lambda value: value.removesuffix(" County").removesuffix(" county").strip()),
]

#: Absolute http(s) URL, carried as a plain ``str``.
HttpUrlStr = Annotated[str, AfterValidator(_validate_http_url)]

#: Timezone-aware UTC datetime. Naive input is an error.
UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]

#: Non-negative integer cents. Money is never a float in this codebase.
Cents = Annotated[int, Field(ge=0)]

#: Model or retrieval confidence in ``[0, 1]``.
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]

#: A trimmed, non-empty single-line string.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

#: User- or editor-authored prose. Bounded so one field cannot hold a document.
PlainText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]

#: Lowercase kebab-case identifier, stable across releases.
Slug = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=120,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]

#: Human-authored id of an official source document, for example
#: ``nj-snap-income-limits-2026``. Not a UUID: these are written by hand in the
#: curated catalog and must stay readable in citations.
SourceId = Slug


def slugify(value: str) -> str:
    """Normalise arbitrary text into a :data:`Slug`.

    Deterministic by design - it is part of the action-item deduplication key,
    so the same requirement phrased by two different programs must collapse to
    the same slug. Accents are folded, everything that is not ``[a-z0-9]``
    becomes a separator, and runs of separators collapse.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    pieces = [
        piece
        for piece in "".join(char if char.isalnum() else " " for char in ascii_only.lower()).split()
        if piece
    ]
    if not pieces:
        msg = f"cannot slugify {value!r}: no alphanumeric characters"
        raise ValueError(msg)
    return "-".join(pieces)[:120].rstrip("-")


def program_uuid(slug: str) -> uuid.UUID:
    """Deterministic program id for a catalog slug."""
    return uuid.uuid5(PROGRAM_NAMESPACE, slug)


def action_item_uuid(run_id: uuid.UUID, dedup_key: str) -> uuid.UUID:
    """Deterministic action-item id.

    Derived from the run and the deduplication key so that regenerating a plan
    for the same run produces stable ids. That is what lets the frontend keep
    completion state across a plan regeneration, and what makes plan creation
    idempotent under a retried request.
    """
    return uuid.uuid5(PROGRAM_NAMESPACE, f"action:{run_id}:{dedup_key}")


def clarification_question_id(run_id: uuid.UUID, field: str) -> str:
    """Deterministic clarification-question id.

    One question per profile field per run, so a resubmitted answer always maps
    back to the same question and cannot create a duplicate.
    """
    return f"q_{uuid.uuid5(PROGRAM_NAMESPACE, f'question:{run_id}:{field}').hex[:12]}"


class Page[T](BaseModel):
    """Offset-paginated collection envelope used by every list endpoint."""

    model_config = ConfigDict(extra="forbid")

    items: list[T] = Field(default_factory=list)
    total: int = Field(ge=0, description="Total rows matching the filter, ignoring the window.")
    limit: int = Field(ge=1, le=200)
    offset: int = Field(ge=0)

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total
