"""Turning a model response body into a validated object.

Shared by the live client and by :class:`~app.ai.fake.FakeGeminiClient`, which
is the point: a fixture replayed in CI goes through exactly the same parsing,
validation and failure path as a live response, so "it works against the fake"
means something.

Why the pydantic error message is never logged
----------------------------------------------
``ValidationError`` renders the offending *input* into its message
(``Input should be a valid integer [input_value='about two thousand']``). Model
output is derived from a user narrative, so that message can contain personal
content. Only the error ``loc`` and ``type`` are ever logged or attached to an
exception; the message itself is dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from app.ai.errors import ModelInvalidOutput

__all__ = ["decode_structured", "validation_error_summary"]

#: ``response_mime_type="application/json"`` makes a fenced block unlikely, but
#: a model that ignores it once should cost one strip, not one failed call.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)
_MAX_SUMMARY_ERRORS: Final[int] = 5


def validation_error_summary(error: ValidationError) -> list[str]:
    """Field paths and error types only - never the offending values."""
    summary: list[str] = []
    for item in error.errors()[:_MAX_SUMMARY_ERRORS]:
        location = ".".join(str(part) for part in item.get("loc", ()))
        summary.append(f"{location or '<root>'}:{item.get('type', 'unknown')}")
    return summary


def decode_structured[M: BaseModel](
    schema: type[M],
    *,
    text: str | None,
    parsed: Any = None,
    task: str | None = None,
    model: str | None = None,
) -> tuple[M, list[str]]:
    """Validate a structured response body against ``schema``.

    Prefers ``response.parsed`` - the SDK has already validated it against the
    same pydantic model - and falls back to parsing ``response.text`` itself,
    because ``parsed`` is ``None`` whenever the SDK could not do that.

    Returns the validated model and a list of non-fatal notes. Raises
    :class:`~app.ai.errors.ModelInvalidOutput` - never retried by the caller -
    for an empty body, unparseable JSON, or a schema violation.
    """
    notes: list[str] = []
    if isinstance(parsed, schema):
        return parsed, notes

    if parsed is not None:
        notes.append("parsed_wrong_type")

    if text is None or not text.strip():
        raise ModelInvalidOutput(task=task, model=model, reason="empty_response")

    body = text
    if (fenced := _FENCE.match(body)) is not None:
        notes.append("stripped_code_fence")
        body = fenced.group("body")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ModelInvalidOutput(
            task=task, model=model, reason="malformed_json", cause=exc
        ) from exc

    if not isinstance(payload, dict):
        raise ModelInvalidOutput(task=task, model=model, reason="not_a_json_object")

    try:
        value = schema.model_validate(payload)
    except ValidationError as exc:
        raise ModelInvalidOutput(
            task=task,
            model=model,
            reason="schema_violation",
            validation_summary=validation_error_summary(exc),
            cause=exc,
        ) from exc
    return value, notes
