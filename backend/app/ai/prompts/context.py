"""Payload builders: contract objects reduced to what a prompt actually needs.

Three reasons this is a separate module rather than inline dumps:

* **Least data.** A prompt gets the fields the task needs and no others. The
  profile projection drops nothing the model needs but also sends nothing it
  does not, which keeps the request small and the blast radius of a prompt bug
  narrow.
* **Stable key names.** The model is told to copy ``source_id`` and
  ``program_slug`` values back verbatim, and the projection is what guarantees
  those keys exist and are spelled the same way in every prompt.
* **Honesty about verification.** Program payloads carry
  ``verification_status`` and ``supports_eligibility_claims`` so the prompt can
  tell the model which programs may be described in terms of requirements at
  all. An ``UNVERIFIED`` program is still shown - a user asking about one
  deserves an answer - but it is labelled.

Every payload produced here is placed inside a sanitised data block by the
prompt builders. None of it is trusted, including the parts this repository
wrote: a curated program description is still third-party prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.schemas.assistant import ChatMessage
from app.schemas.eligibility import EligibilityEvaluation
from app.schemas.matching import ProgramMatch
from app.schemas.profile import ProfileDraft, UserProfile
from app.schemas.program import AssistanceProgram, ProgramSource, RetrievedProgram

__all__ = [
    "candidates_payload",
    "catalog_urls",
    "evaluation_payload",
    "history_payload",
    "match_payload",
    "profile_payload",
    "program_payload",
    "source_payload",
]


def profile_payload(profile: UserProfile | ProfileDraft | None) -> dict[str, Any]:
    """What is known about the person, with unset fields omitted.

    ``exclude_none`` matters: a key with a ``null`` value reads to a model as
    "this was investigated and found to be nothing", whereas an absent key reads
    as "not known", which is what it means.
    """
    if profile is None:
        return {}
    payload = profile.model_dump(mode="json", exclude_none=True)
    composition = payload.get("household_composition")
    if isinstance(composition, dict):
        trimmed = {key: value for key, value in composition.items() if value is not None}
        if trimmed:
            payload["household_composition"] = trimmed
        else:
            payload.pop("household_composition")
    for key in ("needs", "flags", "unknown_fields", "extras"):
        if key in payload and not payload[key]:
            payload.pop(key)
    return payload


def source_payload(source: ProgramSource) -> dict[str, Any]:
    """One source document, including the id the model must cite by."""
    return {
        "source_id": source.id,
        "title": source.title,
        "publisher": source.publisher,
        "url": source.url,
        "retrieved_at": source.retrieved_at.date().isoformat(),
        "freshness": source.freshness.value,
        "excerpt": source.excerpt,
    }


def program_payload(
    program: AssistanceProgram,
    *,
    sources: Sequence[ProgramSource] | None = None,
    include_sources: bool = True,
) -> dict[str, Any]:
    """One program, as the model should see it."""
    payload: dict[str, Any] = {
        "program_slug": program.slug,
        "official_name": program.official_name,
        "administering_org": program.administering_org,
        "category": program.category.value,
        "plain_language_summary": program.plain_language_summary,
        "description": program.description,
        "who_it_helps": list(program.who_it_helps),
        "geo_scope": program.geo_scope.label,
        "official_url": program.official_url,
        "application_url": program.application_url,
        "phone": program.phone,
        "verification_status": program.verification_status.value,
        "may_be_described_with_requirements": program.supports_eligibility_claims(),
    }
    if include_sources:
        chosen = list(sources if sources is not None else program.sources)
        payload["sources"] = [source_payload(source) for source in chosen]
    return {key: value for key, value in payload.items() if value is not None}


def candidates_payload(candidates: Sequence[RetrievedProgram]) -> list[dict[str, Any]]:
    """Retrieved candidates, without their rule trees.

    The rule set is deliberately withheld: the model has no decision to make
    about it, and sending it would invite the model to evaluate it.
    """
    return [
        program_payload(candidate.program, sources=candidate.sources)
        for candidate in candidates
    ]


def evaluation_payload(evaluation: EligibilityEvaluation) -> dict[str, Any]:
    """The engine's decision and its reasoning, as an input to wording it."""
    return {
        "decided_status": evaluation.status.value,
        "root_outcome": evaluation.outcome.value,
        "rule_version": evaluation.rule_version,
        "missing_fields": list(evaluation.missing_fields),
        "limitations": list(evaluation.limitations),
        "evidence": [
            item.model_dump(mode="json", exclude_none=True) for item in evaluation.evidence
        ],
    }


def match_payload(match: ProgramMatch) -> dict[str, Any]:
    """One decided match, for action-plan drafting."""
    return {
        "program_slug": match.program.slug,
        "official_name": match.program.official_name,
        "administering_org": match.program.administering_org,
        "decided_status": match.status.value,
        "plain_language_summary": match.program.plain_language_summary,
        "official_url": match.program.official_url,
        "application_url": match.program.application_url,
        "missing_fields": list(match.evaluation.missing_fields),
        "limitations": list(match.evaluation.limitations),
        "requirements_evidence": [
            {
                "description": item.description,
                "outcome": item.outcome.value,
                "missing_fields": list(item.missing_fields),
            }
            for item in match.evaluation.evidence
        ],
    }


def catalog_urls(matches: Sequence[ProgramMatch]) -> list[str]:
    """Every URL an action item is allowed to carry, from the catalog only."""
    urls: list[str] = []
    for match in matches:
        for url in (match.program.official_url, match.program.application_url):
            if url and url not in urls:
                urls.append(url)
    return urls


def history_payload(history: Sequence[ChatMessage], *, limit: int = 6) -> list[dict[str, str]]:
    """Recent conversation turns, oldest first, capped.

    Client-supplied and therefore untrusted, which is why it is fenced like any
    other data block rather than replayed as real conversation turns.
    """
    recent = list(history)[-limit:]
    return [{"role": message.role, "content": message.content} for message in recent]
