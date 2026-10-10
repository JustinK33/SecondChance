"""Intake: a narrative becomes a structured profile, and nothing else happens.

``POST /api/v1/intake`` is the cheap half of onboarding. It makes one model
call, assembles a profile, saves it as a new version and returns what we
understood together with the questions whose answers would change which
programs get suggested. No retrieval, no eligibility, no model-written prose -
so it answers in roughly one model call instead of in tens of seconds.

It exists because confirming a cheap misunderstanding is far better than
explaining an expensive one. A user who sees "we understood: household of 3 in
Middlesex County" can fix "3" before we spend a workflow run on it.

The extraction is deliberately conservative about what it keeps.
``ExtractedProfile.to_profile_fields`` drops anything below the confidence
floor, so a low-confidence guess never reaches the rule engine; the field stays
unknown and becomes a question instead.
"""

from __future__ import annotations

import uuid
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories import AuditAction, AuditRepository, ProfileRepository, hash_narrative
from app.schemas import (
    GeminiPort,
    IntakeRequest,
    IntakeResponse,
    UserProfile,
)
from app.services.ai_port import provide_ai_port
from app.services.profile_assembly import assemble_profile
from app.workflows.questions import intake_questions, material_missing_fields

__all__ = ["extract", "intake_pseudo_run_id"]

_log = get_logger(__name__)

#: Namespace for the pseudo run id that scopes intake question ids. Intake has
#: no run yet, but ``clarification_question_id`` needs a run-shaped scope and
#: the ids must be stable so a client can answer them in a later request. A
#: uuid5 over the narrative hash is stable, carries no readable content and
#: changes when the narrative does.
_INTAKE_NAMESPACE: Final[uuid.UUID] = uuid.UUID("8a1f1f44-5d1b-4c0e-9f33-6f2a1b0d7c55")


def intake_pseudo_run_id(narrative: str) -> uuid.UUID:
    """A stable, non-personal scope for one narrative's question ids."""
    return uuid.uuid5(_INTAKE_NAMESPACE, hash_narrative(narrative))


async def extract(
    session: AsyncSession,
    user_id: str,
    request: IntakeRequest,
    *,
    request_id: str,
    gemini: GeminiPort | None = None,
) -> IntakeResponse:
    """Extract a profile from ``request.narrative`` and store it.

    The only side effect is the saved profile version (plus its audit row). A
    model failure propagates: the API maps it to 502/503/504, and returning an
    empty profile as if we had read the narrative would be the one lie this
    product must never tell.
    """
    port = gemini if gemini is not None else provide_ai_port()
    result = await port.extract_profile(
        request.narrative, prefilled=request.prefilled, request_id=request_id
    )
    extracted = result.value

    # ``base=None`` on purpose. Intake answers "what did this narrative say",
    # and a stored profile outranks extraction in the precedence order, so
    # passing one would quietly show the user their old answers as if they had
    # just written them.
    profile, changed_fields = assemble_profile(extracted, prefilled=request.prefilled, answers=())

    questions = intake_questions(
        profile,
        material_missing_fields(profile),
        run_id=intake_pseudo_run_id(request.narrative),
    )

    repository = ProfileRepository(session)
    row = await repository.save_new_version(user_id, profile, changed_fields=changed_fields)
    await AuditRepository(session).record(
        user_id,
        AuditAction.PROFILE_SAVED,
        resource_type="profile",
        resource_id=row.id,
        request_id=request_id,
        metadata={
            "version": row.version,
            "changed_field_count": len(changed_fields),
            "unknown_field_count": len(profile.unknown_fields),
            "extraction_source": result.usage.source,
        },
    )
    await session.commit()

    # Counts and enum values only: the narrative and every extracted value are
    # personal content.
    _log.info(
        "intake.extracted",
        source=result.usage.source,
        profile_version=row.version,
        known_fields=len(_known_field_names(profile)),
        unknown_fields=len(extracted.unknown_fields),
        questions=len(questions),
        ambiguities=len(extracted.ambiguities),
    )

    return IntakeResponse(
        profile=profile,
        unknown_fields=list(extracted.unknown_fields),
        clarification_questions=questions[:5],
        ambiguities=list(extracted.ambiguities),
        extraction_source=result.usage.source,
    )


def _known_field_names(profile: UserProfile) -> list[str]:
    """Field names this profile actually determines. For a log count only."""
    return [
        name
        for name in ("state", "county", "household_size", "student_status", "income")
        if not profile.is_unknown(name)
    ]
