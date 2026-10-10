"""``POST /api/v1/intake`` - read a narrative, return what we understood.

The cheap half of onboarding: one model call, no retrieval, no eligibility and
no model-written prose. It exists so a user can correct a misunderstanding
before a full run is spent on it.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.errors import error_responses
from app.api.v1.deps import CurrentUser, DbSession, GeminiDep, RequestId
from app.schemas import IntakeRequest, IntakeResponse
from app.services import intake_service

__all__ = ["router"]

router = APIRouter(tags=["intake"])


@router.post(
    "/intake",
    response_model=IntakeResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract a structured profile from a description",
    description=(
        "Reads a free-text description of someone's situation and returns the "
        "structured profile it determines, the fields it could not determine, "
        "and up to five questions whose answers would change which programs "
        "are suggested.\n\n"
        "Extraction only: no programs are retrieved and no eligibility is "
        "assessed, so this answers in roughly one model call. The only side "
        "effect is that the profile is saved as a new version.\n\n"
        "Nothing is guessed. A field the description does not determine is "
        "returned as unknown rather than inferred, and a low-confidence "
        "reading is discarded instead of being passed to the rule engine."
    ),
    responses=error_responses(401, 422, 429, 502, 503, 504),
)
async def create_intake(
    body: IntakeRequest,
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    gemini: GeminiDep,
) -> IntakeResponse:
    return await intake_service.extract(
        session, user.user_id, body, request_id=request_id, gemini=gemini
    )
