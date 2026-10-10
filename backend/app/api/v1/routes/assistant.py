"""``POST /api/v1/assistant/chat`` - source-grounded follow-up questions.

Every sentence returned as fact carries at least one citation into a stored
official source, and every citation is verified deterministically before the
answer is sent: the id must name a real source, the quoted excerpt must appear
in it, the URL must match and the source must not be stale.

Anything the verified sources do not support is stripped sentence by sentence.
If nothing survives, the answer says plainly that we could not confirm it in an
official source rather than asserting something helpful-sounding.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.errors import error_responses
from app.api.v1.deps import CurrentUser, DbSession, GeminiDep, RequestId
from app.schemas import ChatRequest, ChatResponse
from app.services import assistant_service

__all__ = ["router"]

router = APIRouter(tags=["assistant"])


@router.post(
    "/assistant/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about a program",
    description=(
        "Answers a question from official sources only.\n\n"
        "The response separates `facts` - each traceable to a cited source - "
        "from `interpretation`, which is the assistant's reading and is "
        "labelled as such. `is_grounded` is computed, not supplied: it is true "
        "only when at least one citation survived verification, and when it is "
        "false `ungrounded_notice` is populated so the interface has no way to "
        "render an unsourced answer as if it were official.\n\n"
        "`run_id` and `program_id` only scope which sources are searched. "
        "Neither is trusted as authorisation, and a run id the caller does not "
        "own is ignored rather than rejected."
    ),
    responses=error_responses(401, 422, 429, 502, 503, 504),
)
async def chat(
    body: ChatRequest,
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    gemini: GeminiDep,
) -> ChatResponse:
    return await assistant_service.answer(
        session, user.user_id, body, request_id=request_id, gemini=gemini
    )
