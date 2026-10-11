"""Workflow run endpoints: start, poll, resume, cancel and re-evaluate.

The shape of this resource is the product's answer to a hard constraint: a run
takes tens of seconds, and no HTTP request may be held open for that long.
``POST /workflows`` therefore returns ``202`` with a durable ``run_id`` and a
``poll_url``; the client polls ``GET /workflows/{run_id}``, which is designed
to be useful at every point in the run rather than only at the end.

``POST /matches`` is the odd one out and deliberately so. It re-evaluates an
edited profile with the rule engine and makes **no model call at all**, so it
completes inside the request. It still answers with the same ``202`` run shape
as a full run, because the client's handling of a result set should not depend
on how expensive it was to produce.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Final

from fastapi import APIRouter, Path, Response, status

from app.api.errors import error_responses
from app.api.v1.deps import CurrentUser, DbSession, IdempotencyKey, Pagination, RequestId, RunnerDep
from app.schemas import (
    MatchRequest,
    Page,
    WorkflowCreateRequest,
    WorkflowCreateResponse,
    WorkflowResumeRequest,
    WorkflowRunDetail,
    WorkflowRunSummary,
)
from app.services import workflow_service

__all__ = ["router"]

router = APIRouter(tags=["workflows"])

_RUN_ID = Annotated[uuid.UUID, Path(description="The run id returned by POST /workflows.")]

#: ``Location`` points at the poll URL for the run, so a client that follows
#: headers never has to assemble the path itself.
_LOCATION_HEADER: Final[dict[str, Any]] = {
    "Location": {
        "description": "The poll URL for the run: `/api/v1/workflows/{run_id}`.",
        "schema": {"type": "string"},
    }
}


def _accepted_or_deduplicated(accepted_description: str) -> dict[int | str, dict[str, Any]]:
    """The two success statuses a run-creating endpoint can answer with.

    Both carry a :class:`~app.schemas.WorkflowCreateResponse`, but the status
    is the part a client branches on, and only one of them means work was
    started. FastAPI declares the ``status_code`` of the route and nothing
    else, so without this the generated TypeScript would type the replay as a
    204-shaped absence of a body - the generated client would be wrong about
    precisely the case idempotency exists to produce.
    """
    return {
        status.HTTP_202_ACCEPTED: {
            "model": WorkflowCreateResponse,
            "description": accepted_description,
            "headers": _LOCATION_HEADER,
        },
        status.HTTP_200_OK: {
            "model": WorkflowCreateResponse,
            "description": (
                "An `idempotency_key` that had already been used by this user: "
                "the original run, with `deduplicated: true`. Nothing was started."
            ),
            "headers": _LOCATION_HEADER,
        },
    }


@router.post(
    "/workflows",
    response_model=WorkflowCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a benefits check",
    description=(
        "Starts a full check in the background and returns immediately with a "
        "durable run id and the URL to poll. No request is ever held open for "
        "a model-backed workflow.\n\n"
        "**Idempotency.** Send `idempotency_key` in the body or an "
        "`Idempotency-Key` header. Replaying the same key returns the original "
        "run with `deduplicated: true` instead of starting a second one. "
        "Reusing a key with a different description is a 409 - returning the "
        "first run's results would answer a question the user did not ask."
    ),
    responses={
        **_accepted_or_deduplicated("The run was created and is executing in the background."),
        **error_responses(401, 409, 422, 429),
    },
)
async def create_workflow(
    body: WorkflowCreateRequest,
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    runner: RunnerDep,
    header_key: IdempotencyKey,
    response: Response,
) -> WorkflowCreateResponse:
    request = (
        body if body.idempotency_key else body.model_copy(update={"idempotency_key": header_key})
    )
    created = await workflow_service.create_run(
        session, user.user_id, request, request_id=request_id, runner=runner
    )
    if created.deduplicated:
        # The run already existed, so nothing was accepted for processing.
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = created.poll_url
    return created


@router.get(
    "/workflows",
    response_model=Page[WorkflowRunSummary],
    summary="List your checks",
    description="Your run history, newest first, with match counts and the latest plan id.",
    responses=error_responses(401, 422),
)
async def list_workflows(
    user: CurrentUser, session: DbSession, page: Pagination
) -> Page[WorkflowRunSummary]:
    return await workflow_service.list_runs(
        session, user.user_id, limit=page.limit, offset=page.offset
    )


@router.get(
    "/workflows/{run_id}",
    response_model=WorkflowRunDetail,
    summary="Poll one check",
    description=(
        "The full state of a run: status, phase, a plain-language progress "
        "message, a determinate progress fraction, and every result produced "
        "so far.\n\n"
        "Safe and useful at every point in the run. Matches appear as soon as "
        "the rule engine has produced them, before any wording exists; the "
        "plan appears when it is built; clarification questions appear only "
        "while the status is `AWAITING_INPUT`; and a failure arrives as a "
        "plain-language error with the partial results still attached.\n\n"
        "A run belonging to another user is a 404, not a 403, so this endpoint "
        "cannot be used to confirm that an id exists."
    ),
    responses=error_responses(401, 404),
)
async def get_workflow(user: CurrentUser, session: DbSession, run_id: _RUN_ID) -> WorkflowRunDetail:
    return await workflow_service.get_run(session, user.user_id, run_id)


@router.post(
    "/workflows/{run_id}/resume",
    response_model=WorkflowRunDetail,
    summary="Answer a paused check and continue it",
    description=(
        "Supplies answers to the questions a paused run asked and lets it "
        "continue from its checkpoint - possibly in a different process, and "
        "possibly days later, with nothing re-typed.\n\n"
        "An empty `answers` list is a real answer meaning 'continue with what "
        "you have'. The unanswered fields stay unknown and the programs they "
        "block stay at `MORE_INFO_NEEDED`, which is the honest outcome.\n\n"
        "A run that is not waiting for an answer is a 409."
    ),
    responses=error_responses(401, 404, 409, 422),
)
async def resume_workflow(
    body: WorkflowResumeRequest,
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    runner: RunnerDep,
    run_id: _RUN_ID,
) -> WorkflowRunDetail:
    return await workflow_service.resume_run(
        session, user.user_id, run_id, body, request_id=request_id, runner=runner
    )


@router.post(
    "/workflows/{run_id}/cancel",
    response_model=WorkflowRunDetail,
    summary="Cancel a check in progress",
    description=(
        "Stops a run the user no longer wants. Results already written are "
        "kept. A run that has already finished is a 409."
    ),
    responses=error_responses(401, 404, 409),
)
async def cancel_workflow(
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    runner: RunnerDep,
    run_id: _RUN_ID,
) -> WorkflowRunDetail:
    return await workflow_service.cancel_run(
        session, user.user_id, run_id, request_id=request_id, runner=runner
    )


@router.post(
    "/matches",
    response_model=WorkflowCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-check an edited profile",
    description=(
        "Applies an edit to the profile behind an earlier run and re-evaluates "
        "it. **Deterministic and synchronous**: the rule engine decides every "
        "status and no model is called, so the returned run is already "
        "complete and the first poll has results.\n\n"
        "Only the programs whose rules could actually read a changed field are "
        "re-evaluated. Every other match is carried forward with the wording "
        "the user has already read, so correcting household size neither "
        "re-bills twenty explanations nor changes twenty sentences underneath "
        "them."
    ),
    responses={
        **_accepted_or_deduplicated(
            "The re-evaluation ran synchronously; the run is already complete."
        ),
        **error_responses(401, 404, 409, 422),
    },
)
async def create_matches(
    body: MatchRequest,
    user: CurrentUser,
    session: DbSession,
    request_id: RequestId,
    header_key: IdempotencyKey,
    response: Response,
) -> WorkflowCreateResponse:
    request = (
        body if body.idempotency_key else body.model_copy(update={"idempotency_key": header_key})
    )
    created = await workflow_service.reevaluate(
        session, user.user_id, request, request_id=request_id
    )
    if created.deduplicated:
        response.status_code = status.HTTP_200_OK
    response.headers["Location"] = created.poll_url
    return created
