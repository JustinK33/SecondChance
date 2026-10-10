"""Program catalog endpoints - the only unauthenticated data routes.

The catalog is public information: these are real assistance programs with
official pages anybody can read. Keeping the list and the detail view
unauthenticated means the product is useful before a user signs in, and it
means a link to a program is a link, not a login wall.

They return catalog data only. Nothing here is scoped to a user, nothing here
reads a profile, and no eligibility is assessed - a status is a statement about
one person's situation and cannot be made without one.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Path, Query

from app.api.errors import error_responses
from app.api.v1.deps import Pagination, ProgramRepo
from app.core.errors import NotFoundError
from app.schemas import ProgramCategory, ProgramDetail, ProgramPage, VerificationStatus

__all__ = ["router"]

router = APIRouter(tags=["programs"])


@router.get(
    "/programs",
    response_model=ProgramPage,
    summary="List assistance programs",
    description=(
        "Paginated catalog of curated assistance programs, filterable by "
        "category, state, county, verification status and full-text query.\n\n"
        "A program scoped to a whole state has no counties of its own and still "
        "matches a county filter, so filtering by county cannot hide SNAP.\n\n"
        "Public: no authentication required."
    ),
    responses=error_responses(422),
)
async def list_programs(
    repository: ProgramRepo,
    page: Pagination,
    category: Annotated[
        list[ProgramCategory] | None,
        Query(description="Repeatable. Any program in one of these categories."),
    ] = None,
    state: Annotated[
        str | None, Query(min_length=2, max_length=2, description="Two-letter state code.")
    ] = None,
    county: Annotated[
        str | None, Query(min_length=2, max_length=80, description="County name.")
    ] = None,
    q: Annotated[
        str | None, Query(min_length=2, max_length=120, description="Full-text search.")
    ] = None,
    verification_status: Annotated[
        list[VerificationStatus] | None,
        Query(description="Repeatable. Defaults to every status."),
    ] = None,
) -> ProgramPage:
    return await repository.list_programs(
        categories=category,
        state=state,
        county=county,
        query=q,
        verification_statuses=verification_status,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/programs/{program_id}",
    response_model=ProgramDetail,
    summary="One program, with sources and plain-language requirements",
    description=(
        "Accepts either the program id or its slug, so a shareable link can "
        "read `/api/v1/programs/nj-snap`.\n\n"
        "Includes the official sources behind the program with the date each "
        "was last checked, and the requirements rendered as plain-language "
        "bullets written by a human author - never generated text.\n\n"
        "Public: no authentication required."
    ),
    responses=error_responses(404),
)
async def get_program(
    repository: ProgramRepo,
    program_id: Annotated[
        str, Path(description="The program's UUID or its slug.", min_length=1, max_length=120)
    ],
) -> ProgramDetail:
    detail: ProgramDetail | None
    try:
        parsed = uuid.UUID(program_id)
    except ValueError:
        detail = await repository.get_program_by_slug(program_id)
    else:
        detail = await repository.get_program(parsed)
    if detail is None:
        raise NotFoundError("We could not find that program.")
    return detail
