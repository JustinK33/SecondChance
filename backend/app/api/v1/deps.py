"""Shared FastAPI dependencies for the v1 API.

Everything a route needs that is not its request body: the verified principal,
a database session, pagination with caps, the idempotency key, and the injected
ports. They live here so that a route function reads as a statement of intent
and so a test can override any one of them in isolation.

The rule this module exists to enforce is that **identity comes from the token
and from nowhere else**. :data:`CurrentUser` is the only source of a
``user_id``; a ``user_id`` in a path, a query string or a body is ignored, and
every service takes the authenticated id as a separate argument so that a route
cannot accidentally pass the client's version of it.

Pagination is capped at :data:`MAX_PAGE_LIMIT`. An uncapped ``limit`` is a
denial-of-service primitive handed to anyone with a browser, and the cap is the
same number ``Page`` validates against, so the two cannot drift.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Final

from fastapi import Depends, Header, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedError
from app.core.logging import bind_request_context, new_request_id
from app.core.security import DEV_USER_HEADER, AuthenticatedUser, dev_user_from_header, verify_token
from app.db.repositories import SqlProgramRepository
from app.db.session import get_session
from app.eligibility.engine import RuleEngine
from app.retrieval.citations import SourceCitationVerifier
from app.retrieval.search import HybridRetriever
from app.schemas import CitationVerifier, GeminiPort, ProgramRepository, Retriever
from app.services.ai_port import provide_ai_port
from app.services.workflow_runner import WorkflowRunner, get_runner

__all__ = [
    "MAX_PAGE_LIMIT",
    "CurrentUser",
    "DbSession",
    "EngineDep",
    "GeminiDep",
    "IdempotencyKey",
    "PageParams",
    "Pagination",
    "ProgramRepo",
    "RequestId",
    "RetrieverDep",
    "RunnerDep",
    "SettingsDep",
    "VerifierDep",
    "current_user",
]

#: Matches ``Page.limit``'s own upper bound. One number, one place.
MAX_PAGE_LIMIT: Final[int] = 200
DEFAULT_PAGE_LIMIT: Final[int] = 20

#: ``auto_error=False`` so a missing header reaches our own handler and comes
#: back in the standard envelope rather than as FastAPI's ``{"detail": ...}``.
_bearer = HTTPBearer(auto_error=False, description="Supabase access token.")


# ---------------------------------------------------------------------------
# request scope
# ---------------------------------------------------------------------------
def get_request_id(request: Request) -> str:
    """The correlation id for this request.

    Normally set by :class:`~app.core.middleware.RequestContextMiddleware`. The
    fallback covers a route exercised without the middleware - a unit test
    using ``TestClient`` on a bare router - so no code path has to cope with it
    being absent.
    """
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value:
        return value
    generated = new_request_id()
    request.state.request_id = generated
    return generated


def get_app_settings() -> Settings:
    """The process-wide settings. Overridable in tests."""
    return get_settings()


RequestId = Annotated[str, Depends(get_request_id)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------
async def current_user(
    request: Request,
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    dev_user: Annotated[
        str | None,
        Header(
            alias=DEV_USER_HEADER,
            description=(
                "Local development only: the UUID to act as. Accepted only when "
                "AUTH_MODE=dev and APP_ENV=local."
            ),
        ),
    ] = None,
) -> AuthenticatedUser:
    """The verified principal for this request.

    In ``supabase`` mode this is the ``sub`` claim of a token whose signature,
    expiry, issuer and audience have all been checked. In ``dev`` mode it is
    the ``X-Dev-User`` header, which :func:`dev_user_from_header` refuses
    outside ``APP_ENV=local``.
    """
    if settings.auth_mode == "dev":
        user = dev_user_from_header(dev_user, settings)
    else:
        if credentials is None or not credentials.credentials:
            raise UnauthorizedError()
        user = await verify_token(credentials.credentials, settings)

    # Bind the hashed id so every log line for the rest of the request is
    # attributable without ever carrying the raw id.
    bind_request_context(get_request_id(request), user.log_identity)
    request.state.user_id = user.user_id
    return user


CurrentUser = Annotated[AuthenticatedUser, Depends(current_user)]


# ---------------------------------------------------------------------------
# pagination and headers
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PageParams:
    """A validated, capped pagination window."""

    limit: int
    offset: int


def pagination(
    limit: Annotated[
        int,
        Query(ge=1, le=MAX_PAGE_LIMIT, description="Rows per page."),
    ] = DEFAULT_PAGE_LIMIT,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
) -> PageParams:
    """Pagination with sane caps, shared by every list endpoint."""
    return PageParams(limit=limit, offset=offset)


Pagination = Annotated[PageParams, Depends(pagination)]


def idempotency_key(
    key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            min_length=8,
            max_length=64,
            description=(
                "Replaying a request with the same key returns the original "
                "result instead of creating a second one."
            ),
        ),
    ] = None,
) -> str | None:
    """The ``Idempotency-Key`` header, if the client sent one."""
    return key.strip() if key else None


IdempotencyKey = Annotated[str | None, Depends(idempotency_key)]


# ---------------------------------------------------------------------------
# injected ports
# ---------------------------------------------------------------------------
def provide_program_repository(session: DbSession) -> ProgramRepository:
    """The catalog repository, bound to this request's session."""
    return SqlProgramRepository(session)


def provide_retriever(session: DbSession) -> Retriever:
    """Hybrid retrieval over the catalog, bound to this request's session."""
    return HybridRetriever(SqlProgramRepository(session))


def provide_verifier(session: DbSession) -> CitationVerifier:
    """Deterministic citation verification against stored sources."""
    return SourceCitationVerifier(SqlProgramRepository(session), session=session)


def provide_engine() -> RuleEngine:
    """The deterministic rule engine. Stateless, so one instance is shared."""
    return _ENGINE


def provide_gemini(settings: SettingsDep) -> GeminiPort:
    """The active model port.

    Live when a key is configured; the fixture-replaying implementation on a
    developer machine without one. See :mod:`app.services.ai_port` for why that
    second path exists and how it is kept from passing itself off as live
    output.
    """
    return provide_ai_port(settings)


def provide_runner() -> WorkflowRunner:
    """The background workflow runner for this process."""
    return get_runner()


_ENGINE: Final[RuleEngine] = RuleEngine()

ProgramRepo = Annotated[ProgramRepository, Depends(provide_program_repository)]
RetrieverDep = Annotated[Retriever, Depends(provide_retriever)]
VerifierDep = Annotated[CitationVerifier, Depends(provide_verifier)]
EngineDep = Annotated[RuleEngine, Depends(provide_engine)]
GeminiDep = Annotated[GeminiPort, Depends(provide_gemini)]
RunnerDep = Annotated[WorkflowRunner, Depends(provide_runner)]


def program_key(value: str) -> uuid.UUID | str:
    """Interpret a program path parameter as an id or a slug.

    ``GET /api/v1/programs/{program_id}`` accepts both, because a slug is what
    a shareable link should contain - ``/programs/nj-snap`` is a URL a person
    can read out loud - while the id is what every internal reference uses.
    """
    try:
        return uuid.UUID(value)
    except ValueError:
        return value
