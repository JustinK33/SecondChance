"""The ASGI application: factory, lifespan and middleware stack.

Lifespan, in order
------------------
**Startup**

1. ``configure_logging`` - installed first, so every later line is redacted.
2. ``assert_auth_mode_allowed`` - fails the boot rather than serving a
   deployment with authentication disabled.
3. The engine and session factory, one per process, because the engine owns the
   connection pool.
4. The checkpointer's ``setup()``, which creates the LangGraph checkpoint
   tables. Alembic deliberately does not manage them.
5. Reconciliation of runs left ``RUNNING`` by a process that died, rewritten to
   ``INTERRUPTED`` so they stop being polled forever and can be resumed.

**Shutdown**

1. The runner stops accepting work, records its in-flight runs as
   ``INTERRUPTED`` and then cancels and awaits them.
2. The checkpointer's connection pool closes.
3. The engine's pool closes.

Startup failures are not swallowed. A process that cannot reach its database or
cannot create its checkpoint tables must not report itself ready; it should
crash and let the platform restart or roll back.

Middleware, outermost first: CORS, request context (id, logging, access log,
metrics), request timeout, router. The reasoning for that order is in
:mod:`app.core.middleware`.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Final

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware

from app.api.errors import install_exception_handlers
from app.api.v1.deps import DbSession
from app.api.v1.router import api_router, internal_router
from app.api.v1.routes.system import ready
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import (
    DEFAULT_REQUEST_TIMEOUT_S,
    RequestContextMiddleware,
    RequestTimeoutMiddleware,
    cors_kwargs,
)
from app.core.security import assert_auth_mode_allowed
from app.db.session import dispose_engine, get_engine, get_session_factory
from app.schemas import HealthResponse, ReadinessResponse
from app.services.workflow_runner import WorkflowRunner, get_runner, set_runner
from app.workflows.checkpointer import close_checkpointer, get_checkpointer

__all__ = ["app", "create_app", "lifespan"]

_log = get_logger(__name__)

TITLE: Final[str] = "SecondChance API"
VERSION: Final[str] = "1.0.0"

DESCRIPTION: Final[str] = """
An AI benefits navigator for New Jersey and Rutgers.

Someone describes their situation in their own words. The API extracts a
structured profile, retrieves curated real assistance programs, runs a
**deterministic** eligibility engine over them, and returns plain-language
explanations with an action plan of official links.

### Rules this API guarantees

* **The rule engine decides, the model only words.** A match status is produced
  by deterministic rules over the profile and an approved rule set. No model
  output can create, change or influence a status.
* **Three statuses, and none of them says "eligible".** `POTENTIAL_MATCH`,
  `MORE_INFO_NEEDED`, `UNLIKELY_MATCH`. Only the program that runs it can
  decide eligibility, so this API never claims to.
* **Missing information never disqualifies anyone.** An unresolvable
  requirement produces `MORE_INFO_NEEDED` and names the field it was missing.
* **Nothing is fabricated.** Every program, requirement and citation traces to
  a stored official source with the date it was last checked. Unverified data
  is marked as such and excluded from eligibility claims.
* **We never ask for an SSN, banking credentials, or anything a program does
  not need.** There is nowhere in the schema to put one.
* **Personal content is never logged.** Narratives, income, emails and raw user
  ids do not appear in logs, metrics labels or audit rows.

### Conventions

* Every non-2xx response has the same body: `{"error": {code, message, detail,
  request_id}}`. `code` is stable and safe to branch on; `message` is plain
  language and safe to show a user verbatim.
* Long work is never held open on a request. `POST /api/v1/workflows` returns
  `202` with a run id; poll `GET /api/v1/workflows/{run_id}`.
* A resource belonging to another user returns `404`, not `403`.
* Identity comes from the bearer token. A user id in a body or query string is
  ignored.
""".strip()

TAGS_METADATA: Final[list[dict[str, str]]] = [
    {
        "name": "intake",
        "description": "Turn a free-text description into a structured profile.",
    },
    {
        "name": "workflows",
        "description": "Start, poll, resume, cancel and re-run a benefits check.",
    },
    {
        "name": "programs",
        "description": "The public catalog of curated assistance programs and their sources.",
    },
    {"name": "plans", "description": "Action plans and the steps inside them."},
    {"name": "assistant", "description": "Source-grounded answers to follow-up questions."},
    {"name": "account", "description": "Delete everything we hold about you."},
    {"name": "system", "description": "Liveness and readiness probes."},
]


def _build_app(settings: Settings) -> FastAPI:
    middleware = [
        Middleware(CORSMiddleware, **cors_kwargs(settings)),
        Middleware(RequestContextMiddleware),
        Middleware(RequestTimeoutMiddleware, timeout_s=DEFAULT_REQUEST_TIMEOUT_S),
    ]
    application = FastAPI(
        title=TITLE,
        version=VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        summary="Deterministic benefits matching with plain-language explanations.",
        lifespan=lifespan,
        middleware=middleware,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json",
        contact={"name": "SecondChance", "url": settings.frontend_url},
    )
    application.state.settings = settings
    install_exception_handlers(application)
    application.include_router(api_router)
    application.include_router(internal_router)

    @application.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["system"],
        summary="Liveness probe (root path)",
        description=(
            "Alias of `GET /api/v1/health`, at the root path platforms probe by "
            "default. Consults no dependency."
        ),
    )
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok", app_env=settings.app_env)

    @application.get(
        "/readyz",
        response_model=ReadinessResponse,
        tags=["system"],
        summary="Readiness probe (root path)",
        description=(
            "Alias of `GET /api/v1/ready`, at the root path platforms probe by "
            "default. Checks the database and the configuration, and answers "
            "503 with `ready: false` when a required dependency is down."
        ),
        responses={
            503: {"model": ReadinessResponse, "description": "A dependency is unavailable."}
        },
    )
    async def readyz(session: DbSession, response: Response) -> ReadinessResponse:
        # Delegates rather than duplicating the probe: one definition of what
        # "ready" means, so the root alias and the versioned route can never
        # disagree about whether this instance should take traffic.
        return await ready(session, settings, response)

    return application


@contextlib.asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Bring the process up, and take it down without stranding a run."""
    settings: Settings = getattr(application.state, "settings", None) or get_settings()
    configure_logging(settings)
    assert_auth_mode_allowed(settings)

    get_engine(settings)
    get_session_factory(settings)
    handle = await get_checkpointer(settings)

    runner = WorkflowRunner(settings)
    set_runner(runner)
    await runner.startup()

    _log.info(
        "app.started",
        app_env=settings.app_env,
        auth_mode=settings.auth_mode,
        checkpointer=handle.kind,
        checkpointer_persistent=handle.is_persistent,
        model_access=settings.gemini_enabled,
        max_concurrent_workflows=settings.max_concurrent_workflows,
        workflow_timeout_s=settings.workflow_timeout_s,
    )
    try:
        yield
    finally:
        await get_runner().shutdown()
        await close_checkpointer()
        await dispose_engine()
        _log.info("app.stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level object so a test can construct one
    with different settings, and so ``uvicorn --factory`` works. The
    module-level :data:`app` exists because that is what ``make dev-api`` and
    the deployment command point at.
    """
    return _build_app(settings or get_settings())


app = create_app()
