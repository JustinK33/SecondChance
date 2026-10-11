"""The ``/api/v1`` router, assembled from the route modules.

Declaration order is the order the endpoints appear in the OpenAPI document
and therefore in the generated client, so it follows the user's journey -
describe your situation, start a check, poll it, browse programs, read your
plan, ask a question, delete your data - rather than alphabetical order.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.errors import error_responses
from app.api.v1.routes import account, assistant, intake, plans, programs, system, workflows

__all__ = ["api_router", "internal_router"]

#: Declared on every route in the group. Without the explicit 422, FastAPI
#: advertises its own ``HTTPValidationError`` for any route with a parameter -
#: a body shape this API never returns, which would make the generated
#: TypeScript client wrong about the one case it most needs to be right about.
#: 413 and 500 are here because they are produced outside any one route -
#: :class:`~app.core.middleware.RequestBodySizeLimitMiddleware` can answer 413
#: before routing, and the catch-all handler can produce 500 anywhere - so
#: declaring them per route would mean declaring them on all of them.
_COMMON_RESPONSES = error_responses(413, 422, 500)

api_router = APIRouter(prefix="/api/v1", responses=_COMMON_RESPONSES)
api_router.include_router(intake.router)
api_router.include_router(workflows.router)
api_router.include_router(programs.router)
api_router.include_router(plans.router)
api_router.include_router(assistant.router)
api_router.include_router(account.router)
api_router.include_router(system.router)

#: Not part of the product API: mounted at the application root, excluded from
#: the schema, and intended for a metrics collector inside the deployment.
internal_router = system.internal_router
