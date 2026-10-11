"""Liveness, readiness and the metrics scrape.

Health and readiness are separated on purpose, and the separation is load-
bearing rather than conventional.

``/healthz`` answers "is this process alive" and touches **nothing**. A
liveness probe that queries the database turns a thirty-second database blip
into a restart loop, which is strictly worse than the blip.

``/readyz`` answers "can this process serve traffic" and does check the
database and the configuration, so a load balancer can drain an instance that
cannot reach Postgres without killing it. This is the path Render's
``healthCheckPath`` points at.

Neither response carries a version string, a hostname, a connection string or a
dependency's error text. They are unauthenticated, and an unauthenticated
endpoint should not be a reconnaissance tool: ``ok: false`` plus a short
operator-facing note is all an operator needs and all an attacker gets.

``/internal/metrics`` renders the registry in Prometheus text format. It is
mounted outside ``/api/v1`` because it is not part of the product API, and it
carries route templates, status codes and counts - never a label derived from
anything a user wrote.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text

from app.api.v1.deps import DbSession, SettingsDep, provide_gemini
from app.core.logging import get_logger
from app.core.metrics import PROMETHEUS_CONTENT_TYPE, get_metrics
from app.db.session import get_engine
from app.schemas import DependencyStatus, GeminiPort, HealthResponse, ReadinessResponse, utcnow
from app.services.ai_port import DegradingGeminiPort, provide_ai_port
from app.services.workflow_runner import get_runner

__all__ = ["internal_router", "router"]

_log = get_logger(__name__)

router = APIRouter(tags=["system"])
internal_router = APIRouter(include_in_schema=False)

#: Readiness must answer quickly even when the database is wedged, so the probe
#: query gets a much tighter budget than an ordinary request.
_PROBE_TIMEOUT_S = 2.0

#: The model port, injected on the HTTP path and optional off it.
#:
#: ``app.main`` mounts ``/readyz`` as a root-path alias that *calls* :func:`ready`
#: directly rather than re-declaring the probe, so that one definition of
#: "ready" serves both. A direct call has no dependency injection, which is why
#: this parameter has a default: when it is absent, :func:`ready` resolves the
#: process-wide port itself. Injecting it on the versioned route keeps the
#: probe overridable in a test, exactly like every other port in the API.
ModelPortDep = Annotated[GeminiPort | None, Depends(provide_gemini)]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 as long as the process is running. Consults no dependency, "
        "so a database outage does not cause a restart loop."
    ),
)
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", app_env=settings.app_env)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Checks the database and the configuration. Returns 503 with "
        "`ready: false` when a required dependency is unavailable, so a load "
        "balancer can drain this instance instead of killing it."
    ),
    responses={503: {"model": ReadinessResponse, "description": "A dependency is unavailable."}},
)
async def ready(
    session: DbSession,
    settings: SettingsDep,
    response: Response,
    gemini: ModelPortDep = None,
) -> ReadinessResponse:
    port = gemini if gemini is not None else _process_model_port(settings)
    dependencies = [await _check_database(session), _check_model_access(settings, port)]
    required = [entry for entry in dependencies if entry.name != "gemini"]
    is_ready = all(entry.ok for entry in required)
    if not is_ready:
        response.status_code = 503
    return ReadinessResponse(ready=is_ready, dependencies=dependencies, checked_at=utcnow())


async def _check_database(session: DbSession) -> DependencyStatus:
    started = time.perf_counter()
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        # Type only. A driver message can carry the connection string, and the
        # connection string carries the password.
        _log.warning("readiness.database_unavailable", error_type=type(exc).__name__)
        return DependencyStatus(name="database", ok=False, note="not reachable")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    get_metrics().observe_db_query(
        operation="readiness_probe", duration_s=time.perf_counter() - started
    )
    return DependencyStatus(name="database", ok=True, latency_ms=elapsed_ms)


def _process_model_port(settings: SettingsDep) -> GeminiPort | None:
    """The process-wide model port, for callers that bypass injection.

    Building it is cheap and local - :func:`app.services.ai_port.provide_ai_port`
    caches, and constructing a client opens no connection - so this costs the
    probe nothing. ``None`` on failure: a model port that cannot even be
    constructed is reported as unavailable, which is true, rather than turning
    an unauthenticated readiness probe into a 500.
    """
    try:
        return provide_ai_port(settings)
    except Exception as exc:  # pragma: no cover - construction is local
        _log.warning("readiness.model_port_unavailable", error_type=type(exc).__name__)
        return None


def _check_model_access(settings: SettingsDep, port: GeminiPort | None) -> DependencyStatus:
    """What this process actually knows about model access.

    Deliberately not a required dependency. The whole product works without
    model access - the rule engine decides every status and retrieval is
    deterministic - so an unreachable model degrades the wording, not the
    service, and ``ready`` stays true either way.

    The check makes **no network call**. A readiness endpoint that ran a live
    ``generateContent`` would bill a round trip on every load-balancer poll and
    would time out precisely when the model is slow, which is when being
    drained helps least. But the previous version went to the other extreme and
    reported ``ok=True, note="configured"`` purely because a key *string* was
    present. With this project's key, ``generateContent`` fails for every
    model, so the probe advertised a dependency that was failing 100% of calls
    - the exact outage a readiness endpoint exists to surface.

    So the answer has three states rather than two, and the evidence for each
    is free:

    * **No key.** ``ok=False``. Evidence, from configuration.
    * **Key, and the port is latched off.**  ``ok=False, degraded=True``.
      :class:`~app.services.ai_port.DegradingGeminiPort` already records every
      live failure and the cool-off it entered, so this is a cheap read of
      evidence from a real attempt.
    * **Key, nothing attempted yet.** ``ok=True`` - no evidence either way -
      with a note that says so, so "configured" is never mistaken for
      "reachable".
    """
    if not settings.gemini_enabled:
        return DependencyStatus(
            name="gemini",
            ok=False,
            degraded=True,
            note="model access not provisioned; wording will be deterministic",
        )
    if port is None:
        return DependencyStatus(
            name="gemini",
            ok=False,
            degraded=True,
            note="model port unavailable; wording will be deterministic",
        )
    if not isinstance(port, DegradingGeminiPort):
        # An explicit override - a benchmark stub or a test double. It tracks
        # no failure state, so there is nothing to report beyond the fact that
        # this endpoint did not probe it.
        return DependencyStatus(name="gemini", ok=True, note="configured; not probed")
    if not port.has_live_client:
        return DependencyStatus(
            name="gemini",
            ok=False,
            degraded=True,
            failures=port.failures,
            note="no live model client; wording will be deterministic",
        )
    if port.is_degraded:
        return DependencyStatus(
            name="gemini",
            ok=False,
            degraded=True,
            failures=port.failures,
            note="live calls failing; deterministic wording until the cool-off expires",
        )
    if port.failures:
        return DependencyStatus(
            name="gemini",
            ok=True,
            degraded=False,
            failures=port.failures,
            note="cool-off expired after earlier failures; the next call re-probes",
        )
    return DependencyStatus(
        name="gemini",
        ok=True,
        degraded=False,
        failures=0,
        note="configured; no live call attempted yet",
    )


@internal_router.get("/internal/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    """The in-process registry in Prometheus text exposition format."""
    registry = get_metrics()
    _publish_pool_gauges()
    registry.set_workflows_in_flight(get_runner().snapshot().in_flight)
    return Response(content=registry.render_prometheus(), media_type=PROMETHEUS_CONTENT_TYPE)


@internal_router.get("/internal/metrics.json", summary="Metrics snapshot as JSON")
async def metrics_json() -> dict[str, object]:
    """The same numbers as a structured snapshot, for the benchmarks."""
    _publish_pool_gauges()
    snapshot = get_runner().snapshot()
    payload = get_metrics().snapshot()
    payload["runner"] = {
        "in_flight": snapshot.in_flight,
        "tracked": snapshot.tracked,
        "queued": snapshot.queued,
        "capacity": snapshot.capacity,
        "scheduled": snapshot.scheduled,
        "completed": snapshot.completed,
        "awaiting_input": snapshot.awaiting_input,
        "failed": snapshot.failed,
        "interrupted": snapshot.interrupted,
        "rejected": snapshot.rejected,
    }
    return payload


def _publish_pool_gauges() -> None:
    """Read connection-pool occupancy off the live engine, if there is one."""
    try:
        pool = get_engine().pool
    except Exception:  # pragma: no cover - no engine yet
        return
    size = getattr(pool, "size", None)
    checked_out = getattr(pool, "checkedout", None)
    overflow = getattr(pool, "overflow", None)
    available = getattr(pool, "checkedin", None)
    if not all(callable(fn) for fn in (size, checked_out, overflow, available)):
        return
    get_metrics().set_pool_gauges(
        size=int(size()),  # type: ignore[misc]
        checked_out=int(checked_out()),  # type: ignore[misc]
        # ``QueuePool.overflow()`` counts down from ``-pool_size`` until the
        # pool is full, so the raw value is negative most of the time. A gauge
        # called "overflow" has to mean overflow connections in use.
        overflow=max(0, int(overflow())),  # type: ignore[misc]
        available=int(available()),  # type: ignore[misc]
    )
