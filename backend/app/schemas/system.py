"""Operational endpoint schemas: health, readiness and data deletion.

Health and readiness are separated deliberately. ``/healthz`` answers "is this
process alive" and must never touch a dependency, or a database blip triggers a
restart loop. ``/readyz`` answers "can this process serve traffic" and does
check dependencies, so a load balancer can drain an instance that cannot reach
Postgres without killing it.

Neither response includes a version string, a hostname, a connection string or
a dependency error message: an unauthenticated endpoint should not be a
reconnaissance tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import NonEmptyStr, UtcDatetime

__all__ = [
    "AccountDeletionReceipt",
    "DependencyStatus",
    "HealthResponse",
    "ReadinessResponse",
]


class HealthResponse(BaseModel):
    """Liveness. Static by design - no dependency is consulted."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    app_env: NonEmptyStr


class DependencyStatus(BaseModel):
    """One dependency's reachability. Carries no error detail."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["database", "checkpointer", "gemini"]
    ok: bool
    latency_ms: int | None = Field(default=None, ge=0)
    note: str | None = Field(
        default=None,
        max_length=120,
        description=(
            'Short operator-facing note, e.g. "model access not provisioned". '
            "Never an upstream exception message."
        ),
    )


class ReadinessResponse(BaseModel):
    """Readiness. ``ready`` is false if any required dependency is down."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    dependencies: list[DependencyStatus] = Field(default_factory=list)
    checked_at: UtcDatetime


class AccountDeletionReceipt(BaseModel):
    """Result of ``DELETE /api/v1/account/data``.

    A privacy promise has to be verifiable, so the user gets counts of what was
    actually removed. ``retained`` names what deliberately survives and why -
    currently only aggregate, non-identifying telemetry rows whose user
    reference has been nulled.
    """

    model_config = ConfigDict(extra="forbid")

    deleted_at: UtcDatetime
    deleted_counts: dict[str, int] = Field(
        default_factory=dict, description="Rows removed, keyed by table name."
    )
    retained: list[str] = Field(
        default_factory=list,
        description="Plain-language statements of what was kept and why.",
    )
