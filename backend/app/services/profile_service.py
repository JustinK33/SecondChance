"""Profile reads and edits, and what an edit should set off.

A user who corrects "household of 3" to "household of 4" has not asked for a
new workflow run; they have asked for their existing results to be right. So an
edit here does three things:

1. assembles the new profile through the single precedence policy in
   :mod:`app.services.profile_assembly` - typed values beat stored values beat
   extraction - and records which fields actually moved;
2. saves it as a new *version*, so an old evaluation stays auditable against
   the profile it was really made from rather than appearing to have been made
   from the current one;
3. asks the rule engine which programs could possibly read one of the changed
   fields, and triggers a **selective** re-evaluation of those for the user's
   most recent completed run.

Step 3 is the one worth defending. ``EligibilityEngine.affected_programs`` is
static over the rule trees, so it answers "could this program's rules read
``household_size``?" without evaluating anything. Editing a field that no
program reads therefore costs nothing at all, and editing one that two programs
read re-evaluates two programs instead of twenty - and makes no model call,
because statuses come from the engine and the prose the user already read is
still accurate for the programs that did not move.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db import models
from app.db.repositories import (
    AuditAction,
    AuditRepository,
    MatchRepository,
    ProfileRepository,
    SqlProgramRepository,
)
from app.eligibility.engine import RuleEngine
from app.retrieval.search import HybridRetriever
from app.schemas import (
    MatchRequest,
    ProfileDraft,
    ProfileSnapshot,
    UserProfile,
    WorkflowCreateResponse,
    WorkflowStatus,
)
from app.services import workflow_service
from app.services.profile_assembly import assemble_profile

__all__ = [
    "ProfileUpdate",
    "affected_program_ids",
    "get_profile",
    "latest_profile",
    "list_versions",
    "update_profile",
]

_log = get_logger(__name__)


class ProfileUpdate:
    """What an edit changed, and what it set off.

    Returned rather than a bare profile because the caller - and the user -
    needs to know whether anything is being re-checked. "We updated your
    information and nothing you were matched to changes" is a useful answer.
    """

    __slots__ = ("affected_program_ids", "changed_fields", "profile", "regeneration", "version")

    def __init__(
        self,
        *,
        profile: UserProfile,
        version: int,
        changed_fields: Sequence[str],
        affected_program_ids: Sequence[uuid.UUID],
        regeneration: WorkflowCreateResponse | None,
    ) -> None:
        self.profile = profile
        self.version = version
        self.changed_fields = list(changed_fields)
        self.affected_program_ids = list(affected_program_ids)
        self.regeneration = regeneration


async def latest_profile(session: AsyncSession, user_id: str) -> UserProfile | None:
    """The caller's newest stored profile, or ``None`` if they have none."""
    snapshot = await ProfileRepository(session).get_latest(user_id)
    return snapshot.profile if snapshot is not None else None


async def get_profile(session: AsyncSession, user_id: str) -> ProfileSnapshot:
    """The caller's newest stored profile version."""
    snapshot = await ProfileRepository(session).get_latest(user_id)
    if snapshot is None:
        raise NotFoundError("We do not have any information saved for you yet.")
    return snapshot


async def list_versions(
    session: AsyncSession, user_id: str, *, limit: int = 20
) -> list[ProfileSnapshot]:
    """Profile history, newest first."""
    return await ProfileRepository(session).list_versions(user_id, limit=limit)


async def affected_program_ids(
    session: AsyncSession,
    user_id: str,
    run_id: uuid.UUID,
    changed_fields: Sequence[str],
) -> list[uuid.UUID]:
    """Which of a run's programs could read one of ``changed_fields``.

    Static over the rule trees, so this is answerable before anything has been
    re-evaluated - which is exactly what makes a selective regeneration
    possible rather than aspirational.
    """
    if not changed_fields:
        return []
    matches = await MatchRepository(session).list_for_run(user_id, run_id)
    if not matches:
        return []
    retriever = HybridRetriever(SqlProgramRepository(session))
    candidates = await retriever.retrieve_for_programs([match.program.id for match in matches])
    return RuleEngine().affected_programs(changed_fields, candidates)


async def update_profile(
    session: AsyncSession,
    user_id: str,
    changes: ProfileDraft,
    *,
    request_id: str,
    regenerate: bool = True,
) -> ProfileUpdate:
    """Apply an edit, version it, and selectively re-check what it affects.

    ``regenerate=False`` saves the edit without re-checking anything, which is
    what a multi-step editor wants between steps.
    """
    base = await latest_profile(session, user_id)
    profile, changed_fields = assemble(base, changes)

    row = await ProfileRepository(session).save_new_version(
        user_id, profile, changed_fields=changed_fields
    )
    await AuditRepository(session).record(
        user_id,
        AuditAction.PROFILE_SAVED,
        resource_type="profile",
        resource_id=row.id,
        request_id=request_id,
        metadata={"version": row.version, "changed_field_count": len(changed_fields)},
    )
    await session.commit()

    run_id = await _latest_completed_run_id(session, user_id)
    affected: list[uuid.UUID] = []
    regeneration: WorkflowCreateResponse | None = None

    if changed_fields and run_id is not None:
        affected = await affected_program_ids(session, user_id, run_id, changed_fields)
        if affected and regenerate:
            regeneration = await workflow_service.reevaluate(
                session,
                user_id,
                MatchRequest(run_id=run_id, changes=changes),
                request_id=request_id,
            )

    _log.info(
        "profile.updated",
        version=row.version,
        changed_field_count=len(changed_fields),
        affected_programs=len(affected),
        regenerated=regeneration is not None,
    )
    return ProfileUpdate(
        profile=profile,
        version=row.version,
        changed_fields=changed_fields,
        affected_program_ids=affected,
        regeneration=regeneration,
    )


def assemble(base: UserProfile | None, changes: ProfileDraft) -> tuple[UserProfile, list[str]]:
    """Apply ``changes`` to ``base`` through the shared precedence policy."""
    return assemble_profile(None, base=base, prefilled=changes)


async def _latest_completed_run_id(session: AsyncSession, user_id: str) -> uuid.UUID | None:
    """The newest run whose results an edit could invalidate."""
    statement = (
        select(models.WorkflowRun.id)
        .where(
            models.WorkflowRun.user_id == user_id,
            models.WorkflowRun.status == WorkflowStatus.COMPLETED.value,
        )
        .order_by(models.WorkflowRun.created_at.desc())
        .limit(1)
    )
    return (await session.scalars(statement)).one_or_none()
