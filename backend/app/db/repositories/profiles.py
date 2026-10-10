"""Profile persistence. Versioned, never overwritten.

Saving a profile inserts a new version rather than updating the current row.
Three things depend on that:

* an evaluation recorded three weeks ago stays explainable against the profile
  it was actually made from;
* the "edit my information and re-run" flow can show what changed;
* a completed action plan is never silently re-grounded on information the user
  supplied afterwards.

The cost is that "the user's profile" means "their highest version", which is
what :meth:`ProfileRepository.get_latest` is for. The version number is
allocated inside the insert statement rather than read-then-written, so two
concurrent saves cannot both pick version 4.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select

from app.core.logging import get_logger
from app.db import models
from app.db.repositories.base import UserScopedRepository, dedupe
from app.schemas import (
    EmploymentStatus,
    HousingSituation,
    IncomePeriod,
    NeedCategory,
    ProfileFlag,
    ProfileSnapshot,
    StudentStatus,
    UserProfile,
)

__all__ = ["ProfileRepository"]

_log = get_logger(__name__)


def _to_profile(row: models.Profile) -> UserProfile:
    """Rebuild the validated profile from its columns.

    Everything goes back through ``UserProfile``'s validators, so a row that
    somehow violated the income-range or household-composition rules surfaces
    as a loud validation error here rather than as a wrong match status later.
    """
    return UserProfile(
        state=row.state,
        county=row.county,
        student_status=StudentStatus(row.student_status) if row.student_status else None,
        household_size=row.household_size,
        household_composition=row.household_composition,
        income_min_cents=row.income_min_cents,
        income_max_cents=row.income_max_cents,
        income_period=IncomePeriod(row.income_period) if row.income_period else None,
        employment_status=(
            EmploymentStatus(row.employment_status) if row.employment_status else None
        ),
        housing_situation=(
            HousingSituation(row.housing_situation) if row.housing_situation else None
        ),
        needs=[NeedCategory(need) for need in row.needs],
        age=row.age,
        flags={ProfileFlag(name): bool(value) for name, value in row.flags.items()},
        unknown_fields=list(row.unknown_fields),
        extras=row.extras,
    )


def _to_snapshot(row: models.Profile) -> ProfileSnapshot:
    return ProfileSnapshot(
        version=row.version,
        profile=_to_profile(row),
        created_at=row.created_at,
        changed_fields=list(row.changed_fields),
    )


class ProfileRepository(UserScopedRepository):
    """Read and append profile versions for one user."""

    __slots__ = ()

    async def get_latest(self, user_id: str) -> ProfileSnapshot | None:
        """The user's current profile, or ``None`` if they have none yet."""
        statement = (
            select(models.Profile)
            .where(models.Profile.user_id == user_id)
            .order_by(models.Profile.version.desc())
            .limit(1)
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return _to_snapshot(row) if row else None

    async def get_latest_row(self, user_id: str) -> models.Profile | None:
        """As :meth:`get_latest`, but the ORM row - callers that need its id."""
        statement = (
            select(models.Profile)
            .where(models.Profile.user_id == user_id)
            .order_by(models.Profile.version.desc())
            .limit(1)
        )
        return (await self.session.scalars(statement)).one_or_none()

    async def get_by_id(self, user_id: str, profile_id: uuid.UUID) -> ProfileSnapshot | None:
        """One specific version by id, scoped to its owner.

        The ``user_id`` filter is not redundant with the primary key: the id
        arrives from a request, and without the filter this method would hand
        over any profile whose id someone could guess or observe.
        """
        statement = select(models.Profile).where(
            models.Profile.id == profile_id, models.Profile.user_id == user_id
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return _to_snapshot(row) if row else None

    async def get_version(self, user_id: str, version: int) -> ProfileSnapshot | None:
        statement = select(models.Profile).where(
            models.Profile.user_id == user_id, models.Profile.version == version
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return _to_snapshot(row) if row else None

    async def list_versions(self, user_id: str, *, limit: int = 20) -> list[ProfileSnapshot]:
        statement = (
            select(models.Profile)
            .where(models.Profile.user_id == user_id)
            .order_by(models.Profile.version.desc())
            .limit(max(1, limit))
        )
        rows = (await self.session.scalars(statement)).all()
        return [_to_snapshot(row) for row in rows]

    async def save_new_version(
        self,
        user_id: str,
        profile: UserProfile,
        *,
        changed_fields: Sequence[str] = (),
    ) -> models.Profile:
        """Append a new profile version for ``user_id`` and return its row.

        The version number comes from a scalar subquery evaluated by the
        database as part of the ``INSERT``, not from a prior ``SELECT`` in
        Python. Two requests racing would otherwise both read ``3``, both write
        ``4``, and one would lose to the unique constraint - which is survivable
        but means a user's save silently failed.
        """
        next_version = (
            select(func.coalesce(func.max(models.Profile.version), 0) + 1)
            .where(models.Profile.user_id == user_id)
            .scalar_subquery()
        )
        row = models.Profile(
            user_id=user_id,
            version=next_version,
            state=profile.state,
            county=profile.county,
            student_status=profile.student_status.value if profile.student_status else None,
            household_size=profile.household_size,
            household_composition=profile.household_composition.model_dump(mode="json"),
            income_min_cents=profile.income_min_cents,
            income_max_cents=profile.income_max_cents,
            income_period=profile.income_period.value if profile.income_period else None,
            employment_status=(
                profile.employment_status.value if profile.employment_status else None
            ),
            housing_situation=(
                profile.housing_situation.value if profile.housing_situation else None
            ),
            needs=[need.value for need in profile.needs],
            age=profile.age,
            flags={flag.value: value for flag, value in profile.flags.items()},
            unknown_fields=dedupe(profile.unknown_fields),
            changed_fields=dedupe(changed_fields),
            extras=dict(profile.extras),
        )
        self.session.add(row)
        await self.session.flush()
        # The inserted version was computed server-side, so read it back before
        # anyone relies on it.
        await self.session.refresh(row, ["version", "created_at", "updated_at"])
        _log.debug(
            "profile.version.saved",
            profile_id=str(row.id),
            version=row.version,
            changed_field_count=len(changed_fields),
        )
        return row
