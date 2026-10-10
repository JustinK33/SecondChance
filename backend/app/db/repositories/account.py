"""Account data deletion. The one place a privacy promise becomes real.

"Delete my data" has to mean it, so this is a hard delete rather than a
``deleted_at`` stamp, it happens in one transaction, and it returns counts the
user can read back.

What is removed, child-first so no foreign key ever blocks a parent:

#. ``action_items``
#. ``action_plans``
#. ``program_matches``
#. the LangGraph checkpoint rows for this user's threads - which is where the
   narrative, the extracted profile and the clarification answers actually live
#. ``workflow_runs``
#. ``profiles``
#. ``audit_events``

The checkpoint step matters more than its position in the list suggests.
``checkpoints``, ``checkpoint_blobs`` and ``checkpoint_writes`` are created by
``AsyncPostgresSaver.setup()`` and have no ``user_id`` column, so Alembic does
not manage them and a cascade cannot reach them. Deleting a user's runs without
deleting their checkpoints would leave every narrative they ever submitted
sitting in the database - the single worst outcome this routine could have. The
link is ``workflow_runs.thread_id``, which is why that column exists.

The tables are addressed defensively through ``to_regclass`` because the
checkpointer creates them lazily: on a fresh database that has never run a
workflow they do not exist yet, and a deletion request must not fail because
the user never got far enough to create one.

``model_calls`` is deliberately retained. Its ``run_id`` is
``ON DELETE SET NULL``, so what survives is token counts and latencies with no
reference to the user, the run or anything they wrote. That is reported honestly
in :attr:`~app.schemas.system.AccountDeletionReceipt.retained` rather than
quietly omitted.

Finally one audit event is written, *after* the user's audit history has been
removed, recording that the deletion happened and how many rows it touched. It
carries no personal content, and it is the proof the user was given a real
answer.
"""

from __future__ import annotations

from typing import Any, Final

from sqlalchemy import delete, func, select, text

from app.core.logging import get_logger, hash_user_id
from app.db import models
from app.db.repositories.audit import AuditAction, AuditRepository
from app.db.repositories.base import UserScopedRepository, execute_dml
from app.schemas import AccountDeletionReceipt, utcnow

__all__ = ["CHECKPOINT_TABLES", "AccountDataRepository"]

_log = get_logger(__name__)

#: LangGraph checkpoint tables keyed by ``thread_id``. ``checkpoint_migrations``
#: is excluded on purpose: it is the checkpointer's own schema-version table and
#: holds no user data.
CHECKPOINT_TABLES: Final[tuple[str, ...]] = (
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
)


class AccountDataRepository(UserScopedRepository):
    """Removes everything belonging to one user, in one transaction."""

    __slots__ = ()

    async def delete_user_data(self, user_id: str) -> AccountDeletionReceipt:
        """Delete every row this user owns and return a receipt of the counts.

        Does not commit. The caller - ``app.services.account_service`` - owns
        the transaction boundary, which is what makes the whole deletion
        atomic: either all of it happened or none of it did, and there is no
        window in which a user's plans are gone but their narratives are not.
        """
        counts: dict[str, int] = {}

        thread_ids = list(
            (
                await self.session.scalars(
                    select(models.WorkflowRun.thread_id).where(
                        models.WorkflowRun.user_id == user_id
                    )
                )
            ).all()
        )

        # Children before parents. The cascades would handle most of this, but
        # doing it explicitly is what produces a per-table count, and a receipt
        # that says "we deleted 7 action items" is worth more than one that
        # says "we deleted 1 run".
        child_tables: tuple[tuple[str, Any], ...] = (
            ("action_items", models.ActionItem),
            ("action_plans", models.ActionPlan),
            ("program_matches", models.ProgramMatch),
        )
        for name, table in child_tables:
            counts[name] = await execute_dml(
                self.session, delete(table).where(table.user_id == user_id)
            )

        counts.update(await self._delete_checkpoints(thread_ids))

        parent_tables: tuple[tuple[str, Any], ...] = (
            ("workflow_runs", models.WorkflowRun),
            ("profiles", models.Profile),
            ("audit_events", models.AuditEvent),
        )
        for name, table in parent_tables:
            counts[name] = await execute_dml(
                self.session, delete(table).where(table.user_id == user_id)
            )

        deleted_at = utcnow()
        await AuditRepository(self.session).record(
            user_id,
            AuditAction.ACCOUNT_DATA_DELETED,
            resource_type="account",
            resource_id=hash_user_id(user_id),
            metadata={"deleted_counts": counts, "thread_count": len(thread_ids)},
        )
        _log.info(
            "account.data.deleted",
            user_id_hash=hash_user_id(user_id),
            row_count=sum(counts.values()),
            thread_count=len(thread_ids),
        )

        return AccountDeletionReceipt(
            deleted_at=deleted_at,
            deleted_counts=counts,
            retained=[
                (
                    "Aggregate model usage records - token counts, latencies and costs - "
                    "with every reference to you removed. They contain nothing you wrote "
                    "and nothing that identifies you."
                ),
                (
                    "One audit record showing that this deletion happened, with the "
                    "number of rows removed and no personal information."
                ),
            ],
        )

    async def _delete_checkpoints(self, thread_ids: list[str]) -> dict[str, int]:
        """Remove the LangGraph checkpoint rows for ``thread_ids``.

        Returns zero counts, not an error, when the checkpoint tables do not
        exist yet - the checkpointer creates them on first use, and a user who
        never completed a run must still be able to delete their account.
        """
        counts: dict[str, int] = dict.fromkeys(CHECKPOINT_TABLES, 0)
        if not thread_ids:
            return counts
        for table_name in CHECKPOINT_TABLES:
            exists = await self.session.scalar(select(func.to_regclass(f"public.{table_name}")))
            if exists is None:
                continue
            # Parameterised despite the interpolated table name: the name comes
            # from this module's own constant tuple and never from a caller.
            counts[table_name] = await execute_dml(
                self.session,
                text(
                    f"DELETE FROM public.{table_name} WHERE thread_id = ANY(:thread_ids)"  # noqa: S608
                ).bindparams(thread_ids=thread_ids),
            )
        return counts
