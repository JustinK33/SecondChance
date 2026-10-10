"""Persisting and reading the rule engine's verdicts.

Writing a match is an upsert on ``(run_id, program_id)``. Re-evaluating a run
after a profile edit therefore updates the rows it affects and leaves the rest
alone, which is the whole point of
:meth:`app.schemas.contracts.EligibilityEngine.affected_programs`: editing
household size must not re-bill an explanation for a program whose rules never
read it.

Reading a match reconstructs the full
:class:`~app.schemas.matching.ProgramMatch`, including its
:class:`~app.schemas.eligibility.EligibilityEvaluation`. That object's
validators then re-assert, on the way out of the database, the two invariants
that matter most:

* ``status`` agrees with ``outcome`` through ``STATUS_BY_OUTCOME``;
* ``ProgramMatch.status is evaluation.status``.

So a row that had somehow acquired a status inconsistent with its evidence -
by a bad migration, a manual ``UPDATE``, or a future bug - fails loudly here
instead of being rendered to a person as an assessment.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.logging import get_logger
from app.db import models
from app.db.repositories.base import (
    UserScopedRepository,
    dedupe,
    execute_dml,
    to_float,
)
from app.db.repositories.programs import to_program_summary
from app.schemas import (
    EligibilityEvaluation,
    MatchExplanation,
    MatchStatus,
    ProgramMatch,
)

__all__ = ["MatchRepository"]

_log = get_logger(__name__)


class MatchRepository(UserScopedRepository):
    """Read and write the persisted verdicts for one user's runs."""

    __slots__ = ()

    async def upsert_many(
        self,
        user_id: str,
        run_id: uuid.UUID,
        matches: Sequence[ProgramMatch],
        *,
        rule_version_ids: dict[uuid.UUID, uuid.UUID] | None = None,
    ) -> int:
        """Insert or update the given matches for ``run_id``.

        :param rule_version_ids: optional ``program_id -> rule_version_row_id``
            map, so a stored match can point at the exact rule row and not only
            at its version string. Retrieval knows these ids; the engine works
            in version strings, which is why it is a separate argument rather
            than part of :class:`~app.schemas.matching.ProgramMatch`.
        :returns: the number of rows written.
        """
        if not matches:
            return 0
        ids = rule_version_ids or {}
        rows: list[dict[str, Any]] = []
        for match in matches:
            evaluation = match.evaluation
            rows.append(
                {
                    "user_id": user_id,
                    "run_id": run_id,
                    "program_id": evaluation.program_id,
                    "rule_version": evaluation.rule_version,
                    "rule_version_id": ids.get(evaluation.program_id),
                    "status": match.status.value,
                    "outcome": evaluation.outcome.value,
                    "score": match.score,
                    "missing_fields": dedupe(evaluation.missing_fields),
                    "evidence": [item.model_dump(mode="json") for item in evaluation.evidence],
                    "explanation": (
                        match.explanation.model_dump(mode="json") if match.explanation else None
                    ),
                    "limitations": dedupe(evaluation.limitations),
                    "evaluated_at": evaluation.evaluated_at,
                }
            )

        table = models.ProgramMatch
        statement = pg_insert(table).values(rows)
        updatable = (
            "rule_version",
            "rule_version_id",
            "status",
            "outcome",
            "score",
            "missing_fields",
            "evidence",
            "explanation",
            "limitations",
            "evaluated_at",
        )
        statement = statement.on_conflict_do_update(
            index_elements=[table.run_id, table.program_id],
            set_={name: statement.excluded[name] for name in updatable}
            | {"updated_at": func.now()},
        )
        await self.session.execute(statement)
        await self.session.flush()
        _log.debug("matches.upserted", run_id=str(run_id), count=len(rows))
        return len(rows)

    async def set_explanation(
        self,
        user_id: str,
        run_id: uuid.UUID,
        program_id: uuid.UUID,
        explanation: MatchExplanation,
    ) -> bool:
        """Attach model-written prose to an already-decided match.

        Writes only ``explanation``. There is deliberately no code path from
        this method to ``status``, ``outcome`` or ``evidence``, which is what
        makes "language generation cannot move a verdict" a property of the
        schema rather than a rule someone has to remember.
        """
        table = models.ProgramMatch
        affected = await execute_dml(
            self.session,
            update(table)
            .where(
                table.run_id == run_id,
                table.program_id == program_id,
                table.user_id == user_id,
            )
            .values(explanation=explanation.model_dump(mode="json"), updated_at=func.now()),
        )
        await self.session.flush()
        return bool(affected)

    async def list_for_run(self, user_id: str, run_id: uuid.UUID) -> list[ProgramMatch]:
        """Every match for a run, as validated :class:`ProgramMatch` objects.

        Ordered so that the most actionable results come first:
        ``POTENTIAL_MATCH``, then ``MORE_INFO_NEEDED``, then
        ``UNLIKELY_MATCH``, and by descending score within each group. The
        ordering is in SQL because it is the order the user reads, and a
        ranking that depends on which code path assembled the page is a
        ranking that will eventually differ between two screens.
        """
        matches = models.ProgramMatch
        programs = models.AssistanceProgram
        status_rank = case(
            {
                MatchStatus.POTENTIAL_MATCH.value: 0,
                MatchStatus.MORE_INFO_NEEDED.value: 1,
                MatchStatus.UNLIKELY_MATCH.value: 2,
            },
            value=matches.status,
            else_=3,
        )
        statement = (
            select(matches, programs)
            .join(programs, programs.id == matches.program_id)
            .where(matches.run_id == run_id, matches.user_id == user_id)
            .order_by(status_rank.asc(), matches.score.desc(), programs.official_name.asc())
        )
        rows = (await self.session.execute(statement)).all()
        return [self._to_match(match, program) for match, program in rows]

    async def list_rows_for_run(self, user_id: str, run_id: uuid.UUID) -> list[models.ProgramMatch]:
        """The raw rows, for callers that only need ids and statuses."""
        statement = (
            select(models.ProgramMatch)
            .where(
                models.ProgramMatch.run_id == run_id,
                models.ProgramMatch.user_id == user_id,
            )
            .order_by(models.ProgramMatch.evaluated_at.asc())
        )
        return list((await self.session.scalars(statement)).all())

    async def counts_for_run(self, user_id: str, run_id: uuid.UUID) -> dict[MatchStatus, int]:
        statement = (
            select(models.ProgramMatch.status, func.count())
            .where(
                models.ProgramMatch.run_id == run_id,
                models.ProgramMatch.user_id == user_id,
            )
            .group_by(models.ProgramMatch.status)
        )
        rows = (await self.session.execute(statement)).all()
        return {MatchStatus(status): count for status, count in rows}

    async def delete_for_run(self, user_id: str, run_id: uuid.UUID) -> int:
        table = models.ProgramMatch
        affected = await execute_dml(
            self.session,
            delete(table).where(table.run_id == run_id, table.user_id == user_id),
        )
        await self.session.flush()
        return affected

    @staticmethod
    def _to_match(row: models.ProgramMatch, program: models.AssistanceProgram) -> ProgramMatch:
        evaluation = EligibilityEvaluation.model_validate(
            {
                "program_id": row.program_id,
                "program_slug": program.slug,
                "rule_version": row.rule_version,
                "status": row.status,
                "outcome": row.outcome,
                "score": to_float(row.score),
                "evidence": row.evidence,
                "missing_fields": list(row.missing_fields),
                "limitations": list(row.limitations),
                "evaluated_at": row.evaluated_at,
            }
        )
        return ProgramMatch(
            program=to_program_summary(program),
            status=MatchStatus(row.status),
            score=to_float(row.score),
            evaluation=evaluation,
            explanation=(
                MatchExplanation.model_validate(row.explanation) if row.explanation else None
            ),
            sources=[],
        )
