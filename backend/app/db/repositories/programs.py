"""The curated program catalog, read and written.

:class:`SqlProgramRepository` is the SQL implementation of
:class:`app.schemas.contracts.ProgramRepository`. It lives here, next to the
models, rather than in ``app/retrieval/``, so that there is one place that
knows how a catalog row becomes an :class:`~app.schemas.program.AssistanceProgram`;
``app.retrieval`` composes this class rather than reimplementing the mapping.

Two behaviours are load-bearing rather than incidental.

**Soft-deleted programs are invisible to retrieval but resolvable by id.** A
retired program must stop appearing in search results while an action plan a
user saved last month still renders its name and its link. So the list and
search paths filter ``deleted_at IS NULL`` and the by-id paths do not.

**``get_active_ruleset`` returning ``None`` means "cannot assess".** It never
means "does not qualify". A program with no approved rule set in force is still
shown - it is a useful lead - but it is presented as information rather than as
an assessment, and the run records the gap in its limitations.

Full-text search goes through the generated ``search_vector`` column with
``websearch_to_tsquery``, which accepts what a person actually types (bare
words, quoted phrases, ``-excluded``) and, unlike ``to_tsquery``, raises nothing
when they type punctuation.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Select, delete, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.db import models
from app.db.repositories.base import Repository, as_utc, count_of, normalise_page
from app.schemas import (
    AssistanceProgram,
    ProgramCategory,
    ProgramDetail,
    ProgramPage,
    ProgramSource,
    ProgramSummary,
    RuleSet,
    VerificationStatus,
    render_requirements,
)

__all__ = ["SqlProgramRepository", "to_program_summary"]

_log = get_logger(__name__)

#: The text search configuration the generated ``search_vector`` was built
#: with. A query parsed with a different configuration would stem differently
#: and silently fail to match.
_TS_CONFIG = "english"


def _to_source(row: models.ProgramSource) -> ProgramSource:
    return ProgramSource(
        id=row.id,
        url=row.url,
        title=row.title,
        publisher=row.publisher,
        excerpt=row.excerpt,
        retrieved_at=row.retrieved_at,
        content_hash=row.content_hash,
        freshness=row.freshness,
    )


def _to_program(
    row: models.AssistanceProgram, *, sources: Sequence[ProgramSource]
) -> AssistanceProgram:
    return AssistanceProgram(
        id=row.id,
        slug=row.slug,
        official_name=row.official_name,
        administering_org=row.administering_org,
        category=ProgramCategory(row.category),
        description=row.description,
        plain_language_summary=row.plain_language_summary,
        who_it_helps=list(row.who_it_helps),
        geo_scope=row.geo_scope,
        official_url=row.official_url,
        application_url=row.application_url,
        phone=row.phone,
        keywords=list(row.keywords),
        verification_status=VerificationStatus(row.verification_status),
        last_verified_at=row.last_verified_at,
        active_rule_version=row.active_rule_version,
        sources=list(sources),
    )


def to_program_summary(row: models.AssistanceProgram) -> ProgramSummary:
    """Project a catalog row onto the small list-view shape.

    Public because ``MatchRepository`` needs the identical projection: a
    program must read the same on the results page as it does in the catalog.
    """
    return ProgramSummary(
        id=row.id,
        slug=row.slug,
        official_name=row.official_name,
        administering_org=row.administering_org,
        category=ProgramCategory(row.category),
        plain_language_summary=row.plain_language_summary,
        geo_scope_label=row.geo_scope["label"],
        official_url=row.official_url,
        application_url=row.application_url,
        verification_status=VerificationStatus(row.verification_status),
        last_verified_at=row.last_verified_at,
    )


def _to_ruleset(row: models.ProgramRuleVersion, *, program_slug: str) -> RuleSet:
    return RuleSet.model_validate(
        {
            "version": row.version,
            "program_slug": program_slug,
            "effective_from": row.effective_from,
            "effective_to": row.effective_to,
            "root": row.root,
            "source_ids": list(row.source_ids),
            "review_status": row.review_status,
            "limitations": list(row.limitations),
            "reviewed_by": row.reviewed_by,
            "reviewed_at": row.reviewed_at,
        }
    )


class SqlProgramRepository(Repository):
    """Read and write access to the program catalog over one session.

    Satisfies :class:`app.schemas.contracts.ProgramRepository`. The write
    methods are not part of that Protocol - they exist for the seed script and
    the ingestion tooling, which are the only things that may change the
    catalog.
    """

    __slots__ = ()

    # ------------------------------------------------------------------ reads
    def _live_programs(self) -> Select[models.AssistanceProgram]:
        """Base select over programs that have not been retired."""
        return select(models.AssistanceProgram).where(models.AssistanceProgram.deleted_at.is_(None))

    async def list_programs(
        self,
        *,
        categories: Sequence[ProgramCategory] | None = None,
        state: str | None = None,
        county: str | None = None,
        query: str | None = None,
        verification_statuses: Sequence[VerificationStatus] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ProgramPage:
        limit, offset = normalise_page(limit, offset)
        table = models.AssistanceProgram
        statement = self._live_programs().options(selectinload(table.sources))

        if categories:
            statement = statement.where(table.category.in_([c.value for c in categories]))
        if verification_statuses:
            statement = statement.where(
                table.verification_status.in_([s.value for s in verification_statuses])
            )
        if state:
            # A nationwide program is in scope for every state.
            statement = statement.where(
                or_(
                    table.geo_scope["nationwide"].as_boolean().is_(True),
                    table.geo_scope["states"].has_key(state.upper()),
                )
            )
        if county:
            # A program scoped to a whole state lists no counties, and must
            # still match a county query - otherwise filtering by county would
            # hide SNAP.
            statement = statement.where(
                or_(
                    func.jsonb_array_length(
                        func.coalesce(table.geo_scope["counties"], text("'[]'::jsonb"))
                    )
                    == 0,
                    table.geo_scope["counties"].has_key(county),
                )
            )

        cleaned_query = (query or "").strip()
        if cleaned_query:
            tsquery = func.websearch_to_tsquery(_TS_CONFIG, cleaned_query)
            statement = statement.where(table.search_vector.op("@@")(tsquery)).order_by(
                func.ts_rank(table.search_vector, tsquery).desc(), table.official_name.asc()
            )
        else:
            statement = statement.order_by(table.official_name.asc())

        total = await count_of(self.session, statement)
        rows = (await self.session.scalars(statement.limit(limit).offset(offset))).all()
        return ProgramPage(
            items=[to_program_summary(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_program(self, program_id: uuid.UUID) -> ProgramDetail | None:
        row = await self.session.get(
            models.AssistanceProgram,
            program_id,
            options=[selectinload(models.AssistanceProgram.sources)],
        )
        return await self._build_detail(row)

    async def get_program_by_slug(self, slug: str) -> ProgramDetail | None:
        statement = (
            select(models.AssistanceProgram)
            .where(models.AssistanceProgram.slug == slug)
            .options(selectinload(models.AssistanceProgram.sources))
        )
        row = (await self.session.scalars(statement)).one_or_none()
        return await self._build_detail(row)

    async def get_programs_by_ids(
        self, program_ids: Sequence[uuid.UUID]
    ) -> list[AssistanceProgram]:
        if not program_ids:
            return []
        wanted = list(dict.fromkeys(program_ids))
        statement = (
            select(models.AssistanceProgram)
            .where(models.AssistanceProgram.id.in_(wanted))
            .options(selectinload(models.AssistanceProgram.sources))
        )
        rows = (await self.session.scalars(statement)).all()
        by_id = {
            row.id: _to_program(row, sources=[_to_source(s) for s in row.sources]) for row in rows
        }
        # Returned in the order asked for, so a caller can zip the result
        # against its own list without re-sorting.
        return [by_id[pid] for pid in wanted if pid in by_id]

    async def get_active_ruleset(
        self, program_id: uuid.UUID, *, on: date | None = None
    ) -> RuleSet | None:
        # A rule set's effectiveness is a calendar date, not an instant.
        when = on or datetime.now(UTC).date()
        rules = models.ProgramRuleVersion
        statement = (
            select(rules, models.AssistanceProgram.slug)
            .join(models.AssistanceProgram, models.AssistanceProgram.id == rules.program_id)
            .where(
                rules.program_id == program_id,
                rules.review_status == "APPROVED",
                rules.effective_from <= when,
                or_(rules.effective_to.is_(None), rules.effective_to > when),
            )
            .order_by(rules.effective_from.desc())
            .limit(1)
        )
        result = (await self.session.execute(statement)).first()
        if result is None:
            return None
        row, slug = result
        return _to_ruleset(row, program_slug=slug)

    async def get_sources(
        self, program_id: uuid.UUID, *, source_ids: Sequence[str] | None = None
    ) -> list[ProgramSource]:
        statement = select(models.ProgramSource).where(
            models.ProgramSource.program_id == program_id
        )
        if source_ids is not None:
            if not source_ids:
                return []
            statement = statement.where(models.ProgramSource.id.in_(list(source_ids)))
        rows = (await self.session.scalars(statement.order_by(models.ProgramSource.id))).all()
        return [_to_source(row) for row in rows]

    async def search_sources(
        self, query: str, *, program_id: uuid.UUID | None = None, limit: int = 5
    ) -> list[ProgramSource]:
        """Passage search over source excerpts, for grounding the assistant.

        Deliberately a plain case-insensitive substring match rather than
        full-text: the assistant has to quote an excerpt back verbatim and have
        the citation verifier find it again, and a stemmed match would return
        passages that do not contain the words the model was shown.
        """
        cleaned = query.strip()
        if not cleaned:
            return []
        statement = select(models.ProgramSource)
        if program_id is not None:
            statement = statement.where(models.ProgramSource.program_id == program_id)
        pattern = f"%{cleaned}%"
        statement = statement.where(
            or_(
                models.ProgramSource.excerpt.ilike(pattern),
                models.ProgramSource.title.ilike(pattern),
            )
        ).order_by(models.ProgramSource.retrieved_at.desc())
        rows = (await self.session.scalars(statement.limit(max(1, limit)))).all()
        return [_to_source(row) for row in rows]

    async def _build_detail(self, row: models.AssistanceProgram | None) -> ProgramDetail | None:
        if row is None:
            return None
        sources = [_to_source(source) for source in row.sources]
        program = _to_program(row, sources=sources)
        ruleset = await self.get_active_ruleset(row.id)
        if ruleset is None:
            # No approved rules in force: show the program, show no
            # requirements, and say so. Rendering requirements from an
            # unapproved rule set is how unreviewed text reaches a user as
            # fact.
            return ProgramDetail(
                program=program,
                sources=sources,
                requirements=[],
                rule_version=None,
                rule_effective_from=None,
                limitations=["We have not finished recording this program's requirements yet."],
            )
        return ProgramDetail(
            program=program,
            sources=sources,
            requirements=render_requirements(ruleset.root, ruleset.limitations),
            rule_version=ruleset.version,
            rule_effective_from=as_utc(ruleset.effective_from),
            limitations=list(ruleset.limitations),
        )

    # ----------------------------------------------------------------- writes
    async def upsert_program(self, program: AssistanceProgram) -> uuid.UUID:
        """Insert or update one catalog program and replace its sources.

        Idempotent on ``slug``, so re-running the seed script over a changed
        catalog file converges instead of accumulating duplicates. The id is
        ``uuid5(PROGRAM_NAMESPACE, slug)``, which is why it can be supplied
        rather than generated.

        Sources are replaced wholesale rather than merged: the catalog file is
        the source of truth, and a source removed from it must disappear rather
        than linger as evidence for a claim nobody is making any more.
        """
        table = models.AssistanceProgram
        values: dict[str, Any] = {
            "id": program.id,
            "slug": program.slug,
            "official_name": program.official_name,
            "administering_org": program.administering_org,
            "category": program.category.value,
            "description": program.description,
            "plain_language_summary": program.plain_language_summary,
            "who_it_helps": list(program.who_it_helps),
            "geo_scope": program.geo_scope.model_dump(mode="json"),
            "official_url": program.official_url,
            "application_url": program.application_url,
            "phone": program.phone,
            "keywords": list(program.keywords),
            "verification_status": program.verification_status.value,
            "last_verified_at": program.last_verified_at,
            "active_rule_version": program.active_rule_version,
            "deleted_at": None,
        }
        statement = (
            pg_insert(table)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[table.slug],
                set_={key: value for key, value in values.items() if key not in {"id", "slug"}}
                | {"updated_at": func.now()},
            )
            .returning(table.id)
        )
        program_id = (await self.session.execute(statement)).scalar_one()

        await self.session.execute(
            delete(models.ProgramSource).where(models.ProgramSource.program_id == program_id)
        )
        for source in program.sources:
            self.session.add(
                models.ProgramSource(
                    id=source.id,
                    program_id=program_id,
                    url=source.url,
                    title=source.title,
                    publisher=source.publisher,
                    excerpt=source.excerpt,
                    retrieved_at=source.retrieved_at,
                    content_hash=source.content_hash,
                    freshness=source.freshness.value,
                )
            )
        await self.session.flush()
        _log.debug("catalog.program.upserted", program_id=str(program_id), slug=program.slug)
        return uuid.UUID(str(program_id))

    async def upsert_rule_version(
        self, ruleset: RuleSet, *, supersede_open_versions: bool = True
    ) -> uuid.UUID:
        """Insert or update one rule version for the program named by its slug.

        ``supersede_open_versions`` closes any *other* open version of the same
        program by setting its ``effective_to`` to this one's
        ``effective_from``. That is what keeps the partial unique index - at
        most one open version per program - satisfiable while loading a catalog
        that has moved on, and it is also the honest representation: the old
        rules did apply, right up until the new ones did.

        Old versions are never deleted or rewritten, so a match recorded under
        version ``2026.1`` stays explainable after ``2026.2`` lands.
        """
        program_id = await self.session.scalar(
            select(models.AssistanceProgram.id).where(
                models.AssistanceProgram.slug == ruleset.program_slug
            )
        )
        if program_id is None:
            msg = f"no program with slug {ruleset.program_slug!r}; load the program first"
            raise LookupError(msg)

        rules = models.ProgramRuleVersion
        if supersede_open_versions and ruleset.effective_to is None:
            await self.session.execute(
                update(rules)
                .where(
                    rules.program_id == program_id,
                    rules.version != ruleset.version,
                    rules.effective_to.is_(None),
                    rules.effective_from < ruleset.effective_from,
                )
                .values(effective_to=ruleset.effective_from, updated_at=func.now())
            )

        values: dict[str, Any] = {
            "program_id": program_id,
            "version": ruleset.version,
            "effective_from": ruleset.effective_from,
            "effective_to": ruleset.effective_to,
            "root": ruleset.root.model_dump(mode="json"),
            "source_ids": list(ruleset.source_ids),
            "review_status": ruleset.review_status.value,
            "limitations": list(ruleset.limitations),
            "reviewed_by": ruleset.reviewed_by,
            "reviewed_at": ruleset.reviewed_at,
        }
        statement = (
            pg_insert(rules)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[rules.program_id, rules.version],
                set_={
                    key: value
                    for key, value in values.items()
                    if key not in {"program_id", "version"}
                }
                | {"updated_at": func.now()},
            )
            .returning(rules.id)
        )
        rule_version_id = (await self.session.execute(statement)).scalar_one()
        await self.session.flush()
        return uuid.UUID(str(rule_version_id))

    async def set_active_rule_version(self, program_id: uuid.UUID, version: str | None) -> None:
        """Point a program at the rule version that may be matched against.

        Separate from :meth:`upsert_rule_version` on purpose: loading a draft
        rule set must not make it live. Only this call does, and only for a
        version that is already ``APPROVED``.
        """
        await self.session.execute(
            update(models.AssistanceProgram)
            .where(models.AssistanceProgram.id == program_id)
            .values(active_rule_version=version, updated_at=func.now())
        )
        await self.session.flush()

    async def soft_delete_program(self, program_id: uuid.UUID) -> None:
        """Retire a program without destroying anything that references it."""
        await self.session.execute(
            update(models.AssistanceProgram)
            .where(models.AssistanceProgram.id == program_id)
            .values(deleted_at=func.now(), updated_at=func.now())
        )
        await self.session.flush()
