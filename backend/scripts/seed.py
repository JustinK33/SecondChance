"""Load the curated program catalog into Postgres, idempotently.

Run it with ``./do seed``, or directly::

    python -m scripts.seed                 # validate, then upsert
    python -m scripts.seed --validate-only # validate the JSON, touch nothing
    python -m scripts.seed --dry-run       # read the database, write nothing

Guarantees, in the order they matter:

1. **Validation happens before the database is opened.** Every record is parsed
   into the real :mod:`app.schemas` models, cross-referenced (every
   ``source_ids`` entry resolves, every ``id`` equals ``uuid5`` of the slug,
   every rule set names its own program) and only then persisted. A malformed
   catalog can therefore never leave the tables half-written.
2. **Re-running changes nothing.** Programs and sources upsert on their primary
   keys, sources that a record no longer cites are removed, and a rule version
   is only inserted when the rules actually differ from the newest stored
   version. A second run reports ``unchanged`` for everything.
3. **A rule change is a new version, never an edit.** Rule sets are append-only:
   when the conditions differ, the previous row gets an ``effective_to`` and a
   new version is inserted, so an evaluation recorded last month can still be
   explained against the rules that were in force then.
4. **The table shape is read, not duplicated.** The catalog tables are reflected
   from the live database rather than re-declared here, because the ORM models
   and migrations are owned elsewhere; a column rename surfaces as a precise
   error instead of a silent data loss.

The script logs counts and slugs only. Catalog data is public, but the habit is
the point: nothing in this repository prints row contents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import MetaData, Table, delete, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.core.config import get_settings
from app.schemas import (
    PROGRAM_NAMESPACE,
    AssistanceProgram,
    PlainText,
    ReviewStatus,
    RuleSet,
    VerificationStatus,
    program_uuid,
)
from app.schemas.eligibility import AllOf, AnyOf, Condition, Not

__all__ = [
    "CATALOG_TABLES",
    "CatalogRecord",
    "SeedReport",
    "default_catalog_dir",
    "load_catalog",
    "main",
    "seed",
]

#: The three catalog tables, as documented in ``docs/INTERFACES.md`` and created
#: by the Alembic migrations. Reflected at runtime; never declared here.
CATALOG_TABLES: Final[tuple[str, ...]] = (
    "assistance_programs",
    "program_sources",
    "program_rule_versions",
)

#: ``backend/`` - this file is ``backend/scripts/seed.py``.
BACKEND_DIR: Final[Path] = Path(__file__).resolve().parents[1]


def default_catalog_dir() -> Path:
    """Where the curated JSON lives: ``backend/data/programs``."""
    return BACKEND_DIR / "data" / "programs"


# ---------------------------------------------------------------------------
# the on-disk format
# ---------------------------------------------------------------------------
class CatalogRecord(BaseModel):
    """One program as the JSON files carry it: the record, its rules, its gaps.

    ``AssistanceProgram`` deliberately has no slot for an unapproved rule set or
    for ``limitations`` - those belong to a *view* of a program, not to the
    program itself - so the file format wraps the contract types rather than
    extending them.
    """

    model_config = ConfigDict(extra="forbid")

    program: AssistanceProgram
    rule_version: RuleSet | None = Field(
        default=None, description="Rules for this program, or null when none are encoded yet."
    )
    limitations: list[PlainText] = Field(
        default_factory=list,
        description="What this record does not cover. Surfaced to users, never hidden.",
    )

    @model_validator(mode="after")
    def _ids_are_derived_from_the_slug(self) -> Self:
        expected = program_uuid(self.program.slug)
        if self.program.id != expected:
            msg = (
                f"{self.program.slug}: id must be uuid5(PROGRAM_NAMESPACE, slug) "
                f"= {expected}, got {self.program.id}"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _rule_set_names_its_own_program(self) -> Self:
        rules = self.rule_version
        if rules is None:
            return self
        if rules.program_slug != self.program.slug:
            msg = (
                f"{self.program.slug}: rule_version.program_slug is "
                f"{rules.program_slug!r}, which is a different program"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _every_citation_resolves(self) -> Self:
        """No condition or rule set may cite a source the record does not carry.

        This is the structural half of "never fabricate a citation": a dangling
        ``source_ids`` entry would render as a requirement with a link to
        nothing.
        """
        known = {source.id for source in self.program.sources}
        rules = self.rule_version
        if rules is None:
            return self
        missing = sorted(set(rules.source_ids) - known)
        for condition in _walk(rules.root):
            missing.extend(sorted(set(condition.source_ids) - known))
        if missing:
            msg = f"{self.program.slug}: unknown source_ids {sorted(set(missing))}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _active_version_matches_the_rules(self) -> Self:
        """``active_rule_version`` must name a rule set that exists and is approved.

        ``AssistanceProgram.supports_eligibility_claims()`` is ``True`` as soon
        as this field is set, so an unapproved or absent rule set behind it would
        let an unreviewed requirement decide somebody's status.
        """
        active = self.program.active_rule_version
        if active is None:
            return self
        rules = self.rule_version
        if rules is None:
            msg = f"{self.program.slug}: active_rule_version is set but no rule_version is defined"
            raise ValueError(msg)
        if active != rules.version:
            msg = (
                f"{self.program.slug}: active_rule_version {active!r} does not match "
                f"rule_version.version {rules.version!r}"
            )
            raise ValueError(msg)
        if rules.review_status is not ReviewStatus.APPROVED:
            msg = (
                f"{self.program.slug}: active_rule_version points at a "
                f"{rules.review_status.value} rule set; only APPROVED rules may be active"
            )
            raise ValueError(msg)
        if self.program.verification_status is not VerificationStatus.VERIFIED:
            msg = (
                f"{self.program.slug}: a {self.program.verification_status.value} program "
                "must not carry an active_rule_version - it would support an eligibility claim"
            )
            raise ValueError(msg)
        return self

    def rule_payload(self) -> dict[str, Any] | None:
        """The parts of the rule set that decide whether a version must be bumped."""
        rules = self.rule_version
        if rules is None:
            return None
        return {
            "root": rules.root.model_dump(mode="json"),
            "source_ids": list(rules.source_ids),
            "limitations": list(rules.limitations),
            "review_status": rules.review_status.value,
        }


class CatalogFile(BaseModel):
    """One ``data/programs/*.json`` file: a category and its programs."""

    model_config = ConfigDict(extra="forbid")

    category: str
    description: str | None = None
    programs: list[CatalogRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def _category_matches_every_program(self) -> Self:
        wrong = [
            record.program.slug
            for record in self.programs
            if record.program.category.value != self.category
        ]
        if wrong:
            msg = f"programs in the {self.category} file are filed under another category: {wrong}"
            raise ValueError(msg)
        return self


def _walk(condition: Condition) -> list[Condition]:
    """Every node of a condition tree, parents before children."""
    nodes: list[Condition] = [condition]
    if isinstance(condition, AllOf | AnyOf):
        for child in condition.conditions:
            nodes.extend(_walk(child))
    elif isinstance(condition, Not):
        nodes.extend(_walk(condition.condition))
    return nodes


def load_catalog(catalog_dir: Path | None = None) -> list[CatalogRecord]:
    """Parse and cross-check every catalog file. Raises on the first problem.

    Pure and offline: this is the function the unit tests, the seeder and the
    source verifier all share, so "the catalog is valid" means the same thing in
    all three.
    """
    directory = catalog_dir or default_catalog_dir()
    if not directory.is_dir():
        msg = f"catalog directory not found: {directory}"
        raise FileNotFoundError(msg)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        msg = f"no catalog files in {directory}"
        raise FileNotFoundError(msg)

    records: list[CatalogRecord] = []
    seen: dict[str, Path] = {}
    source_owners: dict[str, str] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"{path.name}: not valid JSON ({exc})"
            raise ValueError(msg) from exc
        try:
            parsed = CatalogFile.model_validate(payload)
        except ValidationError as exc:
            msg = f"{path.name}: {exc}"
            raise ValueError(msg) from exc
        for record in parsed.programs:
            slug = record.program.slug
            if slug in seen:
                msg = f"duplicate slug {slug!r} in {path.name} and {seen[slug].name}"
                raise ValueError(msg)
            seen[slug] = path
            # ``program_sources.id`` is the table's primary key and each row
            # belongs to one program, so two programs citing the same id would
            # silently overwrite each other's row on seed.
            for source in record.program.sources:
                owner = source_owners.get(source.id)
                if owner is not None:
                    msg = (
                        f"source id {source.id!r} is used by both {owner!r} and {slug!r}; "
                        "source ids are globally unique and belong to one program - give each "
                        "program its own id for the same document"
                    )
                    raise ValueError(msg)
                source_owners[source.id] = slug
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# database payloads
# ---------------------------------------------------------------------------
def rule_version_uuid(slug: str, version: str) -> uuid.UUID:
    """Deterministic rule-version id, so a re-seed does not create a second row."""
    return uuid.uuid5(PROGRAM_NAMESPACE, f"rule:{slug}:{version}")


def _program_row(record: CatalogRecord) -> dict[str, Any]:
    program = record.program
    return {
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
        # A program that is back in the catalog is not retired. Explicit, because
        # leaving a stale deleted_at would hide it from every query forever.
        "deleted_at": None,
    }


def _source_rows(record: CatalogRecord) -> list[dict[str, Any]]:
    return [
        {
            "id": source.id,
            "program_id": record.program.id,
            "url": source.url,
            "title": source.title,
            "publisher": source.publisher,
            "excerpt": source.excerpt,
            "retrieved_at": source.retrieved_at,
            "content_hash": source.content_hash,
            "freshness": source.freshness.value,
        }
        for source in record.program.sources
    ]


def _rule_row(record: CatalogRecord, version: str) -> dict[str, Any]:
    rules = record.rule_version
    assert rules is not None  # guarded by the caller
    return {
        "id": rule_version_uuid(record.program.slug, version),
        "program_id": record.program.id,
        "version": version,
        "effective_from": rules.effective_from,
        "effective_to": rules.effective_to,
        "root": rules.root.model_dump(mode="json"),
        "source_ids": list(rules.source_ids),
        "review_status": rules.review_status.value,
        "limitations": list(rules.limitations),
        "reviewed_by": rules.reviewed_by,
        "reviewed_at": rules.reviewed_at,
    }


def _next_version(current: str) -> str:
    """``2026.1`` -> ``2026.2``; anything else gains a ``.2`` suffix.

    Monotonic within a program, which is all ``RuleSet.version`` promises.
    """
    head, _, tail = current.rpartition(".")
    if head and tail.isdigit():
        return f"{head}.{int(tail) + 1}"
    return f"{current}.2"


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SeedReport:
    """Counts only. Printed at the end of a run."""

    programs_inserted: int = 0
    programs_updated: int = 0
    sources_written: int = 0
    sources_removed: int = 0
    rules_inserted: int = 0
    rules_unchanged: int = 0
    rules_superseded: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "seed report",
            f"  programs   inserted={self.programs_inserted} updated={self.programs_updated}",
            f"  sources    written={self.sources_written} removed={self.sources_removed}",
            f"  rules      inserted={self.rules_inserted} "
            f"unchanged={self.rules_unchanged} superseded={self.rules_superseded}",
        ]
        lines.extend(f"  note       {note}" for note in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the write path
# ---------------------------------------------------------------------------
def _reflect(sync_connection: Connection) -> MetaData:
    metadata = MetaData()
    metadata.reflect(bind=sync_connection, only=CATALOG_TABLES)
    return metadata


def _require_columns(table: Table, row: dict[str, Any]) -> None:
    """Fail loudly when the table cannot hold what we are about to write.

    Silently dropping an unknown key would mean a renamed column quietly stops
    being seeded, and the first symptom would be a user seeing a program with no
    phone number.
    """
    unknown = sorted(set(row) - set(table.columns.keys()))
    if unknown:
        msg = (
            f"table {table.name} has no column(s) {unknown}; the catalog schema and the "
            "migrations have drifted - see docs/INTERFACES.md section 10"
        )
        raise RuntimeError(msg)


async def _upsert_program(connection: AsyncConnection, table: Table, row: dict[str, Any]) -> bool:
    """Insert or update one program. Returns ``True`` when the row was new."""
    _require_columns(table, row)
    statement = pg_insert(table).values(**row)
    updates: dict[str, Any] = {key: statement.excluded[key] for key in row if key != "id"}
    if "updated_at" in table.columns:
        # ``onupdate`` does not fire for ON CONFLICT DO UPDATE, so the timestamp
        # is set explicitly; otherwise a re-seed would leave updated_at frozen.
        updates["updated_at"] = func.now()
    # ``xmax = 0`` is Postgres' way of telling an upsert whether the row was
    # actually inserted. Comparing created_at to updated_at cannot work here,
    # because an untouched row keeps both.
    # Annotated Any because RETURNING over a literal column has no inferable
    # row type for the type checker.
    upsert: Any = statement.on_conflict_do_update(index_elements=["id"], set_=updates).returning(
        literal_column("xmax = 0")
    )
    result = await connection.execute(upsert)
    return bool(result.scalar())


async def _sync_sources(
    connection: AsyncConnection,
    table: Table,
    program_id: uuid.UUID,
    rows: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Upsert the record's sources and drop any it no longer cites."""
    for row in rows:
        _require_columns(table, row)
        statement = pg_insert(table).values(**row)
        updates: dict[str, Any] = {key: statement.excluded[key] for key in row if key != "id"}
        if "updated_at" in table.columns:
            updates["updated_at"] = func.now()
        await connection.execute(
            statement.on_conflict_do_update(index_elements=["id"], set_=updates)
        )
    keep = [row["id"] for row in rows]
    condition = table.c.program_id == program_id
    if keep:
        condition = condition & table.c.id.notin_(keep)
    removed = await connection.execute(delete(table).where(condition))
    return len(rows), removed.rowcount or 0


async def _sync_rule_version(
    connection: AsyncConnection,
    table: Table,
    record: CatalogRecord,
    report: SeedReport,
) -> None:
    """Insert a rule version only when the rules genuinely changed."""
    rules = record.rule_version
    if rules is None:
        return
    desired = record.rule_payload()
    assert desired is not None

    existing = (
        await connection.execute(
            select(
                table.c.id,
                table.c.version,
                table.c.effective_from,
                table.c.root,
                table.c.source_ids,
                table.c.limitations,
                table.c.review_status,
            )
            .where(table.c.program_id == record.program.id)
            .order_by(table.c.effective_from.desc(), table.c.version.desc())
        )
    ).all()

    for row in existing:
        stored = {
            "root": row.root,
            "source_ids": list(row.source_ids or []),
            "limitations": list(row.limitations or []),
            "review_status": row.review_status,
        }
        if stored == desired:
            report.rules_unchanged += 1
            return

    version = rules.version
    taken = {row.version for row in existing}
    while version in taken:
        version = _next_version(version)

    if existing:
        newest = existing[0]
        # ``RuleSet`` requires effective_to > effective_from, so a same-day
        # supersede still has to leave the old row parseable.
        closes_on = max(rules.effective_from, newest.effective_from + timedelta(days=1))
        await connection.execute(
            table.update()
            .where(table.c.id == newest.id)
            .values(effective_to=closes_on, review_status=ReviewStatus.SUPERSEDED.value)
        )
        report.rules_superseded += 1
        report.notes.append(
            f"{record.program.slug}: rules changed, {newest.version} superseded by {version}"
        )

    row_values = _rule_row(record, version)
    _require_columns(table, row_values)
    await connection.execute(pg_insert(table).values(**row_values))
    report.rules_inserted += 1

    if version != rules.version and record.program.active_rule_version is not None:
        report.notes.append(
            f"{record.program.slug}: stored as {version}; update rule_version.version "
            "and active_rule_version in the catalog file to match"
        )


async def seed(records: Sequence[CatalogRecord], *, dry_run: bool = False) -> SeedReport:
    """Upsert every record. One transaction: a failure leaves nothing behind."""
    settings = get_settings()
    report = SeedReport()
    engine = create_async_engine(settings.database_url_async, pool_pre_ping=True)
    try:
        async with engine.begin() as connection:
            metadata = await connection.run_sync(_reflect)
            missing = [name for name in CATALOG_TABLES if name not in metadata.tables]
            if missing:
                msg = f"catalog tables {missing} do not exist; run `./do migrate` first"
                raise RuntimeError(msg)
            programs = metadata.tables["assistance_programs"]
            sources = metadata.tables["program_sources"]
            rules = metadata.tables["program_rule_versions"]

            for record in records:
                inserted = await _upsert_program(connection, programs, _program_row(record))
                if inserted:
                    report.programs_inserted += 1
                else:
                    report.programs_updated += 1
                written, removed = await _sync_sources(
                    connection, sources, record.program.id, _source_rows(record)
                )
                report.sources_written += written
                report.sources_removed += removed
                await _sync_rule_version(connection, rules, record, report)

            if dry_run:
                report.notes.append("dry run: transaction rolled back, nothing was written")
                await connection.rollback()
    finally:
        await engine.dispose()
    return report


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _summarise(records: Sequence[CatalogRecord]) -> str:
    by_category: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for record in records:
        key = record.program.category.value
        by_category[key] = by_category.get(key, 0) + 1
        status = record.program.verification_status.value
        by_status[status] = by_status.get(status, 0) + 1
    claimable = sum(1 for record in records if record.program.supports_eligibility_claims())
    lines = [f"catalog: {len(records)} programs, {claimable} may support an eligibility claim"]
    lines.append("  by category: " + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items())))
    lines.append("  by status:   " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.seed",
        description="Validate the curated program catalog and upsert it into Postgres.",
    )
    parser.add_argument("--catalog", type=Path, default=None, help="catalog directory")
    parser.add_argument(
        "--validate-only", action="store_true", help="parse and cross-check, then stop"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="run the writes inside a rolled-back transaction"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        records = load_catalog(args.catalog)
    except (FileNotFoundError, ValueError) as exc:
        print(f"catalog is invalid, nothing was written:\n{exc}", file=sys.stderr)
        return 1

    print(_summarise(records))
    if args.validate_only:
        print("validate-only: the database was not opened")
        return 0

    try:
        report = asyncio.run(seed(records, dry_run=args.dry_run))
    except RuntimeError as exc:
        print(f"seed failed: {exc}", file=sys.stderr)
        return 1
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
