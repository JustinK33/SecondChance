"""Guard rails for the curated program catalog in ``backend/data/programs``.

This catalog is the product's factual substrate: if a record here is wrong, the
app confidently tells a real person something untrue about help they are
counting on. These tests therefore check the data itself, not the code that
loads it, and they deliberately re-derive the file format from the contract
types in :mod:`app.schemas` rather than importing the seeder - so a bug in the
loader cannot make an invalid catalog look valid.

What is enforced here:

* every file parses into :class:`~app.schemas.program.AssistanceProgram` and
  :class:`~app.schemas.eligibility.RuleSet`, which is what makes the rule DSL's
  discriminated union a load-time check rather than a runtime surprise;
* ids, slugs and categories are internally consistent and unique;
* every ``source_ids`` citation resolves to a source carried by the same record;
* a ``VERIFIED`` record actually has the receipts - a non-empty excerpt and a
  ``last_verified_at``;
* anything not ``VERIFIED`` cannot support an eligibility claim;
* every URL is ``https`` on a plausible official domain, because a typo'd or
  hijackable link in a benefits navigator is a safety problem, not a cosmetic
  one.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from app.schemas import (
    AssistanceProgram,
    ReviewStatus,
    RuleSet,
    SourceFreshness,
    VerificationStatus,
    program_uuid,
    utcnow,
)
from app.schemas.eligibility import AllOf, AnyOf, Condition, Not
from app.schemas.enums import ProgramCategory

CATALOG_DIR = Path(__file__).resolve().parents[2] / "data" / "programs"

#: Domains an ``official_url`` or a source URL may live on. Government sites,
#: Rutgers, and the specific nonprofits whose own pages we cite. Anything else
#: is either a mistake or a third-party retelling, and a retelling must never be
#: the authority for a requirement.
OFFICIAL_DOMAIN_SUFFIXES = (
    "nj.gov",
    "state.nj.us",
    "childcarenj.gov",
    "rutgers.edu",
    "cfbnj.org",
    "elijahspromise.org",
    "nj211.org",
    "sharesnation.org",
)

#: Additional domains an ``application_url`` may use. Agencies outsource intake
#: to vendor portals, so this list is wider - but it is still a list.
APPLICATION_DOMAIN_SUFFIXES = (
    *OFFICIAL_DOMAIN_SUFFIXES,
    "mynjhelps.gov",
    "app.pantrysoft.com",
    "dmahs-nj.my.site.com",
)

#: The catalog is curated by hand and reviewed by hand. A file count explosion
#: means something generated records, which is exactly what must not happen.
EXPECTED_PROGRAM_RANGE = (20, 25)


def _catalog_files() -> list[Path]:
    return sorted(CATALOG_DIR.glob("*.json"))


def _load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _records() -> list[tuple[Path, dict[str, Any]]]:
    pairs: list[tuple[Path, dict[str, Any]]] = []
    for path in _catalog_files():
        for raw in _load(path)["programs"]:
            pairs.append((path, raw))
    return pairs


def _walk(condition: Condition) -> list[Condition]:
    nodes: list[Condition] = [condition]
    if isinstance(condition, AllOf | AnyOf):
        for child in condition.conditions:
            nodes.extend(_walk(child))
    elif isinstance(condition, Not):
        nodes.extend(_walk(condition.condition))
    return nodes


def _host_allowed(url: str, suffixes: tuple[str, ...]) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


CATALOG_FILES = _catalog_files()
RECORDS = _records()
RECORD_IDS = [raw["program"]["slug"] for _, raw in RECORDS]


def test_catalog_directory_is_populated() -> None:
    assert CATALOG_DIR.is_dir(), f"missing catalog directory {CATALOG_DIR}"
    assert CATALOG_FILES, "no catalog JSON files found"


def test_program_count_is_in_the_curated_range() -> None:
    low, high = EXPECTED_PROGRAM_RANGE
    assert low <= len(RECORDS) <= high, (
        f"{len(RECORDS)} programs: the catalog is curated by hand and is expected to hold "
        f"between {low} and {high} records"
    )


@pytest.mark.parametrize("path", CATALOG_FILES, ids=lambda path: path.name)
def test_file_declares_one_valid_category(path: Path) -> None:
    payload = _load(path)
    assert set(payload) <= {"category", "description", "programs"}, (
        f"{path.name}: unexpected top-level keys"
    )
    category = ProgramCategory(payload["category"])
    assert payload["programs"], f"{path.name}: no programs"
    for raw in payload["programs"]:
        assert raw["program"]["category"] == category.value, (
            f"{path.name}: {raw['program']['slug']} is filed under the wrong category"
        )


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_record_parses_into_the_contract_models(path: Path, raw: dict[str, Any]) -> None:
    assert set(raw) == {"program", "rule_version", "limitations"}, (
        f"{path.name}: a record must hold exactly program, rule_version and limitations"
    )
    program = AssistanceProgram.model_validate(raw["program"])
    assert program.slug
    if raw["rule_version"] is not None:
        rules = RuleSet.model_validate(raw["rule_version"])
        # Parsing the root at all proves every condition matched the
        # discriminated union; walking it proves the nesting is reachable.
        assert _walk(rules.root)
    for limitation in raw["limitations"]:
        assert limitation.strip(), f"{program.slug}: empty limitation string"


def test_slugs_and_ids_are_unique_and_derived() -> None:
    seen: dict[str, str] = {}
    ids: dict[uuid.UUID, str] = {}
    for path, raw in RECORDS:
        program = AssistanceProgram.model_validate(raw["program"])
        assert program.slug not in seen, (
            f"duplicate slug {program.slug!r} in {path.name} and {seen.get(program.slug)}"
        )
        seen[program.slug] = path.name
        assert program.id == program_uuid(program.slug), (
            f"{program.slug}: id must be uuid5(PROGRAM_NAMESPACE, slug)"
        )
        assert program.id not in ids, f"{program.slug}: id collides with {ids.get(program.id)}"
        ids[program.id] = program.slug


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_every_citation_resolves_to_a_source_in_the_same_record(
    path: Path, raw: dict[str, Any]
) -> None:
    program = AssistanceProgram.model_validate(raw["program"])
    known = {source.id for source in program.sources}
    assert len(known) == len(program.sources), f"{program.slug}: duplicate source ids"
    if raw["rule_version"] is None:
        return
    rules = RuleSet.model_validate(raw["rule_version"])
    assert set(rules.source_ids) <= known, (
        f"{program.slug}: rule set cites unknown sources {sorted(set(rules.source_ids) - known)}"
    )
    for condition in _walk(rules.root):
        unknown = sorted(set(condition.source_ids) - known)
        assert not unknown, (
            f"{program.slug}: condition {condition.kind!r} cites unknown sources {unknown}"
        )


def test_source_ids_are_globally_unique() -> None:
    """``program_sources.id`` is the table's primary key, so two programs citing
    the same id would silently overwrite each other's row on seed."""
    owners: dict[str, str] = {}
    for _path, raw in RECORDS:
        program = AssistanceProgram.model_validate(raw["program"])
        for source in program.sources:
            assert source.id not in owners, (
                f"source id {source.id!r} is used by both {owners.get(source.id)!r} and "
                f"{program.slug!r}; give each program its own id for the same document"
            )
            owners[source.id] = program.slug


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_one_source_per_page_per_program(path: Path, raw: dict[str, Any]) -> None:
    """``program_sources`` is unique on ``(program_id, url)``.

    Two passages from the same page therefore belong in one source entry with
    one concatenated excerpt, not in two rows the database will reject.
    """
    program = AssistanceProgram.model_validate(raw["program"])
    urls = [source.url for source in program.sources]
    duplicates = sorted({url for url in urls if urls.count(url) > 1})
    assert not duplicates, (
        f"{program.slug}: cites the same page twice, which the "
        f"(program_id, url) unique constraint rejects: {duplicates}"
    )


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_rule_set_belongs_to_its_program(path: Path, raw: dict[str, Any]) -> None:
    program = AssistanceProgram.model_validate(raw["program"])
    if raw["rule_version"] is None:
        assert program.active_rule_version is None, (
            f"{program.slug}: active_rule_version is set but there are no rules"
        )
        return
    rules = RuleSet.model_validate(raw["rule_version"])
    assert rules.program_slug == program.slug, (
        f"{program.slug}: rule set names a different program ({rules.program_slug})"
    )
    if program.active_rule_version is not None:
        assert program.active_rule_version == rules.version, (
            f"{program.slug}: active_rule_version does not match the rule set version"
        )
        assert rules.review_status is ReviewStatus.APPROVED, (
            f"{program.slug}: only an APPROVED rule set may be active"
        )


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_verified_records_carry_their_evidence(path: Path, raw: dict[str, Any]) -> None:
    program = AssistanceProgram.model_validate(raw["program"])
    if program.verification_status is not VerificationStatus.VERIFIED:
        return
    assert program.last_verified_at is not None, f"{program.slug}: VERIFIED with no fetch date"
    assert program.last_verified_at <= utcnow(), (
        f"{program.slug}: last_verified_at is in the future"
    )
    assert program.sources, f"{program.slug}: VERIFIED with no sources"
    assert any(source.excerpt.strip() for source in program.sources), (
        f"{program.slug}: VERIFIED but no source carries a non-empty excerpt"
    )
    for source in program.sources:
        assert source.excerpt.strip(), f"{program.slug}/{source.id}: empty excerpt"
        assert source.retrieved_at <= utcnow(), f"{program.slug}/{source.id}: fetched in the future"
        assert source.current_freshness() is not SourceFreshness.STALE, (
            f"{program.slug}/{source.id}: source is stale and cannot support a claim"
        )


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_unverified_records_cannot_support_an_eligibility_claim(
    path: Path, raw: dict[str, Any]
) -> None:
    program = AssistanceProgram.model_validate(raw["program"])
    if program.verification_status is VerificationStatus.VERIFIED:
        return
    assert not program.supports_eligibility_claims(), (
        f"{program.slug}: {program.verification_status.value} but may support a claim"
    )
    assert program.active_rule_version is None, (
        f"{program.slug}: {program.verification_status.value} records must have no active rules"
    )
    assert raw["limitations"], (
        f"{program.slug}: a {program.verification_status.value} record must say what is missing"
    )
    if raw["rule_version"] is not None:
        rules = RuleSet.model_validate(raw["rule_version"])
        assert rules.review_status is not ReviewStatus.APPROVED, (
            f"{program.slug}: unverified record carries an APPROVED rule set"
        )


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_user_facing_prose_is_present(path: Path, raw: dict[str, Any]) -> None:
    program = AssistanceProgram.model_validate(raw["program"])
    assert program.plain_language_summary.strip(), f"{program.slug}: empty plain_language_summary"
    assert len(program.plain_language_summary.split()) >= 10, (
        f"{program.slug}: plain_language_summary is too short to be useful"
    )
    assert program.description.strip(), f"{program.slug}: empty description"
    assert program.who_it_helps, f"{program.slug}: no who_it_helps bullets"
    assert program.keywords, f"{program.slug}: no keywords, so lexical retrieval cannot find it"
    assert program.geo_scope.label.strip(), f"{program.slug}: empty geo_scope label"


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_every_condition_explains_itself(path: Path, raw: dict[str, Any]) -> None:
    if raw["rule_version"] is None:
        return
    program = AssistanceProgram.model_validate(raw["program"])
    rules = RuleSet.model_validate(raw["rule_version"])
    for condition in _walk(rules.root):
        assert condition.description.strip(), (
            f"{program.slug}: a {condition.kind!r} condition has no description"
        )
        assert condition.source_ids, (
            f"{program.slug}: a {condition.kind!r} condition cites no source"
        )


@pytest.mark.parametrize(("path", "raw"), RECORDS, ids=RECORD_IDS)
def test_urls_are_https_and_on_official_domains(path: Path, raw: dict[str, Any]) -> None:
    program = AssistanceProgram.model_validate(raw["program"])
    checked: list[tuple[str, str, tuple[str, ...]]] = [
        ("official_url", program.official_url, OFFICIAL_DOMAIN_SUFFIXES)
    ]
    if program.application_url is not None:
        checked.append(("application_url", program.application_url, APPLICATION_DOMAIN_SUFFIXES))
    checked.extend(
        (f"source {source.id}", source.url, OFFICIAL_DOMAIN_SUFFIXES) for source in program.sources
    )
    for label, url, suffixes in checked:
        assert urlsplit(url).scheme == "https", f"{program.slug}: {label} is not https: {url}"
        assert _host_allowed(url, suffixes), (
            f"{program.slug}: {label} is not on a recognised official domain: {url}"
        )
