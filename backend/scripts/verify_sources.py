"""Re-fetch every catalog source and report how fresh the evidence still is.

**This is an operational script. It must never run inside a web request.** It
makes outbound network calls to third-party government and nonprofit sites with
a bounded concurrency budget, so a request-scoped invocation would add tens of
seconds of latency, hand an unauthenticated caller an SSRF-shaped amplifier,
and get the deployment rate-limited by the agencies we depend on. Run it from a
terminal or a scheduled job::

    python -m scripts.verify_sources                 # re-fetch and compare
    python -m scripts.verify_sources --offline       # age report only, no I/O
    python -m scripts.verify_sources --json          # machine-readable output

What it reports, per source:

``FRESH`` / ``AGING`` / ``STALE``
    The age bucket from :func:`app.schemas.program.compute_freshness`, recomputed
    from ``retrieved_at``. ``STALE`` sources are excluded from eligibility claims
    by the product rules, so they are the queue of work.
``CHANGED``
    The page still resolves, but its normalised text no longer hashes to the
    recorded ``content_hash``. That is the signal that an agency may have edited
    a requirement, and it is why a rule set goes back to review rather than
    silently changing anybody's results.
``DEAD``
    A non-2xx response, a connection failure or a timeout.

The hash is taken over *normalised visible text*, not the raw response body, so
that a rotating CSRF token, a cache-busting asset version or a reformatted
whitespace run does not masquerade as a policy change. The normalisation is
deliberately boring and lives in :func:`normalize_document` so that the value
recorded at curation time and the value computed here cannot drift.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx

from app.schemas import SourceFreshness, utcnow
from app.schemas.program import ProgramSource
from scripts.seed import CatalogRecord, default_catalog_dir, load_catalog

__all__ = [
    "SourceCheck",
    "content_hash",
    "main",
    "normalize_document",
    "render_report",
    "verify_catalog",
]

#: Polite, identifiable user agent. Agencies block unlabelled scrapers, and an
#: honest one is also the only courteous option for public-sector sites.
USER_AGENT: Final[str] = "SecondChance-source-verifier/1.0 (benefits navigator; non-commercial)"

DEFAULT_CONCURRENCY: Final[int] = 6
DEFAULT_TIMEOUT_S: Final[float] = 30.0

_DROPPED_BLOCKS: Final[re.Pattern[str]] = re.compile(
    r"<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_BLOCK_BOUNDARY: Final[re.Pattern[str]] = re.compile(
    r"<br\s*/?>|</(?:p|div|li|tr|td|th|h[1-6]|section|article|header|footer|nav)\s*>",
    re.IGNORECASE,
)
_ANY_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]*>")
_HORIZONTAL_SPACE: Final[re.Pattern[str]] = re.compile(r"[^\S\n]+")


def normalize_document(body: str) -> str:
    """Reduce an HTML document to the visible text a reader would see.

    Scripts, styles and markup are dropped, entities are unescaped, block ends
    become newlines and runs of horizontal whitespace collapse. Deterministic and
    pure: the same bytes always normalise to the same string, which is the whole
    point - :func:`content_hash` is only meaningful if curation and verification
    agree on what "the content" is.
    """
    text = _DROPPED_BLOCKS.sub(" ", body)
    text = _BLOCK_BOUNDARY.sub("\n", text)
    text = _ANY_TAG.sub(" ", text)
    text = html.unescape(text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    return "\n".join(line for line in (raw.strip() for raw in text.split("\n")) if line)


def content_hash(body: str) -> str:
    """SHA-256 hex digest of :func:`normalize_document` applied to ``body``."""
    return hashlib.sha256(normalize_document(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceCheck:
    """One source's verification result. Carries no personal data."""

    program_slug: str
    source_id: str
    url: str
    freshness: SourceFreshness
    age_days: int
    checked: bool = False
    status_code: int | None = None
    content_changed: bool | None = None
    error: str | None = None

    @property
    def is_dead(self) -> bool:
        return self.checked and (self.error is not None or self.status_code != 200)

    @property
    def verdict(self) -> str:
        if self.is_dead:
            return "DEAD"
        if self.content_changed:
            return "CHANGED"
        return self.freshness.value

    @property
    def needs_attention(self) -> bool:
        return self.is_dead or bool(self.content_changed) or self.freshness is SourceFreshness.STALE


@dataclass(slots=True)
class _Plan:
    """Flattened ``(program, source)`` work list."""

    pairs: list[tuple[str, ProgramSource]] = field(default_factory=list)

    @classmethod
    def build(cls, records: Iterable[CatalogRecord]) -> _Plan:
        plan = cls()
        for record in records:
            for source in record.program.sources:
                plan.pairs.append((record.program.slug, source))
        return plan


def _age_days(source: ProgramSource) -> int:
    return max((utcnow() - source.retrieved_at).days, 0)


def _offline_check(program_slug: str, source: ProgramSource) -> SourceCheck:
    return SourceCheck(
        program_slug=program_slug,
        source_id=source.id,
        url=source.url,
        freshness=source.current_freshness(),
        age_days=_age_days(source),
    )


async def _fetch_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    program_slug: str,
    source: ProgramSource,
) -> SourceCheck:
    """Re-fetch one source. Never raises: a failure is a reportable result."""
    base = _offline_check(program_slug, source)
    async with semaphore:
        try:
            response = await client.get(source.url)
        except httpx.HTTPError as exc:
            # Only the exception type is reported. Driver messages can embed the
            # full request, and this output is pasted into tickets and logs.
            return SourceCheck(
                program_slug=base.program_slug,
                source_id=base.source_id,
                url=base.url,
                freshness=base.freshness,
                age_days=base.age_days,
                checked=True,
                error=type(exc).__name__,
            )
    changed: bool | None = None
    if response.status_code == 200:
        changed = content_hash(response.text) != source.content_hash
    return SourceCheck(
        program_slug=base.program_slug,
        source_id=base.source_id,
        url=base.url,
        freshness=base.freshness,
        age_days=base.age_days,
        checked=True,
        status_code=response.status_code,
        content_changed=changed,
    )


async def verify_catalog(
    records: Sequence[CatalogRecord],
    *,
    offline: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[SourceCheck]:
    """Check every source in ``records``, at most ``concurrency`` at a time."""
    plan = _Plan.build(records)
    if offline:
        return [_offline_check(slug, source) for slug, source in plan.pairs]

    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout_s),
        limits=limits,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        return list(
            await asyncio.gather(
                *(_fetch_one(client, semaphore, slug, source) for slug, source in plan.pairs)
            )
        )


def render_report(results: Sequence[SourceCheck], *, offline: bool) -> str:
    """Human-readable freshness report, worst findings first."""
    order = {"DEAD": 0, "CHANGED": 1, "STALE": 2, "AGING": 3, "FRESH": 4}
    ranked = sorted(results, key=lambda check: (order[check.verdict], check.program_slug))
    width = max((len(check.source_id) for check in results), default=9)

    lines: list[str] = []
    mode = "offline (recorded dates only)" if offline else "live re-fetch"
    lines.append(f"source freshness report - {len(results)} sources - {mode}")
    lines.append("")
    for check in ranked:
        detail = ""
        if check.error is not None:
            detail = f"  ({check.error})"
        elif check.status_code is not None and check.status_code != 200:
            detail = f"  (HTTP {check.status_code})"
        lines.append(
            f"  {check.verdict:<8} {check.source_id:<{width}}  "
            f"{check.age_days:>4}d  {check.program_slug}{detail}"
        )

    tally = dict.fromkeys(order, 0)
    for check in results:
        tally[check.verdict] += 1
    lines.append("")
    lines.append("  " + "  ".join(f"{name}={count}" for name, count in tally.items()))

    dead = [check for check in results if check.is_dead]
    if dead:
        lines.append("")
        lines.append("dead links:")
        lines.extend(f"  {check.program_slug}/{check.source_id}  {check.url}" for check in dead)

    changed = [check for check in results if check.content_changed]
    if changed:
        lines.append("")
        lines.append("content changed since curation (re-read the page before trusting the rules):")
        lines.extend(f"  {check.program_slug}/{check.source_id}  {check.url}" for check in changed)
    return "\n".join(lines)


def _as_json(results: Sequence[SourceCheck]) -> str:
    return json.dumps(
        [
            {
                "program_slug": check.program_slug,
                "source_id": check.source_id,
                "url": check.url,
                "verdict": check.verdict,
                "freshness": check.freshness.value,
                "age_days": check.age_days,
                "status_code": check.status_code,
                "content_changed": check.content_changed,
                "error": check.error,
            }
            for check in results
        ],
        indent=2,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.verify_sources",
        description="Re-fetch catalog sources and report freshness. Operational script.",
    )
    parser.add_argument("--catalog", type=Path, default=None, help="catalog directory")
    parser.add_argument(
        "--offline", action="store_true", help="report recorded ages without any network calls"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when anything is dead, changed or stale",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.concurrency < 1:
        print("--concurrency must be at least 1", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    records = load_catalog(args.catalog or default_catalog_dir())
    results = asyncio.run(
        verify_catalog(
            records,
            offline=args.offline,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
        )
    )
    print(_as_json(results) if args.json else render_report(results, offline=args.offline))
    if args.strict and any(check.needs_attention for check in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
