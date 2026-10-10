"""Citation verification: checking generated text against the real catalog.

No model participates in anything here. Verification is identity and string
matching against stored :class:`~app.schemas.program.ProgramSource` rows,
because asking the generator to grade its own citations is not a check.

**Everything fails closed.** A citation is kept only when all four of these
hold:

1. the ``source_id`` exists in ``program_sources``;
2. it belongs to the program the citation attributes it to;
3. the ``url`` matches the stored source's URL;
4. the quoted ``excerpt`` is genuinely present in the stored excerpt - either
   contained in it after normalisation, or a close normalised match of it.

A fifth check rejects a source that is ``STALE`` (over a year old), because a
quotation from a page nobody has re-read in a year cannot support a claim about
today's requirements. Anything that cannot be resolved - an id we do not hold,
a source whose program we cannot confirm - is **dropped and recorded**, never
passed through. ``CitationVerificationReport.rejected`` is the audit trail and
``ok`` is false whenever it is non-empty.

**The database is the authority.** When a session is available, the stored row
decides, and ``known_sources`` is only a fallback for callers that have no
session (the in-memory repository, and the unit tests). Letting a caller-
supplied source object override a stored one would make verification
spoofable by whatever produced the sources.

The scrubbing helpers (:func:`strip_unsupported_claims`,
:func:`detect_guarantee_phrasing`) enforce the other half of the promise: our
own prose may not claim an outcome. "You qualify", "you are eligible",
"guaranteed" and "approved" are determinations only the administering agency
can make, so a sentence containing one is flagged for removal regardless of how
well cited it is.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db import models
from app.schemas import (
    ChatResponse,
    Citation,
    CitationVerificationReport,
    ProgramRepository,
    ProgramSource,
    RejectedCitation,
    utcnow,
)

__all__ = [
    "GUARANTEE_PATTERNS",
    "CatalogSource",
    "CitationReport",
    "GuaranteeFlag",
    "RejectionReason",
    "ScrubbedText",
    "SourceCitationVerifier",
    "detect_guarantee_phrasing",
    "excerpt_matches",
    "has_guarantee_phrasing",
    "load_catalog_sources",
    "normalise_excerpt",
    "split_sentences",
    "strip_unsupported_claims",
    "verify_against",
    "verify_citations",
    "verify_claims",
]

_log = get_logger(__name__)

#: The task brief's name for the report. The wire type is the contract's
#: :class:`~app.schemas.assistant.CitationVerificationReport`; this alias exists
#: so both names resolve to the same object and no translation layer appears.
CitationReport = CitationVerificationReport

#: Shortest quotation we will certify. A three-word excerpt is contained in
#: almost any document, so "it appears in the source" would stop being evidence
#: of anything. Rejected as ``excerpt_not_found`` with a reason that says so.
MIN_EXCERPT_CHARS: Final[int] = 12

#: How much of a quoted excerpt must appear as one unbroken run inside the
#: stored excerpt for a non-exact quotation to pass.
#:
#: Set high on purpose. Truncating a quotation, or re-wrapping it, leaves one
#: long common run and passes; changing a number or a word in the middle breaks
#: the run roughly in half and fails. A ratio low enough to tolerate the latter
#: would certify a quotation the source does not actually make.
EXCERPT_MATCH_RATIO: Final[float] = 0.90

#: Characters publishers and models swap for each other. Folded before any
#: comparison so a smart quote or an en dash is never the reason a real
#: citation is rejected.
_PUNCTUATION_FOLD: Final[dict[int, str]] = {
    0x2018: "'",  # left single quotation mark
    0x2019: "'",  # right single quotation mark
    0x201A: "'",  # single low-9 quotation mark
    0x201C: '"',  # left double quotation mark
    0x201D: '"',  # right double quotation mark
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2212: "-",  # minus sign
    0x00A0: " ",  # no-break space
    0x2026: "...",  # horizontal ellipsis
}

#: Eligibility-guarantee phrasing. Each pattern is a determination this product
#: may not make: the rule engine reports ``POTENTIAL_MATCH`` /
#: ``MORE_INFO_NEEDED`` / ``UNLIKELY_MATCH``, and "eligible", "ineligible",
#: "qualify", "guaranteed" and "approved" are the agency's words, not ours.
#:
#: Deliberately matched on the bare words as well as the full phrases. A
#: sentence reading "most students are eligible" is just as much of a promise
#: as "you are eligible", and the narrow patterns exist only so the report can
#: name the phrase it objected to.
GUARANTEE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\byou(?:'ll|'re|\s+(?:are|will|do|would|have))?\s+(?:definitely\s+)?qualif", re.I),
    re.compile(r"\byou(?:'re|\s+are)\s+(?:not\s+|definitely\s+)?(?:in)?eligible\b", re.I),
    re.compile(r"\byou(?:'ll|\s+will)\s+(?:be\s+)?(?:approved|accepted|denied|rejected)\b", re.I),
    re.compile(r"\byou\s+(?:are|have\s+been)\s+(?:pre-?)?approved\b", re.I),
    re.compile(r"\bguarantee(?:d|s|ing)?\b", re.I),
    re.compile(r"\b(?:pre-?)?approved\b", re.I),
    re.compile(r"\b(?:in)?eligible\b", re.I),
    re.compile(r"\bqualif(?:y|ies|ied)\b", re.I),
)

#: Inline citation markers a generator might leave in prose: ``[nj-snap-2026]``
#: or ``(source: nj-snap-2026)``.
_SOURCE_MARKER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\[([a-z0-9]+(?:-[a-z0-9]+)*)\]"),
    re.compile(r"\(\s*sources?\s*:\s*([a-z0-9]+(?:-[a-z0-9]+)*)\s*\)", re.I),
)

#: A bare URL in prose. Trailing punctuation is trimmed by the caller.
_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://[^\s\]\)>\"']+", re.I)

#: Sentence boundary: terminal punctuation followed by whitespace, or a newline.
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# normalisation and matching
# ---------------------------------------------------------------------------
def normalise_excerpt(value: str) -> str:
    """Fold a quotation to the form comparisons are made in.

    Unicode-normalised, punctuation variants folded, whitespace collapsed,
    lowercased. Every difference removed here is one a publisher's HTML or a
    model's re-typing can introduce without changing what the source says;
    nothing that changes meaning is removed.
    """
    folded = unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_FOLD)
    return " ".join(folded.lower().split())


def excerpt_matches(quoted: str, stored: str) -> bool:
    """Whether ``quoted`` is genuinely a quotation of ``stored``.

    Containment after normalisation is the common case. Otherwise the longest
    unbroken common run must cover :data:`EXCERPT_MATCH_RATIO` of the
    quotation, which tolerates truncation and re-wrapping but not an altered
    number or substituted word.
    """
    needle = normalise_excerpt(quoted)
    haystack = normalise_excerpt(stored)
    if len(needle) < MIN_EXCERPT_CHARS or not haystack:
        return False
    if needle in haystack:
        return True
    matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
    block = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return block.size / len(needle) >= EXCERPT_MATCH_RATIO


def _normalise_url(value: str) -> str:
    """Compare URLs by what identifies the document, not by how it was typed.

    Scheme and host are case-insensitive and a trailing slash is meaningless,
    so both are normalised. The query string is kept: on several state benefits
    sites it is the only thing that distinguishes two pages. The fragment is
    dropped - it addresses a position within one document, not a document.
    """
    parts = urlsplit(value.strip())
    path = parts.path.rstrip("/")
    normalised = f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}"
    return f"{normalised}?{parts.query}" if parts.query else normalised


# ---------------------------------------------------------------------------
# the sources a check runs against
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CatalogSource:
    """A stored source plus the program it belongs to.

    ``ProgramSource`` deliberately carries no ``program_id`` - it is nested
    inside its program everywhere it is serialised - but attribution is one of
    the things being verified, so the id travels alongside it here.
    ``program_id=None`` means "we do not know which program this belongs to",
    which makes an attributed citation unverifiable rather than valid.
    """

    source: ProgramSource
    program_id: uuid.UUID | None = None

    @property
    def id(self) -> str:
        return self.source.id


def _as_catalog_sources(
    sources: Iterable[CatalogSource | ProgramSource],
) -> dict[str, CatalogSource]:
    """Index sources by id, keeping the first occurrence of a repeated id."""
    indexed: dict[str, CatalogSource] = {}
    for item in sources:
        entry = item if isinstance(item, CatalogSource) else CatalogSource(source=item)
        indexed.setdefault(entry.id, entry)
    return indexed


async def load_catalog_sources(
    session: AsyncSession, source_ids: Sequence[str]
) -> dict[str, CatalogSource]:
    """One query: the stored rows for the ids a response cited.

    One ``IN`` query regardless of how many citations the answer carried, and
    the only lookup the verifier makes - a per-citation query here would turn
    an eight-citation answer into eight round trips on a user-facing path.
    """
    wanted = list(dict.fromkeys(source_ids))
    if not wanted:
        return {}
    statement = select(models.ProgramSource).where(models.ProgramSource.id.in_(wanted))
    rows = (await session.scalars(statement)).all()
    return {
        row.id: CatalogSource(
            source=ProgramSource(
                id=row.id,
                url=row.url,
                title=row.title,
                publisher=row.publisher,
                excerpt=row.excerpt,
                retrieved_at=row.retrieved_at,
                content_hash=row.content_hash,
                freshness=row.freshness,
            ),
            program_id=row.program_id,
        )
        for row in rows
    }


# ---------------------------------------------------------------------------
# the deterministic core
# ---------------------------------------------------------------------------
#: The rejection reasons ``RejectedCitation`` accepts, named so the helper
#: below is type-checked rather than cast.
RejectionReason = Literal[
    "unknown_source",
    "excerpt_not_found",
    "url_mismatch",
    "stale_source",
    "program_mismatch",
]


def _reject(citation: Citation, reason: RejectionReason, detail: str) -> RejectedCitation:
    return RejectedCitation(source_id=citation.source_id, reason=reason, detail=detail[:300])


def _check(
    citation: Citation, indexed: Mapping[str, CatalogSource], *, now: datetime
) -> RejectedCitation | None:
    """Run the five checks in order. ``None`` means the citation is good."""
    entry = indexed.get(citation.source_id)
    if entry is None:
        return _reject(citation, "unknown_source", "no source with that id exists in the catalog")
    if citation.program_id is not None:
        if entry.program_id is None:
            return _reject(
                citation,
                "program_mismatch",
                "the stored source could not be attributed to a program, so the "
                "claimed program could not be confirmed",
            )
        if entry.program_id != citation.program_id:
            return _reject(
                citation, "program_mismatch", "the source belongs to a different program"
            )
    if _normalise_url(citation.url) != _normalise_url(entry.source.url):
        return _reject(citation, "url_mismatch", "the URL does not match the stored source")
    if not excerpt_matches(citation.excerpt, entry.source.excerpt):
        detail = (
            f"the quoted excerpt is shorter than {MIN_EXCERPT_CHARS} characters"
            if len(normalise_excerpt(citation.excerpt)) < MIN_EXCERPT_CHARS
            else "the quoted excerpt does not appear in the stored source text"
        )
        return _reject(citation, "excerpt_not_found", detail)
    if not entry.source.supports_eligibility_claim(now):
        return _reject(
            citation,
            "stale_source",
            "the source has not been re-checked within the last year",
        )
    return None


def verify_against(
    citations: Sequence[Citation],
    sources: Iterable[CatalogSource | ProgramSource],
    *,
    now: datetime | None = None,
) -> CitationVerificationReport:
    """Verify citations against a set of stored sources. Pure and synchronous.

    Identical citations are collapsed, so a response that cites the same
    passage twice does not inflate ``unsupported_claim_rate`` - or, if it is
    bad, get rejected twice.
    """
    moment = now or utcnow()
    indexed = _as_catalog_sources(sources)
    verified: list[Citation] = []
    rejected: list[RejectedCitation] = []
    seen: set[tuple[str, str, str]] = set()

    for citation in citations:
        key = (
            citation.source_id,
            _normalise_url(citation.url),
            normalise_excerpt(citation.excerpt),
        )
        if key in seen:
            continue
        seen.add(key)
        failure = _check(citation, indexed, now=moment)
        if failure is None:
            verified.append(citation)
        else:
            rejected.append(failure)

    checked = [entry.source for entry in sorted(indexed.values(), key=lambda item: item.id)]
    report = CitationVerificationReport(
        ok=not rejected,
        verified=verified,
        rejected=rejected,
        unsupported_claims=[],
        checked_sources=checked,
    )
    _log.debug(
        "citations.verified",
        verified=len(report.verified),
        rejected=len(report.rejected),
        checked=len(checked),
    )
    return report


def verify_claims(
    response: ChatResponse,
    sources: Iterable[CatalogSource | ProgramSource],
    *,
    now: datetime | None = None,
) -> CitationVerificationReport:
    """Verify a whole assistant response: its citations and its claims.

    A :class:`~app.schemas.assistant.FactStatement` survives only if *every*
    source it names verified. Partial support is treated as no support, because
    a sentence built from two sources where one is fabricated is not
    three-quarters true.

    Guarantee phrasing is reported as an unsupported claim wherever it appears -
    in a fact or in the answer prose - because no source can support a promise
    about an outcome.
    """
    base = verify_against(response.citations, sources, now=now)
    good_ids = {citation.source_id for citation in base.verified}

    unsupported: dict[str, None] = {}
    for fact in response.facts:
        missing = [source_id for source_id in fact.source_ids if source_id not in good_ids]
        if missing:
            unsupported.setdefault(fact.text, None)
    for text in (response.answer, *(fact.text for fact in response.facts)):
        for flag in detect_guarantee_phrasing(text):
            unsupported.setdefault(flag.sentence, None)

    claims = list(unsupported)
    return CitationVerificationReport(
        ok=not base.rejected and not claims,
        verified=base.verified,
        rejected=base.rejected,
        unsupported_claims=claims,
        checked_sources=base.checked_sources,
    )


async def verify_citations(
    session: AsyncSession | None,
    claims: Sequence[Citation] | ChatResponse,
    *,
    known_sources: Sequence[ProgramSource] | None = None,
    now: datetime | None = None,
) -> CitationVerificationReport:
    """Verify citations, or a whole response, against the stored catalog.

    ``claims`` is either a list of citations or a
    :class:`~app.schemas.assistant.ChatResponse`, in which case its facts and
    answer prose are checked too.

    ``session`` may be ``None``, and then only ``known_sources`` are consulted
    and anything else is unresolvable and therefore rejected. That is the
    fail-closed path the in-memory repository and the unit tests use; when a
    session is present the stored row is the authority and ``known_sources``
    is ignored, so a caller cannot supply the source that vouches for its own
    citation.
    """
    citations = claims.citations if isinstance(claims, ChatResponse) else list(claims)
    if session is not None:
        indexed: Iterable[CatalogSource | ProgramSource] = list(
            (await load_catalog_sources(session, [item.source_id for item in citations])).values()
        )
    else:
        indexed = list(known_sources or ())
    if isinstance(claims, ChatResponse):
        return verify_claims(claims, indexed, now=now)
    return verify_against(citations, indexed, now=now)


# ---------------------------------------------------------------------------
# scrubbing model prose
# ---------------------------------------------------------------------------
class GuaranteeFlag(BaseModel):
    """A sentence that promises an outcome, and the phrasing that gave it away."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sentence: str = Field(min_length=1, max_length=4000)
    phrases: list[str] = Field(min_length=1, description="The matched phrases, lowercased.")


class ScrubbedText(BaseModel):
    """The result of :func:`strip_unsupported_claims`.

    ``text`` may be empty: if every sentence failed, removing them all is the
    correct outcome and the caller must handle having nothing to say rather
    than being handed something unsafe.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(description="The kept sentences, joined with single spaces.")
    removed: list[str] = Field(default_factory=list, description="Sentences that were dropped.")
    guarantee_flags: list[GuaranteeFlag] = Field(default_factory=list)
    removed_source_ids: list[str] = Field(
        default_factory=list, description="Cited ids that were not in the allowed set."
    )
    removed_urls: list[str] = Field(
        default_factory=list, description="Links that no allowed source published."
    )

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences on terminal punctuation and newlines.

    Good enough on purpose: the consumers are a safety filter and a flag
    report, and a filter that errs towards treating a run-on as one sentence
    removes slightly more text than strictly necessary, which is the right
    direction for this failure to point.
    """
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]


def detect_guarantee_phrasing(text: str) -> list[GuaranteeFlag]:
    """Sentences that claim an eligibility outcome, flagged for removal."""
    flags: list[GuaranteeFlag] = []
    for sentence in split_sentences(text):
        phrases: dict[str, None] = {}
        for pattern in GUARANTEE_PATTERNS:
            for match in pattern.finditer(sentence):
                phrases.setdefault(" ".join(match.group(0).lower().split()), None)
        if phrases:
            flags.append(GuaranteeFlag(sentence=sentence[:4000], phrases=list(phrases)))
    return flags


def has_guarantee_phrasing(text: str) -> bool:
    """Whether any sentence in ``text`` claims an eligibility outcome."""
    return any(pattern.search(text) for pattern in GUARANTEE_PATTERNS)


def _allowed_ids_and_urls(
    allowed_sources: Iterable[ProgramSource | CatalogSource | str],
) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    urls: set[str] = set()
    for item in allowed_sources:
        if isinstance(item, str):
            ids.add(item)
            continue
        source = item.source if isinstance(item, CatalogSource) else item
        ids.add(source.id)
        urls.add(_normalise_url(source.url))
    return ids, urls


def strip_unsupported_claims(
    text: str,
    allowed_sources: Iterable[ProgramSource | CatalogSource | str],
    *,
    allowed_urls: Iterable[str] = (),
) -> ScrubbedText:
    """Remove every sentence of model prose we cannot stand behind.

    A sentence is dropped when it promises an outcome, when it cites a source
    id that is not in ``allowed_sources``, or when it links to a URL that no
    allowed source published. ``allowed_urls`` extends the link allowlist with
    URLs the catalog holds elsewhere - a program's official and application
    links, which are legitimate to offer but are not source documents.

    Sentence-level rather than phrase-level on purpose: deleting "not" or a
    citation marker from the middle of a sentence leaves fluent text that now
    says something the author never wrote, which is worse than saying less.
    """
    ids, urls = _allowed_ids_and_urls(allowed_sources)
    urls |= {_normalise_url(url) for url in allowed_urls}

    kept: list[str] = []
    removed: list[str] = []
    flags: list[GuaranteeFlag] = []
    bad_ids: dict[str, None] = {}
    bad_urls: dict[str, None] = {}

    for sentence in split_sentences(text):
        sentence_flags = detect_guarantee_phrasing(sentence)
        cited = {
            match.group(1).lower()
            for pattern in _SOURCE_MARKER_PATTERNS
            for match in pattern.finditer(sentence)
        }
        unknown_ids = sorted(cited - ids)
        linked = [match.group(0).rstrip(".,;:!?") for match in _URL_PATTERN.finditer(sentence)]
        unknown_urls = [url for url in linked if _normalise_url(url) not in urls]

        if sentence_flags or unknown_ids or unknown_urls:
            removed.append(sentence)
            flags.extend(sentence_flags)
            for source_id in unknown_ids:
                bad_ids.setdefault(source_id, None)
            for url in unknown_urls:
                bad_urls.setdefault(url, None)
        else:
            kept.append(sentence)

    if removed:
        # Counts only: the prose itself is a user's answer and is never logged.
        _log.info(
            "citations.scrubbed",
            removed=len(removed),
            guarantees=len(flags),
            unknown_source_ids=len(bad_ids),
            unknown_urls=len(bad_urls),
        )
    return ScrubbedText(
        text=" ".join(kept),
        removed=removed,
        guarantee_flags=flags,
        removed_source_ids=list(bad_ids),
        removed_urls=list(bad_urls),
    )


# ---------------------------------------------------------------------------
# the contract implementation
# ---------------------------------------------------------------------------
class SourceCitationVerifier:
    """The :class:`app.schemas.contracts.CitationVerifier` implementation.

    Three ways of resolving a cited id, in strict order of authority:

    1. **The session**, when the repository carries one. One ``IN`` query
       returns the stored rows with their ``program_id``, and they decide.
    2. **The repository**, otherwise. Citations are grouped by the program they
       name and resolved with one
       :meth:`~app.schemas.contracts.ProgramRepository.get_sources` call per
       program - which also settles attribution, since that call filters by
       ``program_id``.
    3. **``known_sources``**, last. A caller-supplied source can only ever fill
       a gap, never override a stored row, because a generator that supplied
       both the citation and the source vouching for it would be grading
       itself.

    Anything still unresolved after all three is rejected.
    """

    __slots__ = ("_repository", "_session")

    def __init__(
        self, repository: ProgramRepository, *, session: AsyncSession | None = None
    ) -> None:
        self._repository = repository
        # ``SqlProgramRepository`` carries its session; an in-memory one does
        # not. Read off the repository rather than required, so the contract's
        # one-argument construction keeps working.
        resolved = session if session is not None else getattr(repository, "session", None)
        self._session = resolved if isinstance(resolved, AsyncSession) else None

    async def _sources_for(
        self, citations: Sequence[Citation], known_sources: Sequence[ProgramSource] | None
    ) -> list[CatalogSource | ProgramSource]:
        if self._session is not None:
            indexed = await load_catalog_sources(
                self._session, [citation.source_id for citation in citations]
            )
            return list(indexed.values())

        resolved: dict[str, CatalogSource | ProgramSource] = {}
        attributed = {
            citation.program_id for citation in citations if citation.program_id is not None
        }
        for program_id in sorted(attributed, key=str):
            wanted = [
                citation.source_id for citation in citations if citation.program_id == program_id
            ]
            for source in await self._repository.get_sources(program_id, source_ids=wanted):
                resolved.setdefault(source.id, CatalogSource(source=source, program_id=program_id))
        for source in known_sources or ():
            resolved.setdefault(source.id, source)
        return list(resolved.values())

    async def verify(
        self,
        citations: Sequence[Citation],
        *,
        known_sources: Sequence[ProgramSource] | None = None,
    ) -> CitationVerificationReport:
        """Verify that each citation names a real source and quotes it accurately."""
        sources = await self._sources_for(citations, known_sources)
        return verify_against(citations, sources)

    async def verify_response(
        self,
        response: ChatResponse,
        *,
        known_sources: Sequence[ProgramSource] | None = None,
    ) -> CitationVerificationReport:
        """Verify a whole response, including that every fact is supported."""
        sources = await self._sources_for(response.citations, known_sources)
        return verify_claims(response, sources)
