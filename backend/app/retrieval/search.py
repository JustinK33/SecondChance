"""Candidate retrieval: which programs are worth evaluating for this profile.

Retrieval decides **relevance**, never eligibility. It is deliberately
generous: a candidate the rule engine later rules out costs one evaluation,
whereas a candidate that is never retrieved is help the user never hears about.
Three consequences run through this module.

**A hard filter is only ever a statement about the program, never about the
person.** Geography and campus scope are properties the catalog asserts, so
they filter. Anything that depends on a profile field we might simply not know
- income, household size, student status when it was never stated - is scored,
not filtered, because "unknown" must never behave like "no".

**The score is transparent.** It is a weighted sum of three named components,
each in ``[0, 1]``:

.. code-block:: text

    total = 0.50 * text_match        # normalised ts_rank over search_vector
          + 0.35 * need_overlap      # stated needs vs category and keywords
          + 0.15 * source_freshness  # freshness of the best source x verification

and :class:`RetrievalScore` carries every component alongside the total, so the
UI can explain an ordering and the benchmarks can regress on one part of it
without reverse-engineering the whole. ``search_programs`` returns the plain
:class:`~app.schemas.program.RetrievedProgram` list the rest of the system is
typed against; ``search_scored_programs`` is the same work with the breakdown
attached.

**Unverified records are structurally excluded from eligibility claims.** An
``UNVERIFIED`` or ``NEEDS_REVIEW`` program is still returned - it is a useful
lead - but it comes back with ``ruleset=None``, so
:meth:`app.eligibility.engine.ProgramRuleset.from_candidate` yields ``None`` and
the engine reports "cannot assess" rather than a status. The flag is therefore
not advisory: there is no code path that turns an unverified record into a
verdict. ``AssistanceProgram.supports_eligibility_claims()`` is the predicate.

Query counts are bounded and independent of the number of results: one search
query for the ranked ids, one batch load of the programs (plus its ``selectin``
load of sources) and one batch load of the in-force rule versions. Nothing in
here loops a per-row query.
"""

from __future__ import annotations

import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import ColumnElement, Text, func, literal, or_, select, text
from sqlalchemy.dialects.postgresql import array as pg_array
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db import models
from app.db.repositories.programs import SqlProgramRepository
from app.schemas import (
    NEED_TO_PROGRAM_CATEGORIES,
    AssistanceProgram,
    ConfidenceScore,
    NeedCategory,
    ProfileFlag,
    ProgramCategory,
    ProgramDetail,
    ProgramSource,
    RetrievedProgram,
    RuleSet,
    SourceFreshness,
    StudentStatus,
    UserProfile,
    VerificationStatus,
)

__all__ = [
    "MAX_CANDIDATE_POOL",
    "NEED_QUERY_TERMS",
    "WEIGHT_NEED_OVERLAP",
    "WEIGHT_SOURCE_FRESHNESS",
    "WEIGHT_TEXT_MATCH",
    "HybridRetriever",
    "RetrievalScore",
    "ScoredProgram",
    "get_program_detail",
    "score_program",
    "search_programs",
    "search_scored_programs",
]

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# tuning constants
# ---------------------------------------------------------------------------

#: The text search configuration the generated ``search_vector`` was built
#: with (see ``app.db.models.program.SEARCH_VECTOR_EXPRESSION``). Parsing a
#: query with a different configuration would stem differently and silently
#: fail to match.
_TS_CONFIG: Final[str] = "english"

#: Score weights. They sum to 1.0, which is what keeps ``total`` in ``[0, 1]``
#: without a clamp doing the work.
WEIGHT_TEXT_MATCH: Final[float] = 0.50
WEIGHT_NEED_OVERLAP: Final[float] = 0.35
WEIGHT_SOURCE_FRESHNESS: Final[float] = 0.15

#: ``ts_rank`` value that maps to a ``text_match`` of 0.5, used by the
#: saturating transform ``rank / (rank + TS_RANK_HALF)``.
#:
#: Postgres' default ts_rank weights give a single one-term hit roughly 0.61 in
#: the ``A`` band (the official name), 0.24 in ``B`` (summary and keywords) and
#: 0.12 in ``C`` (description), and sums contributions across matched terms. A
#: linear clamp at some ceiling would flatten every name match to 1.0 and throw
#: away the ordering between them, so the transform is monotone and bounded
#: instead: nothing ever saturates and no two different ranks collide.
_TS_RANK_HALF: Final[float] = 0.25

#: Relative weight of the two halves of ``need_overlap``. The category mapping
#: is the stronger signal because it is curated
#: (``NEED_TO_PROGRAM_CATEGORIES``); keyword overlap is a heuristic over
#: author-written keywords and only breaks ties inside a category.
_NEED_CATEGORY_SHARE: Final[float] = 0.70
_NEED_KEYWORD_SHARE: Final[float] = 0.30

#: Score for a program whose category is the *primary* mapping for a stated
#: need, versus a secondary one. ``NEED_TO_PROGRAM_CATEGORIES`` lists the
#: primary category first.
_PRIMARY_CATEGORY_SCORE: Final[float] = 1.0
_SECONDARY_CATEGORY_SCORE: Final[float] = 0.70

#: Freshness of the best source a program has. A program with no source at all
#: scores zero here rather than being filtered out: it is still a lead, it just
#: has nothing to support a claim with.
_FRESHNESS_WEIGHTS: Final[dict[SourceFreshness, float]] = {
    SourceFreshness.FRESH: 1.0,
    SourceFreshness.AGING: 0.60,
    SourceFreshness.STALE: 0.10,
}

#: How much human verification is worth inside the ``source_freshness``
#: component. Folded in here rather than added as a fourth term because
#: freshness and verification answer the same question - how current and how
#: checked is this record - and because an unverified lead must rank below a
#: verified program of equal relevance without ever being hidden.
_VERIFICATION_FACTORS: Final[dict[VerificationStatus, float]] = {
    VerificationStatus.VERIFIED: 1.0,
    VerificationStatus.NEEDS_REVIEW: 0.60,
    VerificationStatus.UNVERIFIED: 0.40,
}

#: Candidate pool sizing. The pool is ranked by ``ts_rank`` in Postgres, then
#: rescored in Python with the need and freshness components before being cut
#: to ``limit``; it has to be wider than ``limit`` or the components that
#: Postgres cannot see could never promote anything.
_POOL_MULTIPLIER: Final[int] = 4
_MIN_CANDIDATE_POOL: Final[int] = 60
MAX_CANDIDATE_POOL: Final[int] = 200

#: Upper bound on returned candidates, matching ``Page.limit``.
_MAX_LIMIT: Final[int] = 200

#: Query terms worth adding when the user gave no search text of their own but
#: did state a need. These expand a :class:`NeedCategory` into words that
#: actually appear in catalog keywords; they are a retrieval aid only and are
#: never shown to a user or read by the rule engine.
NEED_QUERY_TERMS: Final[dict[NeedCategory, tuple[str, ...]]] = {
    NeedCategory.FOOD: ("food", "groceries", "meals", "nutrition"),
    NeedCategory.HEALTHCARE: ("health", "insurance", "medical", "prescription"),
    NeedCategory.UTILITIES: ("utility", "electric", "heating", "water"),
    NeedCategory.HOUSING: ("rent", "housing", "eviction", "shelter"),
    NeedCategory.CHILDCARE: ("childcare", "daycare", "children"),
    NeedCategory.TRANSPORTATION: ("transportation", "transit", "bus", "fare"),
    NeedCategory.EDUCATION_COSTS: ("tuition", "textbooks", "education", "student"),
    NeedCategory.INCOME_SUPPORT: ("income", "cash", "credit", "unemployment"),
    NeedCategory.INTERNET_PHONE: ("internet", "phone", "broadband"),
    NeedCategory.LEGAL_HELP: ("legal", "lawyer", "court"),
    NeedCategory.MENTAL_HEALTH: ("mental", "counseling", "therapy"),
    NeedCategory.EMERGENCY_EXPENSE: ("emergency", "grant", "crisis"),
    NeedCategory.OTHER: (),
    NeedCategory.UNKNOWN: (),
}

#: Plain-language label for a need, used only in ``retrieval_reasons``.
_NEED_LABELS: Final[dict[NeedCategory, str]] = {
    NeedCategory.FOOD: "food and groceries",
    NeedCategory.HEALTHCARE: "health care",
    NeedCategory.UTILITIES: "utility bills",
    NeedCategory.HOUSING: "housing or rent",
    NeedCategory.CHILDCARE: "child care",
    NeedCategory.TRANSPORTATION: "getting around",
    NeedCategory.EDUCATION_COSTS: "school costs",
    NeedCategory.INCOME_SUPPORT: "income support",
    NeedCategory.INTERNET_PHONE: "internet or phone",
    NeedCategory.LEGAL_HELP: "legal help",
    NeedCategory.MENTAL_HEALTH: "mental health",
    NeedCategory.EMERGENCY_EXPENSE: "an emergency expense",
    NeedCategory.OTHER: "something else",
    NeedCategory.UNKNOWN: "help",
}

#: Dropped from a query before it is handed to ``plainto_tsquery``.
#:
#: Postgres would drop these itself, but a query made up of nothing else would
#: then produce an *empty* tsquery, and ``search_vector @@ ''::tsquery`` matches
#: no row at all. Filtering here means "the and of" degrades to "no text
#: signal" rather than to "no results".
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "about", "all", "am", "an", "and", "any", "are", "as", "at", "be", "been", "but",
        "by", "can", "do", "does", "for", "from", "get", "got", "had", "has", "have", "help",
        "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "need", "no",
        "not", "of", "on", "or", "our", "out", "so", "some", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "to", "up", "was", "we", "what", "when",
        "where", "which", "who", "will", "with", "would", "you", "your",
    }
)  # fmt: skip

#: Longest query we will tokenise. A query longer than this is truncated rather
#: than rejected: retrieval is not a validation boundary.
_MAX_QUERY_CHARS: Final[int] = 400
#: Most tokens we will OR into one tsquery.
_MAX_QUERY_TOKENS: Final[int] = 12
#: Shortest token worth searching for. One-character tokens match everything.
_MIN_TOKEN_CHARS: Final[int] = 2


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
class RetrievalScore(BaseModel):
    """The ranking score for one candidate, with every component kept.

    Kept rather than collapsed because an ordering nobody can explain is an
    ordering nobody can fix: the UI shows why a program surfaced, and the
    retrieval benchmark asserts on one component at a time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: ConfidenceScore = Field(description="Weighted sum of the three components below.")
    text_match: ConfidenceScore = Field(description="Normalised ts_rank over search_vector.")
    need_overlap: ConfidenceScore = Field(
        description="Stated needs against the program's category and keywords."
    )
    source_freshness: ConfidenceScore = Field(
        description="freshness x verification_factor for the program's best source."
    )
    ts_rank: float = Field(ge=0.0, description="Raw Postgres ts_rank, before normalisation.")
    freshness: ConfidenceScore = Field(description="Freshness weight of the best source alone.")
    verification_factor: ConfidenceScore = Field(
        description="How far a human has checked the record."
    )

    def explain(self) -> str:
        """One-line breakdown. Diagnostics only; never shown to a user."""
        return (
            f"total={self.total:.4f} "
            f"= {WEIGHT_TEXT_MATCH:.2f}*text {self.text_match:.4f} "
            f"+ {WEIGHT_NEED_OVERLAP:.2f}*need {self.need_overlap:.4f} "
            f"+ {WEIGHT_SOURCE_FRESHNESS:.2f}*fresh {self.source_freshness:.4f} "
            f"(ts_rank={self.ts_rank:.4f}, verification={self.verification_factor:.2f})"
        )


@dataclass(frozen=True, slots=True)
class ScoredProgram:
    """A retrieval candidate together with the breakdown that ranked it."""

    candidate: RetrievedProgram
    score: RetrievalScore


# ---------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------
def _fold(value: str) -> str:
    """Lowercase, strip accents, and reduce everything else to spaces."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    return "".join(char if char.isalnum() else " " for char in ascii_only)


def _tokenise(value: str) -> list[str]:
    """Fold, split and drop stopwords, preserving first-seen order."""
    seen: dict[str, None] = {}
    for token in _fold(value[:_MAX_QUERY_CHARS]).split():
        if len(token) < _MIN_TOKEN_CHARS or token in _STOPWORDS:
            continue
        seen.setdefault(token, None)
    return list(seen)[:_MAX_QUERY_TOKENS]


def _stem_variants(token: str) -> frozenset[str]:
    """Crude singular/plural forms of one word, for ``matched_terms`` reporting.

    Postgres does the real stemming inside ``to_tsvector``; this exists so that
    a query for "groceries" can be *reported* as having matched a keyword
    spelled "grocery" without reimplementing Snowball in Python. It is only
    ever used to explain a match, never to decide one - the ``@@`` test in
    Postgres does that.
    """
    forms = {token}
    if token.endswith("ies") and len(token) > 4:
        forms |= {token[:-3] + "y", token[:-3], token[:-2]}
    elif token.endswith("es") and len(token) > 3:
        forms |= {token[:-2], token[:-1]}
    elif token.endswith("s") and len(token) > 3:
        forms.add(token[:-1])
    return frozenset(forms)


def _program_text(program: AssistanceProgram) -> str:
    """Everything about a program that a query token may legitimately match."""
    return _fold(
        " ".join(
            [
                program.official_name,
                program.plain_language_summary,
                program.description,
                program.category.value,
                *program.keywords,
                *program.who_it_helps,
            ]
        )
    )


def _matched_terms(tokens: Sequence[str], program: AssistanceProgram) -> list[str]:
    """Which query tokens plausibly matched, for retrieval debugging."""
    haystack = _program_text(program)
    forms: set[str] = set()
    for word in haystack.split():
        forms |= _stem_variants(word)
    return [token for token in tokens if token in haystack or _stem_variants(token) & forms]


# ---------------------------------------------------------------------------
# profile-derived filters
# ---------------------------------------------------------------------------
def _is_student(profile: UserProfile) -> bool | None:
    """Tri-state: ``True`` a student, ``False`` explicitly not, ``None`` unstated.

    ``None`` is not ``False``. A profile that never mentioned school must still
    see campus programs, because the alternative is hiding emergency aid from a
    student who only told us about their rent.
    """
    if profile.flag(ProfileFlag.IS_RUTGERS_AFFILIATED) is True:
        return True
    status = profile.student_status
    if status is None or status is StudentStatus.UNKNOWN:
        return None
    if status is StudentStatus.NOT_A_STUDENT:
        return False
    if status is StudentStatus.OTHER:
        return None
    return True


def _county_variants(county: str) -> list[str]:
    """Spellings of a county name a catalog record might plausibly use.

    ``CountyName`` strips a trailing "County" on the way in, and the curated
    catalog writes "Middlesex", but a JSONB ``?|`` test is exact, so the
    variants are enumerated rather than hoped for.
    """
    bases = [county, f"{county} County"]
    variants: dict[str, None] = {}
    for base in bases:
        for spelling in (base, base.title(), base.upper(), base.lower()):
            variants.setdefault(spelling, None)
    return list(variants)


def _jsonb_array_length(column: ColumnElement[Any]) -> ColumnElement[int]:
    return func.jsonb_array_length(func.coalesce(column, text("'[]'::jsonb")))


def _scope_filters(profile: UserProfile) -> list[ColumnElement[bool]]:
    """Hard filters: claims the catalog makes about where a program operates.

    Every one of these is a property of the *program*. A filter that depended
    on a profile field we might not know would turn missing information into a
    silent exclusion, which is the one thing retrieval may never do.
    """
    table = models.AssistanceProgram
    filters: list[ColumnElement[bool]] = [table.deleted_at.is_(None)]

    if profile.state:
        # A nationwide program is in scope for every state.
        filters.append(
            or_(
                table.geo_scope["nationwide"].as_boolean().is_(True),
                table.geo_scope["states"].has_key(profile.state),
            )
        )
    if profile.county:
        # A program scoped to a whole state lists no counties and must still
        # match a county query - otherwise filtering by county would hide SNAP.
        filters.append(
            or_(
                _jsonb_array_length(table.geo_scope["counties"]) == 0,
                table.geo_scope["counties"].has_any(
                    pg_array(_county_variants(profile.county), type_=Text)
                ),
            )
        )
    if _is_student(profile) is False:
        # Campus-scoped programs are only for people at that campus. Applied
        # only when the user said they are not a student; an unstated student
        # status leaves them visible.
        filters.append(_jsonb_array_length(table.geo_scope["campuses"]) == 0)
    return filters


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _wanted_categories(
    profile: UserProfile, categories: Sequence[ProgramCategory] | None
) -> tuple[frozenset[ProgramCategory], frozenset[ProgramCategory]]:
    """``(primary, secondary)`` categories implied by the request.

    An explicitly requested category is always primary - the caller asked for
    it. Otherwise the stated needs are expanded through the curated
    ``NEED_TO_PROGRAM_CATEGORIES`` mapping, whose first entry per need is the
    primary one.
    """
    if categories:
        return frozenset(categories), frozenset()
    primary: set[ProgramCategory] = set()
    secondary: set[ProgramCategory] = set()
    for need in profile.needs:
        mapped = NEED_TO_PROGRAM_CATEGORIES.get(need, ())
        if mapped:
            primary.add(mapped[0])
            secondary.update(mapped[1:])
    return frozenset(primary), frozenset(secondary - primary)


def _need_terms(profile: UserProfile) -> list[str]:
    """Query terms implied by the stated needs, in need order."""
    terms: dict[str, None] = {}
    for need in profile.needs:
        for term in NEED_QUERY_TERMS.get(need, ()):
            terms.setdefault(term, None)
    return list(terms)


def _need_overlap(
    program: AssistanceProgram,
    *,
    profile: UserProfile,
    categories: Sequence[ProgramCategory] | None,
) -> float:
    """How well this program answers what the user said they need, in ``[0, 1]``."""
    primary, secondary = _wanted_categories(profile, categories)
    if program.category in primary:
        category_component = _PRIMARY_CATEGORY_SCORE
    elif program.category in secondary:
        category_component = _SECONDARY_CATEGORY_SCORE
    else:
        category_component = 0.0

    terms = _need_terms(profile)
    if terms:
        hits = len(_matched_terms(terms, program))
        keyword_component = min(1.0, hits / len(terms))
    else:
        keyword_component = 0.0

    if not primary and not secondary and not terms:
        # Nothing was asked for, so this component is genuinely neutral rather
        # than zero-because-bad. Returning 0.0 keeps the weighted sum honest:
        # every candidate loses the same amount.
        return 0.0
    return _NEED_CATEGORY_SHARE * category_component + _NEED_KEYWORD_SHARE * keyword_component


def _best_freshness(program: AssistanceProgram, *, now: datetime) -> float:
    """Freshness weight of the program's most recently retrieved source."""
    if not program.sources:
        return 0.0
    return max(_FRESHNESS_WEIGHTS[source.current_freshness(now)] for source in program.sources)


def score_program(
    program: AssistanceProgram,
    *,
    profile: UserProfile,
    ts_rank: float,
    categories: Sequence[ProgramCategory] | None = None,
    now: datetime | None = None,
) -> RetrievalScore:
    """The full ranking breakdown for one program. Pure and deterministic."""
    moment = now or datetime.now(UTC)
    text_match = ts_rank / (ts_rank + _TS_RANK_HALF) if ts_rank > 0.0 else 0.0
    need_overlap = _need_overlap(program, profile=profile, categories=categories)
    freshness = _best_freshness(program, now=moment)
    verification = _VERIFICATION_FACTORS[program.verification_status]
    source_freshness = freshness * verification
    total = (
        WEIGHT_TEXT_MATCH * text_match
        + WEIGHT_NEED_OVERLAP * need_overlap
        + WEIGHT_SOURCE_FRESHNESS * source_freshness
    )
    return RetrievalScore(
        total=round(min(max(total, 0.0), 1.0), 6),
        text_match=round(text_match, 6),
        need_overlap=round(need_overlap, 6),
        source_freshness=round(source_freshness, 6),
        ts_rank=round(ts_rank, 6),
        freshness=round(freshness, 6),
        verification_factor=verification,
    )


def _reasons(
    program: AssistanceProgram,
    *,
    profile: UserProfile,
    matched: Sequence[str],
    categories: Sequence[ProgramCategory] | None,
    has_ruleset: bool,
) -> list[str]:
    """Plain-language, user-safe explanations of why this candidate surfaced.

    Everything here is either the user's own words back to them or text the
    catalog's authors wrote (``geo_scope.label``). Nothing is generated about
    the program itself, so a reason cannot become a fabricated claim.
    """
    reasons: list[str] = []
    primary, secondary = _wanted_categories(profile, categories)
    for need in profile.needs:
        mapped = NEED_TO_PROGRAM_CATEGORIES.get(need, ())
        if program.category in mapped:
            reasons.append(f"You said you need help with {_NEED_LABELS[need]}.")
            break
    if not reasons and program.category in (primary | secondary):
        reasons.append(f"It is in the {program.category.value.replace('_', ' ').lower()} category.")
    if matched:
        quoted = ", ".join(f'"{term}"' for term in matched[:3])
        reasons.append(f"It matches what you wrote: {quoted}.")
    if program.geo_scope.label:
        reasons.append(f"Service area: {program.geo_scope.label}.")
    if program.geo_scope.campuses:
        reasons.append(f"Open to people at {', '.join(program.geo_scope.campuses)}.")
    if program.verification_status is not VerificationStatus.VERIFIED:
        reasons.append(
            "We have not finished checking this listing against its official "
            "source, so it is shown as a lead rather than an assessment."
        )
    elif not has_ruleset:
        reasons.append(
            "We have not finished recording this program's requirements, so we "
            "cannot assess it for you yet."
        )
    return reasons


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
def _tsquery(tokens: Sequence[str]) -> ColumnElement[Any] | None:
    """OR the tokens into one tsquery, or ``None`` if there is no text signal.

    ``plainto_tsquery`` is used per token - it never raises on punctuation,
    unlike ``to_tsquery`` - and the per-token results are OR-ed with ``||``.
    Handing the whole phrase to ``plainto_tsquery`` in one go would AND every
    word, so "help with groceries and rent" would require *all* of them and
    match nothing. OR-ing recovers recall, and ``ts_rank`` still rewards a
    program that matches more of the query than its neighbours.
    """
    if not tokens:
        return None
    combined: ColumnElement[Any] = func.plainto_tsquery(_TS_CONFIG, tokens[0])
    for token in tokens[1:]:
        combined = combined.op("||")(func.plainto_tsquery(_TS_CONFIG, token))
    return combined


async def _ranked_ids(
    session: AsyncSession,
    *,
    profile: UserProfile,
    tokens: Sequence[str],
    require_text_match: bool,
    categories: Sequence[ProgramCategory] | None,
    pool: int,
) -> list[tuple[uuid.UUID, float]]:
    """One query: the candidate pool as ``(program_id, ts_rank)``, ranked.

    This is also the single place a vector search would slot in. pgvector would
    add an embedding column to ``assistance_programs`` and a second ordered
    pool from ``embedding <=> :query_embedding``; the two id lists would be
    fused here (reciprocal rank or a weighted blend) and handed to the same
    rescoring below, which is why the score lives in Python rather than in the
    ``ORDER BY``.
    """
    table = models.AssistanceProgram
    tsquery = _tsquery(tokens)
    rank: ColumnElement[float] = (
        func.ts_rank(table.search_vector, tsquery) if tsquery is not None else literal(0.0)
    )

    statement = select(table.id, rank.label("ts_rank")).where(*_scope_filters(profile))
    if categories:
        statement = statement.where(table.category.in_([item.value for item in categories]))
    if tsquery is not None and require_text_match:
        statement = statement.where(table.search_vector.op("@@")(tsquery))

    # Deterministic: ``slug`` is unique, so the pool is never order-dependent.
    statement = statement.order_by(rank.desc(), table.slug.asc()).limit(pool)
    result = await session.execute(statement)
    return [(row[0], float(row[1] or 0.0)) for row in result.all()]


async def _rulesets_for(
    session: AsyncSession,
    programs: Sequence[AssistanceProgram],
    *,
    on: date,
) -> dict[uuid.UUID, RuleSet]:
    """One query: the in-force approved rule version for each program.

    "In force on this date" can match more than one version - a version with a
    future ``effective_to`` overlaps the one that superseded it - so the rows
    come back newest-effective-first and the first per program wins, which is
    the same rule ``SqlProgramRepository.get_active_ruleset`` applies to one
    program at a time.

    Only programs that may support an eligibility claim are asked about: see
    the module docstring on why an unverified record comes back with no rules.
    """
    claimable = {
        program.id: program.slug for program in programs if program.supports_eligibility_claims()
    }
    if not claimable:
        return {}
    rules = models.ProgramRuleVersion
    statement = (
        select(rules)
        .where(
            rules.program_id.in_(list(claimable)),
            rules.review_status == "APPROVED",
            rules.effective_from <= on,
            or_(rules.effective_to.is_(None), rules.effective_to > on),
        )
        .order_by(rules.program_id, rules.effective_from.desc(), rules.version.desc())
    )
    rows = (await session.scalars(statement)).all()
    found: dict[uuid.UUID, RuleSet] = {}
    for row in rows:
        if row.program_id in found:
            continue
        found[row.program_id] = RuleSet.model_validate(
            {
                "version": row.version,
                "program_slug": claimable[row.program_id],
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
    return found


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------
async def search_scored_programs(
    session: AsyncSession,
    *,
    profile: UserProfile,
    query: str | None = None,
    categories: Sequence[ProgramCategory] | None = None,
    limit: int = 20,
    now: datetime | None = None,
) -> list[ScoredProgram]:
    """:func:`search_programs` with the score breakdown attached.

    Ordering is ``(-total, slug)``, so it is total-ordered and reproducible for
    a given catalog: two programs that score identically always come back in
    slug order rather than in whatever order Postgres felt like.
    """
    moment = now or datetime.now(UTC)
    bounded = max(1, min(limit, _MAX_LIMIT))
    pool = min(MAX_CANDIDATE_POOL, max(bounded * _POOL_MULTIPLIER, _MIN_CANDIDATE_POOL))

    explicit_tokens = _tokenise(query) if query else []
    # Derived need terms rank but never filter: they are our inference about
    # what the user meant, so they must not be able to hide a program.
    tokens = explicit_tokens or _need_terms(profile)
    ranked = await _ranked_ids(
        session,
        profile=profile,
        tokens=tokens,
        require_text_match=bool(explicit_tokens),
        categories=categories,
        pool=pool,
    )
    if not ranked:
        return []

    ranks = dict(ranked)
    repository = SqlProgramRepository(session)
    programs = await repository.get_programs_by_ids(list(ranks))
    rulesets = await _rulesets_for(session, programs, on=moment.date())

    scored: list[ScoredProgram] = []
    for program in programs:
        score = score_program(
            program,
            profile=profile,
            ts_rank=ranks.get(program.id, 0.0),
            categories=categories,
            now=moment,
        )
        ruleset = rulesets.get(program.id)
        matched = _matched_terms(tokens, program)
        candidate = RetrievedProgram(
            program=program,
            ruleset=ruleset,
            sources=list(program.sources),
            retrieval_score=score.total,
            matched_terms=matched,
            retrieval_reasons=_reasons(
                program,
                profile=profile,
                matched=matched,
                categories=categories,
                has_ruleset=ruleset is not None,
            ),
        )
        scored.append(ScoredProgram(candidate=candidate, score=score))

    scored.sort(key=lambda item: (-item.score.total, item.candidate.program.slug))
    del scored[bounded:]
    # Counts and slugs only. The query is the user's own words and never
    # reaches a log line.
    _log.debug(
        "retrieval.search.completed",
        pool=len(ranked),
        returned=len(scored),
        has_query=bool(explicit_tokens),
        assessable=sum(1 for item in scored if item.candidate.ruleset is not None),
    )
    return scored


async def search_programs(
    session: AsyncSession,
    *,
    profile: UserProfile,
    query: str | None = None,
    categories: Sequence[ProgramCategory] | None = None,
    limit: int = 20,
) -> list[RetrievedProgram]:
    """Candidates for ``profile``, ordered by descending ``retrieval_score``.

    ``query`` is the user's own search text when there is one; it is a hard
    full-text filter. ``categories``, when supplied, is a hard filter too. The
    profile's stated needs are never a filter - only a ranking signal.

    Each candidate comes back complete (program, sources, and the in-force
    approved rule set when the record may support a claim) so that the
    eligibility engine performs no I/O of its own.
    """
    scored = await search_scored_programs(
        session, profile=profile, query=query, categories=categories, limit=limit
    )
    return [item.candidate for item in scored]


async def get_program_detail(
    session: AsyncSession, slug_or_id: str | uuid.UUID
) -> ProgramDetail | None:
    """One program with its sources and in-force rule version, or ``None``.

    Accepts either the catalog slug or the program id, because the API exposes
    both and a caller holding one should not have to resolve the other first.

    Three statements, independent of how many sources or rule versions the
    program has: the program row, a ``selectin`` load of its sources, and the
    in-force rule version. There is no per-source or per-version query, and
    ``AssistanceProgram.rule_versions`` is mapped ``lazy="raise_on_sql"``, so an
    accidental N+1 through this path raises rather than quietly costing a
    round trip per row. ``tests/integration/test_retrieval_db.py`` asserts the
    count.
    """
    repository = SqlProgramRepository(session)
    if isinstance(slug_or_id, uuid.UUID):
        return await repository.get_program(slug_or_id)
    try:
        program_id = uuid.UUID(slug_or_id)
    except ValueError:
        return await repository.get_program_by_slug(slug_or_id)
    return await repository.get_program(program_id)


class HybridRetriever:
    """The :class:`app.schemas.contracts.Retriever` implementation.

    Constructed from a :class:`~app.db.repositories.programs.SqlProgramRepository`
    so that the mapping from a catalog row to an
    :class:`~app.schemas.program.AssistanceProgram` lives in exactly one place,
    and so that this class can reach the repository's session for the full-text
    ranking query, which the ``ProgramRepository`` Protocol does not expose.

    Bound to one session, and therefore to one task: do not share an instance
    across concurrent branches of the graph.
    """

    __slots__ = ("_repository", "_session")

    def __init__(self, repository: SqlProgramRepository) -> None:
        self._repository = repository
        self._session = repository.session

    @classmethod
    def from_session(cls, session: AsyncSession) -> HybridRetriever:
        """Convenience constructor for callers that only hold a session."""
        return cls(SqlProgramRepository(session))

    async def retrieve(
        self,
        profile: UserProfile,
        *,
        needs: Sequence[NeedCategory] | None = None,
        limit: int = 20,
    ) -> list[RetrievedProgram]:
        """Candidates for a profile, ordered by descending ``retrieval_score``.

        ``needs`` overrides the profile's own ``needs`` when supplied, which is
        how the graph's needs-categorisation step feeds a refined reading of the
        narrative back into retrieval without mutating the stored profile.
        """
        effective = profile if needs is None else profile.model_copy(update={"needs": list(needs)})
        return await search_programs(self._session, profile=effective, limit=limit)

    async def retrieve_for_programs(
        self, program_ids: Sequence[uuid.UUID]
    ) -> list[RetrievedProgram]:
        """Rehydrate specific candidates for selective re-evaluation.

        No scoring: the caller already decided these are the programs it cares
        about, and inventing a relevance score for them would reorder a result
        list the user is already looking at. ``retrieval_score`` stays at its
        default of 0.0.
        """
        if not program_ids:
            return []
        programs = await self._repository.get_programs_by_ids(program_ids)
        rulesets = await _rulesets_for(self._session, programs, on=datetime.now(UTC).date())
        return [
            RetrievedProgram(
                program=program,
                ruleset=rulesets.get(program.id),
                sources=list(program.sources),
            )
            for program in programs
        ]

    async def search_sources(
        self,
        query: str,
        *,
        program_id: uuid.UUID | None = None,
        limit: int = 5,
    ) -> list[ProgramSource]:
        """Passage search over source excerpts, for grounding the assistant."""
        return await self._repository.search_sources(query, program_id=program_id, limit=limit)
