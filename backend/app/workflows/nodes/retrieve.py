"""The first fan-out, and the barrier that joins it.

Three branches run concurrently and write three disjoint sets of channels:

====================================  =========================
node                                  writes
====================================  =========================
``retrieve_programs``                 ``candidates``
``validate_profile_evidence``         ``evidence_warnings``
``categorize_needs``                  ``need_categories``, ``program_categories``
====================================  =========================

They are genuinely independent, which is why they are concurrent rather than
sequential. In particular ``categorize_needs`` does **not** feed
``retrieve_programs``: retrieval reads ``profile.needs``, which is already
assembled, and a branch cannot see a sibling's writes in the same superstep.
The categories it derives are used downstream, by ``merge_outputs``, to order
results so that a program answering a need the user actually stated appears
above one that merely happens to fit.

Each branch owns its own session. ``ProgramRepository`` and ``Retriever`` are
bound to one ``AsyncSession`` and sharing one between concurrent tasks is a
data race, so :meth:`~app.workflows.deps.WorkflowDeps.with_retriever` opens a
scope per branch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from app.core.logging import get_logger
from app.schemas import (
    NEED_TO_PROGRAM_CATEGORIES,
    ExtractedProfile,
    NeedCategory,
    ProgramCategory,
    RetrievedProgram,
    VerificationStatus,
)
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import RECOVERABLE_ERRORS, entering, profile_of, recovered
from app.workflows.state import NavigatorState, StateUpdate

__all__ = [
    "CATEGORIZE_NODE",
    "EVIDENCE_NODE",
    "JOIN_NODE",
    "NEED_KEYWORDS",
    "RETRIEVE_NODE",
    "categorize_needs",
    "join_candidates",
    "retrieve_programs",
    "validate_profile_evidence",
]

RETRIEVE_NODE = "retrieve_programs"
EVIDENCE_NODE = "validate_profile_evidence"
CATEGORIZE_NODE = "categorize_needs"
JOIN_NODE = "join_candidates"

_log = get_logger(__name__)

#: Plain words people actually use, mapped onto the need vocabulary. A
#: deterministic table rather than a model call: categorisation steers which
#: programs a person is shown, so it has to be reviewable and reproducible.
#: Matched on word boundaries against the narrative, so "food" does not fire on
#: "foodie" and "rent" does not fire on "parent".
NEED_KEYWORDS: Final[dict[NeedCategory, tuple[str, ...]]] = {
    NeedCategory.FOOD: (
        "food",
        "groceries",
        "grocery",
        "hungry",
        "hunger",
        "meals",
        "eat",
        "pantry",
        "snap",
        "wic",
    ),
    NeedCategory.HEALTHCARE: (
        "doctor",
        "medical",
        "medicaid",
        "health",
        "insurance",
        "prescription",
        "dentist",
        "clinic",
    ),
    NeedCategory.UTILITIES: (
        "utility",
        "utilities",
        "electric",
        "electricity",
        "gas bill",
        "heat",
        "heating",
        "water bill",
        "shut off",
        "shutoff",
        "shut-off",
        "disconnection",
    ),
    NeedCategory.HOUSING: (
        "rent",
        "landlord",
        "eviction",
        "evicted",
        "housing",
        "homeless",
        "unhoused",
        "shelter",
        "mortgage",
        "apartment",
    ),
    NeedCategory.CHILDCARE: ("childcare", "child care", "daycare", "day care", "babysitter"),
    NeedCategory.TRANSPORTATION: (
        "bus",
        "transit",
        "car",
        "commute",
        "transportation",
        "gas money",
    ),
    NeedCategory.EDUCATION_COSTS: (
        "tuition",
        "textbook",
        "textbooks",
        "books",
        "semester",
        "school fees",
        "course",
    ),
    NeedCategory.INCOME_SUPPORT: (
        "income",
        "cash",
        "laid off",
        "lost my job",
        "unemployed",
        "unemployment",
        "benefits",
        "tax credit",
        "refund",
    ),
    NeedCategory.INTERNET_PHONE: (
        "internet",
        "wifi",
        "wi-fi",
        "broadband",
        "phone bill",
        "data plan",
    ),
    NeedCategory.LEGAL_HELP: ("lawyer", "legal", "court", "attorney"),
    NeedCategory.MENTAL_HEALTH: (
        "therapy",
        "therapist",
        "counseling",
        "counselling",
        "anxiety",
        "depression",
    ),
    NeedCategory.EMERGENCY_EXPENSE: ("emergency", "urgent", "unexpected bill", "right away"),
}

#: Shortest quotation we will treat as checkable evidence. A two-word span
#: occurs in almost any sentence, so "it appears in the narrative" would stop
#: being evidence of anything.
_MIN_SPAN_CHARS: Final[int] = 8

_NO_CANDIDATES = (
    "We could not find programs to check against your situation this time. "
    "Nothing here means you do not qualify for help."
)
_RETRIEVAL_FAILED = (
    "We could not search the program catalog just now, so this run has no programs in it."
)


# ---------------------------------------------------------------------------
# branch 1: retrieval
# ---------------------------------------------------------------------------
async def retrieve_programs(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Find the programs worth evaluating. Generous by design.

    Retrieval decides relevance, never eligibility. A candidate the engine
    later rules out costs one evaluation; a candidate never retrieved is help
    the user never hears about.
    """
    profile = profile_of(state)
    try:
        async with deps.with_retriever() as retriever:
            candidates = await retriever.retrieve(profile, limit=deps.max_candidates)
    except RECOVERABLE_ERRORS as exc:
        return recovered(
            exc, node=RETRIEVE_NODE, limitation=_RETRIEVAL_FAILED, extra={"candidates": []}
        )

    _log.info(
        "workflow.retrieve.done",
        node=RETRIEVE_NODE,
        candidates=len(candidates),
        with_rules=sum(1 for item in candidates if item.ruleset is not None),
        unverified=sum(
            1
            for item in candidates
            if item.program.verification_status is not VerificationStatus.VERIFIED
        ),
    )
    return {"candidates": list(candidates)}


# ---------------------------------------------------------------------------
# branch 2: the extraction's own evidence
# ---------------------------------------------------------------------------
def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _evidence_warnings(extracted: ExtractedProfile | None, narrative: str) -> list[str]:
    """Which extracted fields are not backed by the user's own words.

    Every value the model wrote came with a ``source_span`` it claims to have
    read it from. Checking that the span is really in the narrative is cheap,
    deterministic, and catches the one failure mode that matters most here: a
    confident field nobody ever stated.

    Returns machine-readable reason codes - ``field:reason`` - rather than
    prose, because these are attached to state and must be safe to count and
    compare without quoting anything the user wrote.
    """
    if extracted is None:
        return []
    haystack = _normalise(narrative)
    warnings: list[str] = []
    for evidence in extracted.field_evidence:
        span = (evidence.source_span or "").strip()
        if not span:
            warnings.append(f"{evidence.field}:no_source_span")
            continue
        if len(span) < _MIN_SPAN_CHARS:
            warnings.append(f"{evidence.field}:span_too_short")
            continue
        if _normalise(span) not in haystack:
            warnings.append(f"{evidence.field}:span_not_in_narrative")
    for flag in extracted.flags:
        span = (flag.source_span or "").strip()
        if span and len(span) >= _MIN_SPAN_CHARS and _normalise(span) not in haystack:
            warnings.append(f"flags.{flag.flag.value}:span_not_in_narrative")
    return list(dict.fromkeys(warnings))


async def validate_profile_evidence(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Check the extraction quoted the narrative rather than invented it.

    Deterministic and local: it compares spans against the text the user
    actually submitted. Nothing is corrected here - a profile field is already
    confidence-filtered by the time it arrives - so the output is a warning
    list that ``merge_outputs`` turns into a user-visible caveat when it is
    non-empty.
    """
    warnings = _evidence_warnings(state.get("extracted"), state.get("narrative", ""))
    _log.info(
        "workflow.evidence.checked",
        node=EVIDENCE_NODE,
        warnings=len(warnings),
        reasons=sorted({warning.split(":", 1)[1] for warning in warnings}),
    )
    return {"evidence_warnings": warnings}


# ---------------------------------------------------------------------------
# branch 3: needs
# ---------------------------------------------------------------------------
def _keyword_needs(narrative: str) -> list[NeedCategory]:
    haystack = _normalise(narrative)
    found: list[NeedCategory] = []
    for need, keywords in NEED_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", haystack):
                found.append(need)
                break
    return found


def program_categories_for(needs: Sequence[NeedCategory]) -> list[ProgramCategory]:
    """Catalog categories implied by a set of needs, in first-seen order."""
    categories: dict[ProgramCategory, None] = {}
    for need in needs:
        for category in NEED_TO_PROGRAM_CATEGORIES.get(need, ()):
            categories.setdefault(category, None)
    return list(categories)


async def categorize_needs(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Decide what kind of help this person is asking for.

    The profile's own ``needs`` come first, because they were either stated
    directly or extracted with evidence. The keyword pass over the narrative
    only adds categories the profile does not already carry, so it can widen
    the reading of a situation but never contradict it.
    """
    profile = profile_of(state)
    needs: dict[NeedCategory, None] = dict.fromkeys(profile.needs)
    for need in _keyword_needs(state.get("narrative", "")):
        needs.setdefault(need, None)
    needs.pop(NeedCategory.UNKNOWN, None)

    ordered = list(needs)
    categories = program_categories_for(ordered)
    _log.info(
        "workflow.categorize.done",
        node=CATEGORIZE_NODE,
        needs=[need.value for need in ordered],
        categories=[category.value for category in categories],
    )
    return {"need_categories": ordered, "program_categories": categories}


# ---------------------------------------------------------------------------
# the barrier
# ---------------------------------------------------------------------------
def _unrated(candidates: Sequence[RetrievedProgram]) -> list[str]:
    return [
        candidate.program.official_name
        for candidate in candidates
        if candidate.ruleset is None or not candidate.program.supports_eligibility_claims()
    ]


async def join_candidates(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Barrier for the first fan-out. Records what each branch contributed.

    LangGraph runs this once, after all three branches have committed, because
    all three have an edge into it. It writes no result of its own beyond that
    audit record and the caveats the combination implies - which is the point:
    a join that recomputed anything would make the branches' single-writer
    guarantee meaningless.
    """
    update = entering(JOIN_NODE)
    candidates = list(state.get("candidates") or ())
    warnings = list(state.get("evidence_warnings") or ())
    needs = list(state.get("need_categories") or ())

    limitations: list[str] = []
    if not candidates:
        limitations.append(_NO_CANDIDATES)
    if warnings:
        limitations.append(
            "We could not match part of what we understood back to your own words, "
            "so please check the details we show you."
        )
    unrated = _unrated(candidates)
    if unrated:
        limitations.append(
            "We have not finished encoding the rules for "
            f"{len(unrated)} of the programs we found, so we describe them without assessing them."
        )

    update["branch_contributions"] = {
        RETRIEVE_NODE: len(candidates),
        EVIDENCE_NODE: len(warnings),
        CATEGORIZE_NODE: len(needs),
    }
    if limitations:
        update["limitations"] = limitations
    _log.info(
        "workflow.join.done",
        node=JOIN_NODE,
        candidates=len(candidates),
        evidence_warnings=len(warnings),
        needs=len(needs),
        unrated=len(unrated),
    )
    return update
