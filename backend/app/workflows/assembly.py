"""Turning evaluations and candidates into :class:`ProgramMatch` objects.

Shared by the generation fan-out, which needs provisional matches to draft a
plan against, and by ``merge_outputs``, which needs the final ones. Having one
builder means the plan and the results page cannot disagree about which
programs were assessed or what their sources were.

The status is copied from the evaluation and nowhere else. ``ProgramMatch``
validates ``status is evaluation.status``, so a bug here is a validation error
rather than a wrong answer shown to a person.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

from app.schemas import (
    ConditionEvidence,
    EligibilityEvaluation,
    MatchExplanation,
    MatchStatus,
    ProgramCategory,
    ProgramMatch,
    ProgramSource,
    RetrievedProgram,
)

__all__ = [
    "STATUS_ORDER",
    "build_matches",
    "candidate_index",
    "cited_source_ids",
    "sources_for",
    "walk_evidence",
]

#: Reading order for the results page. A program the user might actually get
#: comes first; one that was ruled out comes last but is still shown, with the
#: requirement that ruled it out, because "why not" is the second most useful
#: thing this product can tell someone.
STATUS_ORDER: Final[dict[MatchStatus, int]] = {
    MatchStatus.POTENTIAL_MATCH: 0,
    MatchStatus.MORE_INFO_NEEDED: 1,
    MatchStatus.UNLIKELY_MATCH: 2,
}


def walk_evidence(evidence: ConditionEvidence) -> Iterator[ConditionEvidence]:
    """Every node of one evidence tree, parents before children."""
    yield evidence
    for child in evidence.children:
        yield from walk_evidence(child)


def candidate_index(
    candidates: Sequence[RetrievedProgram],
) -> dict[uuid.UUID, RetrievedProgram]:
    """Candidates keyed by program id, first occurrence winning."""
    index: dict[uuid.UUID, RetrievedProgram] = {}
    for candidate in candidates:
        index.setdefault(candidate.program.id, candidate)
    return index


def cited_source_ids(evaluation: EligibilityEvaluation) -> list[str]:
    """Every source id the evaluation's evidence tree cites, in first-seen order."""
    found: dict[str, None] = {}
    for root in evaluation.evidence:
        for node in walk_evidence(root):
            for source_id in node.source_ids:
                found.setdefault(source_id, None)
    return list(found)


def sources_for(
    evaluation: EligibilityEvaluation,
    candidate: RetrievedProgram,
    *,
    exclude: frozenset[str] = frozenset(),
) -> list[ProgramSource]:
    """The sources this evaluation actually relied on.

    Filtered three ways, each for a stated reason: a source the evidence never
    cited is not evidence for this verdict; a source in ``exclude`` failed
    citation verification; and a ``STALE`` source may not support an
    eligibility claim at all. The result can legitimately be empty, which
    ``merge_outputs`` turns into a visible caveat rather than hiding.
    """
    wanted = set(cited_source_ids(evaluation)) - exclude
    return [
        source
        for source in candidate.sources
        if source.id in wanted and source.supports_eligibility_claim()
    ]


def build_matches(
    evaluations: Sequence[EligibilityEvaluation],
    candidates: Sequence[RetrievedProgram],
    *,
    explanations: Mapping[str, MatchExplanation] | None = None,
    category_order: Sequence[ProgramCategory] = (),
    exclude_source_ids: frozenset[str] = frozenset(),
) -> list[ProgramMatch]:
    """Assemble and order the matches for one run.

    ``category_order`` is the output of ``categorize_needs``: a program whose
    category answers a need the user actually stated sorts above one that
    merely happens to fit. Within a status and a category the engine's score
    orders them, and the slug breaks ties so the list is deterministic.
    """
    index = candidate_index(candidates)
    prose = explanations or {}
    matches: list[ProgramMatch] = []
    for evaluation in evaluations:
        candidate = index.get(evaluation.program_id)
        if candidate is None:
            # An evaluation with no candidate behind it cannot be rendered:
            # there is no program record to name, link or attribute.
            continue
        matches.append(
            ProgramMatch(
                program=candidate.program.to_summary(),
                status=evaluation.status,
                score=evaluation.score,
                evaluation=evaluation,
                explanation=prose.get(evaluation.program_slug),
                sources=sources_for(evaluation, candidate, exclude=exclude_source_ids),
            )
        )

    ranks = {category: position for position, category in enumerate(category_order)}
    fallback = len(ranks)

    def sort_key(match: ProgramMatch) -> tuple[int, int, float, str]:
        return (
            STATUS_ORDER[match.status],
            ranks.get(match.program.category, fallback),
            -match.score,
            match.program.slug,
        )

    matches.sort(key=sort_key)
    return matches
