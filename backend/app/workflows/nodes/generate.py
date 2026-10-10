"""The second fan-out: language, next steps and citation checking.

Three concurrent branches over three disjoint channels:

===========================  =========================
node                         writes
===========================  =========================
``generate_explanations``    ``explanations``
``generate_action_plan``     ``plan_draft``, ``action_items``
``verify_citations``         ``citation_results``
===========================  =========================

Every verdict already exists by the time any of this runs, which is the whole
design: the two model-backed branches receive a decided status and may only
word it or act on it. ``verify_citations`` is deliberately in the same fan-out
rather than after them - it checks the catalog against itself, not the model's
prose, so it has nothing to wait for.

All three degrade rather than fail. A timeout in explanation generation costs
the user some prose, not their results; the matches, the evidence and the
requirement bullets are all deterministic and already decided.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final

from app.core.errors import AppError
from app.core.logging import get_logger
from app.eligibility.documents import document_action_drafts
from app.schemas import (
    ActionItemDraft,
    ActionItemKind,
    ActionPlanDraft,
    AssistanceProgram,
    Citation,
    CitationVerificationReport,
    EligibilityEvaluation,
    MatchExplanation,
    MatchStatus,
    ModelUsage,
    ProgramMatch,
    ProgramSource,
    RejectedCitation,
    RetrievedProgram,
    make_dedup_key,
)
from app.workflows.assembly import build_matches, candidate_index, cited_source_ids
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import (
    RECOVERABLE_ERRORS,
    log_failure,
    profile_of,
    recovered,
    user_facing,
)
from app.workflows.state import NavigatorState, StateUpdate

__all__ = [
    "EXPLAIN_NODE",
    "PLAN_NODE",
    "VERIFY_NODE",
    "fallback_plan_draft",
    "generate_action_plan",
    "generate_explanations",
    "verify_citations",
]

EXPLAIN_NODE = "generate_explanations"
PLAN_NODE = "generate_action_plan"
VERIFY_NODE = "verify_citations"

_log = get_logger(__name__)

#: Statuses that belong in an action plan. There is nothing to do about a
#: program whose hard requirement the user does not meet, and putting one in
#: the plan would send somebody to fill in a form for nothing.
_PLANNABLE: Final[frozenset[MatchStatus]] = frozenset(
    {MatchStatus.POTENTIAL_MATCH, MatchStatus.MORE_INFO_NEEDED}
)

_SOME_PROSE_MISSING = (
    "We could not write a plain-language summary for every program this time. "
    "The requirements and official links below are still complete."
)
_NO_PROSE = (
    "We could not write plain-language summaries this time, so each program is shown "
    "with its official requirements instead."
)
_PLAN_FELL_BACK = (
    "We built your next steps from the programs' own requirements rather than writing them, "
    "so the wording is plainer than usual."
)
_CITATIONS_UNCHECKED = (
    "We could not re-check every official source for this run, so please follow the links "
    "to confirm the details."
)


# ---------------------------------------------------------------------------
# branch 1: explanations
# ---------------------------------------------------------------------------
async def _explain_one(
    evaluation: EligibilityEvaluation,
    *,
    state: NavigatorState,
    deps: WorkflowDeps,
    program: AssistanceProgram,
    gate: asyncio.Semaphore,
) -> tuple[str, MatchExplanation | None, ModelUsage | None, AppError | None]:
    """One explanation call, with its own failure contained.

    Only the four dependency failures in
    :data:`~app.workflows.nodes.base.RECOVERABLE_ERRORS` are caught.
    ``CancelledError`` and programming errors propagate, because a cancelled
    run must stay cancelled and a bug must not be reported to the user as "we
    could not word this".
    """
    async with gate:
        try:
            result = await deps.gemini.explain_match(
                profile_of(state),
                program,
                evaluation,
                request_id=state.get("request_id"),
            )
        except RECOVERABLE_ERRORS as exc:
            log_failure(EXPLAIN_NODE, exc, recoverable=True)
            return evaluation.program_slug, None, None, exc
    return evaluation.program_slug, result.value, result.usage, None


async def generate_explanations(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Word each decided status in plain language, concurrently and bounded.

    Capped at ``deps.max_explanations`` and ordered the way the results page
    is, so when the cap bites it is the programs furthest down the page that go
    without prose - never the one the user is most likely to act on.
    """
    evaluations = list(state.get("evaluations") or ())
    candidates = list(state.get("candidates") or ())
    if not evaluations:
        return {"explanations": {}}

    index = candidate_index(candidates)
    ordered = build_matches(
        evaluations, candidates, category_order=state.get("program_categories") or ()
    )
    wanted = ordered[: deps.max_explanations]
    gate = asyncio.Semaphore(deps.explanation_concurrency)
    by_slug = {evaluation.program_slug: evaluation for evaluation in evaluations}

    tasks = [
        _explain_one(
            by_slug[match.program.slug],
            state=state,
            deps=deps,
            program=index[match.program_id].program,
            gate=gate,
        )
        for match in wanted
        if match.program.slug in by_slug and match.program_id in index
    ]
    results = await asyncio.gather(*tasks)

    explanations: dict[str, MatchExplanation] = {}
    usage: list[ModelUsage] = []
    failures: list[AppError] = []
    for slug, explanation, call_usage, error in results:
        if explanation is not None:
            explanations[slug] = explanation
        if call_usage is not None:
            usage.append(call_usage)
        if error is not None:
            failures.append(error)

    update: StateUpdate = {"explanations": explanations}
    if usage:
        update["model_usage"] = usage
    if failures:
        update["errors"] = [user_facing(failures[0])]
        update["limitations"] = [_NO_PROSE if not explanations else _SOME_PROSE_MISSING]
    _log.info(
        "workflow.explain.done",
        node=EXPLAIN_NODE,
        requested=len(tasks),
        written=len(explanations),
        failed=len(failures),
        fallback=sum(1 for item in usage if item.source == "fallback"),
    )
    return update


# ---------------------------------------------------------------------------
# branch 2: the action plan
# ---------------------------------------------------------------------------
def fallback_plan_draft(
    matches: Sequence[ProgramMatch],
    document_drafts: Sequence[ActionItemDraft],
) -> ActionPlanDraft:
    """A usable plan built from the programs' own requirements.

    Used when the model is unavailable or its draft fails validation. Every
    line is author-written copy over catalog data, so it is plain but never
    wrong, and the user still leaves with documents to gather and links to
    follow rather than an empty page.
    """
    items: list[ActionItemDraft] = list(document_drafts)
    for match in matches:
        if match.status is MatchStatus.MORE_INFO_NEEDED:
            items.append(
                ActionItemDraft(
                    program_slug=match.program.slug,
                    kind=ActionItemKind.CONFIRM_INFO,
                    subject=f"confirm details for {match.program.slug}",
                    title=f"Check the details for {match.program.official_name}",
                    detail=(
                        "We could not check every requirement with what we know. "
                        f"{match.program.administering_org} can tell you where you stand."
                    ),
                    url=match.program.official_url,
                )
            )
        if match.status is MatchStatus.POTENTIAL_MATCH:
            items.append(
                ActionItemDraft(
                    program_slug=match.program.slug,
                    kind=ActionItemKind.APPLY,
                    subject=f"apply to {match.program.slug}",
                    title=f"Apply to {match.program.official_name}",
                    detail=(
                        f"{match.program.administering_org} runs this program. "
                        "Applying is free and you can stop at any point."
                    ),
                    url=match.program.application_url or match.program.official_url,
                )
            )
    return ActionPlanDraft(
        title="Your next steps",
        summary=(
            "Start with the documents - most programs ask for the same few. "
            "Then work down the list. Nothing here is a decision; each program decides for itself."
        ),
        items=items[:60],
    )


def _merge_drafts(
    model_items: Sequence[ActionItemDraft],
    document_items: Sequence[ActionItemDraft],
    *,
    allowed_slugs: frozenset[str],
) -> list[ActionItemDraft]:
    """Model drafts first, with any document requirement they missed added.

    The deterministic document list is derived from the evidence, so it is the
    authority on what paperwork a requirement implies. A model draft for the
    same (program, document) pair is preferred only for its wording.
    """
    merged: dict[tuple[str, str], ActionItemDraft] = {}
    for draft in (*model_items, *document_items):
        if draft.program_slug not in allowed_slugs:
            continue
        key = (draft.program_slug, make_dedup_key(draft.kind, draft.subject))
        merged.setdefault(key, draft)
    return list(merged.values())


async def generate_action_plan(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Draft the next steps for the programs worth acting on."""
    evaluations = list(state.get("evaluations") or ())
    candidates = list(state.get("candidates") or ())
    matches = [
        match
        for match in build_matches(
            evaluations, candidates, category_order=state.get("program_categories") or ()
        )
        if match.status in _PLANNABLE
    ]
    if not matches:
        return {"plan_draft": None, "action_items": []}

    allowed = frozenset(match.program.slug for match in matches)
    plannable_ids = {match.program_id for match in matches}
    document_items = [
        draft
        for draft in document_action_drafts(
            [item for item in evaluations if item.program_id in plannable_ids]
        )
        if draft.program_slug in allowed
    ]

    try:
        result = await deps.gemini.draft_action_plan(
            profile_of(state), matches, request_id=state.get("request_id")
        )
    except RECOVERABLE_ERRORS as exc:
        draft = fallback_plan_draft(matches, document_items)
        items = _merge_drafts((), draft.items, allowed_slugs=allowed)
        failure_update: StateUpdate = recovered(
            exc,
            node=PLAN_NODE,
            limitation=_PLAN_FELL_BACK,
            extra={
                "plan_draft": draft.model_copy(update={"items": items}),
                "action_items": items,
            },
        )
        _log.info("workflow.plan.done", node=PLAN_NODE, items=len(items), source="fallback")
        return failure_update

    items = _merge_drafts(result.value.items, document_items, allowed_slugs=allowed)
    update: StateUpdate = {
        "plan_draft": result.value.model_copy(update={"items": items}),
        "action_items": items,
        "model_usage": [result.usage],
    }
    if result.usage.source == "fallback":
        update["limitations"] = [_PLAN_FELL_BACK]
    _log.info(
        "workflow.plan.done",
        node=PLAN_NODE,
        items=len(items),
        from_model=len(result.value.items),
        from_evidence=len(document_items),
        source=result.usage.source,
    )
    return update


# ---------------------------------------------------------------------------
# branch 3: citation verification
# ---------------------------------------------------------------------------
def _citations_for(
    evaluations: Sequence[EligibilityEvaluation],
    candidates: Sequence[RetrievedProgram],
) -> tuple[list[Citation], list[RejectedCitation], list[ProgramSource]]:
    """One citation per (evidence, source) pair the engine actually relied on.

    A cited id with no source behind it is rejected here rather than passed to
    the verifier: there is no URL or excerpt to check, and "the rule cites a
    document we do not hold" is exactly the ``unknown_source`` case.
    """
    index = candidate_index(list(candidates))
    citations: list[Citation] = []
    rejected: list[RejectedCitation] = []
    known: list[ProgramSource] = []
    seen: set[tuple[str, str]] = set()

    for evaluation in evaluations:
        candidate = index.get(evaluation.program_id)
        if candidate is None:
            continue
        by_id = {source.id: source for source in candidate.sources}
        known.extend(candidate.sources)
        for source_id in cited_source_ids(evaluation):
            key = (evaluation.program_slug, source_id)
            if key in seen:
                continue
            seen.add(key)
            source = by_id.get(source_id)
            if source is None:
                rejected.append(
                    RejectedCitation(
                        source_id=source_id,
                        reason="unknown_source",
                        detail=f"cited by {evaluation.program_slug} but not held for it",
                    )
                )
                continue
            citations.append(
                Citation(
                    source_id=source.id,
                    program_id=candidate.program.id,
                    url=source.url,
                    title=source.title,
                    excerpt=source.excerpt,
                )
            )
    return citations, rejected, known


async def verify_citations(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Re-check every source the verdicts lean on, against the stored catalog.

    No model participates: verification is identity, URL and excerpt matching
    against the stored row, because asking the generator to grade itself is not
    verification. The row in the database is the authority, so this catches
    drift between what retrieval carried and what the catalog now says, as well
    as a source that has gone ``STALE`` since it was ingested.
    """
    evaluations = list(state.get("evaluations") or ())
    candidates = list(state.get("candidates") or ())
    citations, unresolved, known = _citations_for(evaluations, candidates)
    if not citations and not unresolved:
        return {"citation_results": None}

    try:
        async with deps.with_verifier() as verifier:
            report = await verifier.verify(citations, known_sources=known)
    except RECOVERABLE_ERRORS as exc:
        return recovered(
            exc,
            node=VERIFY_NODE,
            limitation=_CITATIONS_UNCHECKED,
            extra={"citation_results": None},
        )

    combined = CitationVerificationReport(
        ok=report.ok and not unresolved,
        verified=report.verified,
        rejected=[*report.rejected, *unresolved],
        unsupported_claims=report.unsupported_claims,
        checked_sources=report.checked_sources,
    )
    _log.info(
        "workflow.citations.done",
        node=VERIFY_NODE,
        checked=len(citations),
        verified=len(combined.verified),
        rejected=len(combined.rejected),
        reasons=sorted({item.reason for item in combined.rejected}),
    )
    return {"citation_results": combined}
