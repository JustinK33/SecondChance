"""``merge_outputs``: the integrity gate between generation and the user.

Everything model-written passes through here, and nothing leaves without being
checked against the three rules that this product's trustworthiness rests on.

1. **The rule engine owns the status.** The status on every match is copied
   from its evaluation, and ``ProgramMatch`` validates ``status is
   evaluation.status``, so an explanation cannot move a verdict even in
   principle. Prose that *claims* an outcome - "you qualify", "you are
   eligible", "guaranteed" - is removed by
   :func:`~app.retrieval.citations.strip_unsupported_claims` sentence by
   sentence, and an explanation whose headline makes such a claim is dropped
   whole rather than patched: a half-corrected explanation is less trustworthy
   than no explanation.
2. **Nothing unverified supports a claim.** A source whose citation failed
   verification is removed from the match it was attached to, and a program
   whose evidence leaned on it gets a visible caveat.
3. **Every removal is recorded.** ``validation`` holds what was dropped and
   why, so the gate is auditable rather than merely clean. It stays in the
   checkpoint, is never logged, and is never written to a column.

Graceful degradation is the fourth rule, and the reason this node is forgiving
where the three above are strict. If the explanation branch failed, the matches
are still complete; if the plan branch failed, a deterministic plan is built
from the programs' own requirements. The user always leaves with the
deterministic half of the product, plus a caveat naming what is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from app.core.logging import get_logger
from app.retrieval.citations import (
    GuaranteeFlag,
    ScrubbedText,
    detect_guarantee_phrasing,
    strip_unsupported_claims,
)
from app.schemas import (
    ActionItemDraft,
    ActionPlanDraft,
    MatchExplanation,
    MatchStatus,
    ProgramMatch,
    ProgramSource,
    RetrievedProgram,
)
from app.workflows.assembly import build_matches, candidate_index
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import entering
from app.workflows.nodes.generate import fallback_plan_draft
from app.workflows.state import NavigatorState, StateUpdate

__all__ = ["NODE", "merge_outputs"]

NODE = "merge_outputs"

_log = get_logger(__name__)

#: Statuses that belong in the plan. Mirrors ``generate_action_plan``.
_PLANNABLE: Final[frozenset[MatchStatus]] = frozenset(
    {MatchStatus.POTENTIAL_MATCH, MatchStatus.MORE_INFO_NEEDED}
)

_PROSE_REMOVED = (
    "We removed some wording that sounded like a decision. Only the program itself can "
    "decide, so we show you its requirements instead."
)
_CITATION_REMOVED = (
    "We could not re-confirm one or more official sources, so we left them out of the "
    "reasoning rather than quote something we cannot stand behind."
)
_NO_PLAN = (
    "We could not put together a list of next steps this time. The programs below still "
    "link to their own official instructions."
)


def _scrub(
    text: str,
    sources: Sequence[ProgramSource],
    urls: Sequence[str],
) -> ScrubbedText:
    return strip_unsupported_claims(text, sources, allowed_urls=urls)


def _scrub_lines(
    lines: Sequence[str],
    sources: Sequence[ProgramSource],
    urls: Sequence[str],
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for line in lines:
        result = _scrub(line, sources, urls)
        if result.text:
            kept.append(result.text)
        removed.extend(result.removed)
    return kept, removed


def _program_urls(candidate: RetrievedProgram) -> list[str]:
    urls = [candidate.program.official_url]
    if candidate.program.application_url:
        urls.append(candidate.program.application_url)
    return urls


def _clean_explanation(
    explanation: MatchExplanation,
    candidate: RetrievedProgram,
    *,
    required_caveats: Sequence[str],
) -> tuple[MatchExplanation | None, list[str], list[GuaranteeFlag]]:
    """Scrub one explanation, or reject it.

    The headline is load-bearing: it is the one line most people read, so a
    headline that claims an outcome invalidates the whole explanation. Bullets
    are independent statements, so a bad one is dropped and the rest survive.

    ``caveats`` are copied from the evaluation's limitations and are never
    scrubbed away - they are the part the user most needs and the model did not
    write them.
    """
    sources = candidate.sources
    urls = _program_urls(candidate)

    headline_flags = detect_guarantee_phrasing(explanation.headline)
    headline = _scrub(explanation.headline, sources, urls)
    if headline_flags or not headline.text:
        return None, [explanation.headline, *headline.removed], headline_flags

    why, removed_why = _scrub_lines(explanation.why_it_may_fit, sources, urls)
    still_need, removed_need = _scrub_lines(explanation.what_we_still_need, sources, urls)
    removed = [*headline.removed, *removed_why, *removed_need]
    flags = [
        flag
        for line in (*explanation.why_it_may_fit, *explanation.what_we_still_need)
        for flag in detect_guarantee_phrasing(line)
    ]

    cleaned = explanation.model_copy(
        update={
            "headline": headline.text,
            "why_it_may_fit": why,
            "what_we_still_need": still_need,
            "caveats": list(dict.fromkeys([*explanation.caveats, *required_caveats])),
        }
    )
    return cleaned, removed, flags


def _clean_plan_draft(
    draft: ActionPlanDraft,
    matches: Sequence[ProgramMatch],
    candidates: Sequence[RetrievedProgram],
) -> tuple[ActionPlanDraft | None, list[str]]:
    """Scrub a plan draft and drop anything it may not point at."""
    index = candidate_index(list(candidates))
    plannable = {match.program.slug: match for match in matches if match.status in _PLANNABLE}
    all_sources = [source for candidate in index.values() for source in candidate.sources]
    all_urls = [url for candidate in index.values() for url in _program_urls(candidate)]

    removed: list[str] = []
    title = _scrub(draft.title, all_sources, all_urls)
    summary = _scrub(draft.summary, all_sources, all_urls)
    removed.extend([*title.removed, *summary.removed])

    items: list[ActionItemDraft] = []
    for item in draft.items:
        match = plannable.get(item.program_slug)
        if match is None:
            removed.append(item.title)
            continue
        candidate = index.get(match.program_id)
        sources = candidate.sources if candidate else []
        urls = _program_urls(candidate) if candidate else []
        item_title = _scrub(item.title, sources, urls)
        item_detail = _scrub(item.detail, sources, urls)
        removed.extend([*item_title.removed, *item_detail.removed])
        if not item_title.text or not item_detail.text:
            continue
        items.append(item.model_copy(update={"title": item_title.text, "detail": item_detail.text}))

    if not items:
        return None, removed
    return (
        ActionPlanDraft(
            title=title.text or "Your next steps",
            summary=summary.text or "Start with the documents, then work down the list.",
            items=items,
        ),
        removed,
    )


async def merge_outputs(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Assemble the final matches and plan, and record what the gate removed."""
    update = entering(NODE)
    evaluations = list(state.get("evaluations") or ())
    candidates = list(state.get("candidates") or ())
    explanations = dict(state.get("explanations") or {})
    report = state.get("citation_results")
    index = candidate_index(candidates)

    rejected_ids = frozenset(item.source_id for item in (report.rejected if report else ()))
    limitations: list[str] = []
    validation: dict[str, Any] = {
        # The invariant, recorded rather than assumed: a status on a match is
        # the engine's status, copied, and ProgramMatch validates it.
        "status_source": "rule_engine",
        "explanations_offered": len(explanations),
        "explanations_kept": 0,
        "explanations_rejected": [],
        "removed_sentences": [],
        "guarantee_phrases": [],
        "rejected_source_ids": sorted(rejected_ids),
        "dropped_plan_lines": [],
        "degraded_branches": [],
    }

    kept: dict[str, MatchExplanation] = {}
    for slug, explanation in explanations.items():
        evaluation = next(
            (item for item in evaluations if item.program_slug == slug),
            None,
        )
        candidate = index.get(evaluation.program_id) if evaluation else None
        if evaluation is None or candidate is None:
            validation["explanations_rejected"].append(slug)
            continue
        cleaned, removed, flags = _clean_explanation(
            explanation, candidate, required_caveats=evaluation.limitations
        )
        validation["removed_sentences"].extend(removed)
        validation["guarantee_phrases"].extend(phrase for flag in flags for phrase in flag.phrases)
        if cleaned is None:
            validation["explanations_rejected"].append(slug)
            continue
        kept[slug] = cleaned
    validation["explanations_kept"] = len(kept)

    if validation["removed_sentences"] or validation["explanations_rejected"]:
        limitations.append(_PROSE_REMOVED)
    if rejected_ids:
        limitations.append(_CITATION_REMOVED)
    if explanations and not kept:
        validation["degraded_branches"].append("generate_explanations")
    if report is None:
        validation["degraded_branches"].append("verify_citations")

    matches = build_matches(
        evaluations,
        candidates,
        explanations=kept,
        category_order=state.get("program_categories") or (),
        exclude_source_ids=rejected_ids,
    )

    draft = state.get("plan_draft")
    if draft is None and matches:
        validation["degraded_branches"].append("generate_action_plan")
        draft = fallback_plan_draft([match for match in matches if match.status in _PLANNABLE], ())
    plan = None
    if draft is not None:
        cleaned_draft, dropped = _clean_plan_draft(draft, matches, candidates)
        validation["dropped_plan_lines"].extend(dropped)
        if cleaned_draft is not None:
            plan = deps.build_plan(state["run_id"], matches, cleaned_draft)
        else:
            limitations.append(_NO_PLAN)
    elif matches:
        limitations.append(_NO_PLAN)

    _log.info(
        "workflow.merge.done",
        node=NODE,
        matches=len(matches),
        potential=sum(1 for match in matches if match.status is MatchStatus.POTENTIAL_MATCH),
        explanations_kept=len(kept),
        explanations_rejected=len(validation["explanations_rejected"]),
        removed_sentences=len(validation["removed_sentences"]),
        rejected_sources=len(rejected_ids),
        plan_items=len(plan.items) if plan else 0,
        shared_documents=len(plan.shared_documents) if plan else 0,
        degraded=validation["degraded_branches"],
    )

    update["matches"] = matches
    update["plan"] = plan
    update["shared_documents"] = list(plan.shared_documents) if plan else []
    update["validation"] = validation
    if limitations:
        update["limitations"] = limitations
    return update
