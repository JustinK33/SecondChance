"""``extract_intake``: a free-text narrative becomes structured, optional fields.

The only node in the graph that may fail the whole run. Everything downstream
is a function of the profile, so a failure here leaves nothing honest to say -
and recording "no programs found" when the truth is "we never read your
description" would be the worst possible lie for this product to tell. The
exception propagates, :func:`~app.workflows.graph.run_workflow` records a
user-safe failure, and the checkpoint stays resumable so the user never
re-types anything.

Two product rules are enforced upstream of this node rather than inside it, and
are worth naming because they are the reason it is this short:

* Unknown stays unknown. ``ExtractedProfile`` has no required field and
  ``to_profile_fields`` drops anything below the confidence floor, so a guess
  cannot reach the rule engine.
* Sensitive attributes are never inferred. The extraction response schema in
  ``app.ai.schemas`` is a closed set of ``PROFILE_FIELD_NAMES`` and
  ``ProfileFlag`` members, so there is no field for the model to write one to.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.workflows.deps import WorkflowDeps
from app.workflows.nodes.base import entering, log_failure
from app.workflows.state import NavigatorState, StateUpdate

__all__ = ["NODE", "extract_intake"]

NODE = "extract_intake"

_log = get_logger(__name__)

#: Said when the deterministic reader ran instead of the model. The product is
#: fully usable that way, but a user is entitled to know which one read their
#: words.
_FALLBACK_LIMITATION = (
    "We read your description with our built-in reader rather than the language model, "
    "so we may have missed a detail. You can add anything we missed."
)


async def extract_intake(state: NavigatorState, *, deps: WorkflowDeps) -> StateUpdate:
    """Extract profile fields from the narrative with the model seam."""
    update = entering(NODE)
    narrative = state.get("narrative", "")

    try:
        result = await deps.gemini.extract_profile(
            narrative,
            prefilled=state.get("prefilled"),
            request_id=state.get("request_id"),
        )
    except BaseException as exc:
        log_failure(NODE, exc, recoverable=False)
        raise

    extracted = result.value
    limitations: list[str] = []
    if result.usage.source == "fallback":
        limitations.append(_FALLBACK_LIMITATION)
    if extracted.ambiguities:
        limitations.append(
            "Some of what you wrote could be read more than one way, so we may ask you about it."
        )

    # Counts and names only. The narrative, the spans and the values are
    # personal content and never reach a log line.
    _log.info(
        "workflow.extract.done",
        node=NODE,
        source=result.usage.source,
        fields=len(extracted.field_evidence),
        unknown=len(extracted.unknown_fields),
        ambiguities=len(extracted.ambiguities),
    )

    update["extracted"] = extracted
    update["extracted_fields"] = extracted.to_profile_fields(min_confidence=deps.min_confidence)
    update["model_usage"] = [result.usage]
    if limitations:
        update["limitations"] = limitations
    return update
