"""Action plan prompt: turn decided matches into an ordered list of next steps.

The plan is the product's actual output, and the thing that makes it useful is
deduplication: five programs that each want proof of address are one errand, not
five. The model proposes ``subject`` values and the server collapses them
through :func:`~app.schemas.plan.make_dedup_key`, so the prompt's job is to get
the *canonical* subject wording right rather than to group anything itself.

Links are the other risk. A fabricated application URL is the single most
harmful thing this system could emit, so the prompt restricts URLs to the exact
list supplied and :meth:`~app.ai.schemas.ActionPlanResponse.to_draft` strips any
URL that is not in it.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.ai.prompts.base import PromptSpec, bullets, json_block, system_instruction
from app.ai.prompts.context import catalog_urls, match_payload, profile_payload
from app.schemas.ai import PromptRef
from app.schemas.enums import ActionItemKind
from app.schemas.matching import ProgramMatch
from app.schemas.plan import DOCUMENT_SUBJECT_ALIASES
from app.schemas.profile import UserProfile

__all__ = ["NAME", "VERSION", "build"]

NAME = "draft_action_plan"
VERSION = "v1"

_RULES = """\
Draft the next steps for the programs in the data block.

ORDER AND CONTENT

- Lead with the step that unblocks the most. A document that several programs
  ask for comes before any single application; something with a deadline or a
  shut-off notice comes before anything else.
- One item per concrete step. kind=DOCUMENT for something to find or request,
  APPLY for submitting an application, CONFIRM_INFO for checking a fact,
  CONTACT for calling or visiting, FOLLOW_UP for checking back later, LEARN for
  reading something official.
- subject is the canonical requirement, lower case, no program name: "proof of
  address", "photo id", "proof of income", "pay stubs". Use the exact canonical
  wording from the list below when one applies. The server deduplicates on
  subject, so wording it consistently across programs is what collapses five
  errands into one.
- title is imperative and specific: "Find a utility bill from the last 90 days".
  detail says what to do and why in two or three plain sentences.
- program_slug must be copied exactly from the supplied data. Never invent a
  program.
- url must be one of the allowed URLs listed below, or null. Never write a URL
  that is not in that list, and never reconstruct one you remember.
- Only describe a requirement the supplied program text actually states. If a
  program's requirements are not in the data, use CONTACT or LEARN to point the
  person at the official page instead of guessing at documents.
- A program whose result still needs information gets a CONFIRM_INFO item for
  what is missing, not an APPLY item.
- summary is two or three sentences on what to do first. Never promise an
  outcome and never use the words "eligible", "ineligible" or "qualify".
- If there is genuinely nothing to do, return an empty items list.
"""


def build(profile: UserProfile, matches: Sequence[ProgramMatch]) -> PromptSpec:
    """Build the action-plan prompt for one set of decided matches."""
    canonical = sorted(set(DOCUMENT_SUBJECT_ALIASES.values()))
    sections = [
        "Draft an action plan for the decided results below.",
        "",
        "Allowed kind values:",
        bullets(kind.value for kind in ActionItemKind),
        "",
        "Canonical document subjects (use this wording when it applies):",
        bullets(subject.replace("-", " ") for subject in canonical),
        "",
        "Allowed URLs (use one of these or null):",
        bullets(catalog_urls(matches)),
        "",
        "Decided results, with each program's own text (untrusted data):",
        json_block("matches", [match_payload(match) for match in matches]),
        "",
        "What is known about the person (untrusted data):",
        json_block("user_profile", profile_payload(profile)),
    ]
    return PromptSpec(
        ref=PromptRef(name=NAME, version=VERSION),
        system_instruction=system_instruction(_RULES),
        prompt="\n".join(sections),
    )
