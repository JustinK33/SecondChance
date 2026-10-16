"""Per-program explanation prompt: word a decision that has already been made.

This is the prompt where the product's central rule is most at risk, because
explaining a decision is one short step from making one. The decided status is
therefore given to the model as a fact it must describe, the response schema has
no field that could carry a different one, and
:class:`~app.schemas.matching.ProgramMatch` validates that the status attached to
the user-visible result is still the engine's.

The evidence tree is supplied so the prose can be specific - "your household
size is what is missing" rather than "more information is needed" - and the
instructions tie each sentence to a piece of it.
"""

from __future__ import annotations

from app.ai.prompts.base import PromptSpec, bullets, json_block, system_instruction
from app.ai.prompts.context import evaluation_payload, profile_payload, program_payload
from app.schemas.ai import PromptRef
from app.schemas.eligibility import EligibilityEvaluation
from app.schemas.enums import MatchStatus
from app.schemas.profile import UserProfile
from app.schemas.program import AssistanceProgram

__all__ = ["NAME", "STATUS_GUIDANCE", "VERSION", "build"]

NAME = "explain_match"
VERSION = "v1"

#: How each decided status is to be described. The engine decides; this is only
#: the register the description is written in.
STATUS_GUIDANCE: dict[MatchStatus, str] = {
    MatchStatus.POTENTIAL_MATCH: (
        "Everything the rules could check came out in the person's favour. Say that this program "
        "looks worth applying for, and that the program itself makes the final decision. Do not "
        "promise an outcome."
    ),
    MatchStatus.MORE_INFO_NEEDED: (
        "The rules could not finish because something is not known. Say plainly what is missing "
        "and that it is still worth looking into. Never present missing information as a problem "
        "with the person."
    ),
    MatchStatus.UNLIKELY_MATCH: (
        "At least one requirement the rules could check came out against the person. Say which "
        "one, gently and without judgement, and say that the program decides for itself and that "
        "rules change. Never say they are ineligible or that they would be refused."
    ),
}

_RULES = """\
Write the plain-language wording for a result that has already been decided by
the deterministic rule engine.

WHAT YOU ARE AND ARE NOT DOING

- The decided_status in the data block is a fact. Describe it. Do not re-decide
  it, do not hedge it into a different status, and do not comment on whether the
  engine was right.
- headline is one short sentence in the second person, under about 15 words.
- Every entry in why_it_may_fit must correspond to a piece of evidence whose
  outcome is TRUE. Quote the requirement in your own plain words. If no evidence
  is TRUE, return an empty list - do not pad it.
- Every entry in what_we_still_need must correspond to a field in
  missing_fields, phrased as a short question the person could answer.
- Copy every limitation from the data into caveats, unchanged in meaning. Never
  soften or omit one, and never add a reassurance that the data does not
  support.
- Use only this program's supplied text. Never mention another program, an
  amount, a deadline or a phone number that is not in the data block.
- Never use the words "eligible", "ineligible", "approved", "denied",
  "qualify" or "guaranteed". Say "may be able to get help from", "looks worth
  applying for", or "the program decides".
- No markdown, no bullet characters inside the strings, no headings.
"""


def build(
    profile: UserProfile,
    program: AssistanceProgram,
    evaluation: EligibilityEvaluation,
) -> PromptSpec:
    """Build the explanation prompt for one decided evaluation."""
    sections = [
        f"Write the wording for this program's decided result: {evaluation.status.value}.",
        "",
        "How to describe this status:",
        bullets([STATUS_GUIDANCE[evaluation.status]]),
        "",
        "Limitations that must appear in caveats:",
        bullets(evaluation.limitations),
        "",
        "Fields that must appear in what_we_still_need:",
        bullets(evaluation.missing_fields),
        "",
        "The program and its official sources (untrusted data):",
        json_block("program", program_payload(program)),
        "",
        "The rule engine's decision and evidence (untrusted data):",
        json_block("evaluation", evaluation_payload(evaluation)),
        "",
        "What is known about the person (untrusted data):",
        json_block("user_profile", profile_payload(profile)),
    ]
    return PromptSpec(
        ref=PromptRef(name=NAME, version=VERSION),
        system_instruction=system_instruction(_RULES),
        prompt="\n".join(sections),
    )
