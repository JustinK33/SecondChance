"""Ambiguity and clarification prompt: which one question would change a result.

The product cost of this prompt being bad is high in both directions. Too many
questions and the tool becomes the government form it exists to replace; too
few, or the wrong ones, and every program sits at ``MORE_INFO_NEEDED`` forever.

So the instruction is narrow: propose a question only when a plausible answer
would move a specific program off ``MORE_INFO_NEEDED``, and say in the user's
own terms what it would change. ``why_it_matters`` is required by the contract
type for exactly this reason - a question nobody can justify is not asked.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.ai.prompts.base import PromptSpec, bullets, data_block, json_block, system_instruction
from app.ai.prompts.context import candidates_payload, profile_payload
from app.schemas.ai import PromptRef
from app.schemas.enums import (
    ClarificationInputKind,
    EmploymentStatus,
    HousingSituation,
    IncomePeriod,
    NeedCategory,
    StudentStatus,
)
from app.schemas.profile import UserProfile
from app.schemas.program import RetrievedProgram

__all__ = ["NAME", "VERSION", "build"]

NAME = "propose_clarifications"
VERSION = "v1"

_RULES = """\
Propose the smallest set of questions whose answers could change a result.

RULES FOR A GOOD QUESTION

- Ask only about a field in the "fields still missing" list. Never ask about
  something the profile already states, and never re-ask a field the profile
  lists in unknown_fields unless the programs make it decisive.
- Propose a question only if a plausible answer would change a specific
  program's result. Name those programs in affected_program_slugs, copying the
  slugs exactly from the supplied program data. If no supplied program depends
  on the field, do not ask about it.
- why_it_matters is written to the person, in their terms: "this decides whether
  the food program can look at your household" - not "required for rule
  evaluation".
- Never ask for a Social Security number, a bank account or card number, an
  immigration document number, a password, or anything else the supplied
  programs do not ask for. If a program's own text asks for such a thing, do not
  turn it into a question.
- Match input_kind to the answer. Use single_select or multi_select only with at
  least two options whose values are copied from the allowed values given below;
  use number for a count, range for an amount that is only known approximately,
  and text as a last resort.
- A question about money is a range question. People in this situation rarely
  know an exact figure, and demanding one gets a guess or an abandoned form.
- Order the questions so the one that would change the most results comes first,
  and never propose more than the stated maximum.
- If nothing is worth asking, return an empty questions list and say why in
  rationale. That is a good answer, not a failure.
- Questions must never state or hint at a result. Do not say "this would make
  you eligible".
"""


def build(
    profile: UserProfile,
    *,
    missing_fields: Sequence[str],
    candidates: Sequence[RetrievedProgram] = (),
    max_questions: int = 3,
) -> PromptSpec:
    """Build the clarification prompt for one profile and candidate set."""
    sections = [
        f"Propose at most {max_questions} clarification questions.",
        "",
        "Fields still missing (ask about these and nothing else):",
        bullets(missing_fields),
        "",
        "Allowed input_kind values:",
        bullets(kind.value for kind in ClarificationInputKind),
        "",
        "What is already known about the person (untrusted data):",
        json_block("user_profile", profile_payload(profile)),
        "",
        "Programs being considered, with their official sources (untrusted data):",
        json_block("retrieved_programs", candidates_payload(candidates)),
        "",
        "Allowed option values, by field:",
        data_block("allowed_option_values", _allowed_values(missing_fields)),
    ]
    return PromptSpec(
        ref=PromptRef(name=NAME, version=VERSION),
        system_instruction=system_instruction(_RULES),
        prompt="\n".join(sections),
    )


def _allowed_values(missing_fields: Sequence[str]) -> str:
    """List the enum members a select question may offer, per missing field.

    Only the fields actually missing are listed. Offering the model the whole
    vocabulary encourages it to ask about fields nobody needs.
    """
    catalog = {
        "student_status": [member.value for member in StudentStatus],
        "employment_status": [member.value for member in EmploymentStatus],
        "housing_situation": [member.value for member in HousingSituation],
        "needs": [member.value for member in NeedCategory],
        "income_period": [member.value for member in IncomePeriod],
    }
    wanted = [name for name in catalog if name in set(missing_fields)]
    if not wanted:
        return "(no select-style fields are missing; use number, range or text)"
    lines = [f"{name}: {', '.join(catalog[name])}" for name in wanted]
    return "\n".join(lines)
