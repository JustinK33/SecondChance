"""Which documents the rules imply, and how five programs' asks become one.

Why this is derived from the rules rather than written per program
-----------------------------------------------------------------
A user told they might qualify for five programs does not need five
instruction sets that each begin "gather proof of address". They need one list
of documents and five applications. The only trustworthy way to produce that
list is from the conditions that were actually evaluated: a rule that checked
the state implies proof of address, a rule that checked income implies proof of
income, and a rule that checked nothing implies nothing. Nothing here is
invented or model-written, so a document requirement cannot be fabricated.

The dedup derivation
--------------------
Every requirement collapses through two deterministic steps.

1. **Condition to subject.** Each evidence leaf maps to a canonical subject
   phrase by its ``kind`` (and, for flags and categorical fields, by which
   field it read) using :data:`SUBJECT_BY_CONDITION` and
   :data:`SUBJECT_BY_FLAG`. The subject is the canonical requirement - "proof
   of address" - never a display title, so wording differences between programs
   cannot defeat the collapse.
2. **Subject to key.** ``dedup_key = make_dedup_key(ActionItemKind.DOCUMENT,
   subject)``, which is ``"document:{alias(slugify(subject))}"`` where the
   alias table is :data:`app.schemas.plan.DOCUMENT_SUBJECT_ALIASES`. That is
   the same function and the same table the action-plan builder uses, so a
   requirement found here and an item created there carry identical keys and
   group together without a second rule.

Items with the same key merge and record every program that asked; items with
different keys stay separate, because "proof of income" and "proof of
enrollment" are two trips to two places even when one program asks for both.

What is excluded, and why
-------------------------
* Conditions that resolved ``FALSE`` contribute nothing. Their branch of the
  rule set already failed, so the document could not help.
* Programs at ``UNLIKELY_MATCH`` contribute nothing. Sending someone to find
  paperwork for a program they almost certainly cannot get is the clearest
  possible way to waste their afternoon.
* Nothing here ever asks for a Social Security card, a bank statement for
  account verification, or any identifier the rules do not consume. The subject
  tables are a fixed, reviewed list, so that is a property of the code rather
  than a promise.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from app.eligibility.engine import leaf_evidence
from app.schemas import (
    ActionItemDraft,
    ActionItemKind,
    CategoricalProfileField,
    ConditionEvidence,
    EligibilityEvaluation,
    MatchStatus,
    ProfileFlag,
    Trilean,
    make_dedup_key,
)

__all__ = [
    "SUBJECT_BY_CONDITION",
    "SUBJECT_BY_FLAG",
    "DocumentRequirement",
    "document_action_drafts",
    "required_documents",
    "subject_for_evidence",
]

#: Canonical document subject implied by each condition kind that implies one.
#: Combinators imply nothing of their own - their children do - and
#: ``household_size`` is here because every program that counts a household
#: asks you to show who is in it.
SUBJECT_BY_CONDITION: Final[Mapping[str, str]] = {
    "geography": "proof of address",
    "income_fpl": "proof of income",
    "income_absolute": "proof of income",
    "student_status": "proof of enrollment",
    "household_size": "proof of household members",
    "age": "photo id",
}

#: Canonical document subject implied by a rule testing a particular flag.
#: A flag with no entry implies no paperwork: "do you have children" is
#: answered on the form, not evidenced at the counter.
SUBJECT_BY_FLAG: Final[Mapping[ProfileFlag, str]] = {
    ProfileFlag.RECEIVES_SNAP: "benefit award letter",
    ProfileFlag.RECEIVES_WIC: "benefit award letter",
    ProfileFlag.RECEIVES_MEDICAID: "benefit award letter",
    ProfileFlag.RECEIVES_TANF: "benefit award letter",
    ProfileFlag.RECEIVES_SSI: "benefit award letter",
    ProfileFlag.RECEIVES_SSDI: "benefit award letter",
    ProfileFlag.RECEIVES_UNEMPLOYMENT: "benefit award letter",
    ProfileFlag.RECEIVES_FREE_SCHOOL_MEALS: "benefit award letter",
    ProfileFlag.IS_PREGNANT: "proof of pregnancy",
    ProfileFlag.IS_RUTGERS_AFFILIATED: "proof of enrollment",
    ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE: "utility shutoff notice",
    ProfileFlag.HAS_EVICTION_NOTICE: "eviction notice",
    ProfileFlag.FILED_TAXES_LAST_YEAR: "tax return",
}

#: Canonical document subject implied by a rule testing a categorical field.
_SUBJECT_BY_CATEGORICAL: Final[Mapping[CategoricalProfileField, str]] = {
    CategoricalProfileField.STATE: "proof of address",
    CategoricalProfileField.COUNTY: "proof of address",
    CategoricalProfileField.STUDENT_STATUS: "proof of enrollment",
    CategoricalProfileField.EMPLOYMENT_STATUS: "proof of income",
    CategoricalProfileField.HOUSING_SITUATION: "lease or mortgage",
}

#: Author-written copy for each subject: an imperative title and the "why".
#: Keyed by subject so two programs asking for the same thing also get the same
#: sentence, which is what makes the merged item read naturally.
_COPY: Final[Mapping[str, tuple[str, str]]] = {
    "proof of address": (
        "Find something that shows your address",
        "A recent utility bill, lease or piece of official mail works. Programs use it "
        "to confirm you live in the area they serve.",
    ),
    "proof of income": (
        "Gather recent proof of your income",
        "Recent pay stubs, a benefit letter or last year's tax return. If your income "
        "changes month to month, bring what you have and say so.",
    ),
    "proof of enrollment": (
        "Get proof that you are enrolled",
        "A class schedule, enrollment verification or student ID shows you are a current "
        "student.",
    ),
    "proof of household members": (
        "Make a list of everyone in your household",
        "Write down who lives with you. Some programs ask for a birth certificate or "
        "school letter for children.",
    ),
    "photo id": (
        "Bring a photo ID",
        "A driver's licence, state ID or any government-issued photo ID.",
    ),
    "benefit award letter": (
        "Find your benefit award letter",
        "The letter telling you a benefit was approved. Already being in one program "
        "often makes another one quicker.",
    ),
    "proof of pregnancy": (
        "Ask your clinic for proof of pregnancy",
        "A short note from a doctor or clinic is enough.",
    ),
    "utility shutoff notice": (
        "Keep the shut-off notice",
        "The notice from the utility company is what moves an application to the front "
        "of the queue.",
    ),
    "eviction notice": (
        "Keep the eviction notice and any court papers",
        "Emergency housing help is based on the notice, so keep every page.",
    ),
    "tax return": (
        "Find last year's tax return",
        "A copy of your federal return, or the transcript you can download from the IRS.",
    ),
    "lease or mortgage": (
        "Find your lease or mortgage statement",
        "It shows what you pay for housing, which utility and housing programs need.",
    ),
}


@dataclass(frozen=True, slots=True)
class DocumentRequirement:
    """One document, and every program that needs it.

    ``dedup_key`` is the collapse identity (see the module docstring) and is
    identical to the key the action-plan builder computes for the same subject,
    so this object can be handed straight to plan generation.
    """

    dedup_key: str
    subject: str
    title: str
    detail: str
    program_ids: tuple[uuid.UUID, ...] = ()
    program_slugs: tuple[str, ...] = ()
    condition_kinds: tuple[str, ...] = ()

    @property
    def is_shared(self) -> bool:
        """Whether two or more distinct programs ask for this document."""
        return len(self.program_ids) >= 2


def subject_for_evidence(evidence: ConditionEvidence) -> str | None:
    """Canonical document subject implied by one evidence leaf, if any.

    Returns ``None`` for conditions that imply no paperwork, including every
    combinator and any leaf that resolved ``FALSE``.
    """
    if evidence.children or evidence.outcome is Trilean.FALSE:
        return None
    if evidence.kind == "flag":
        flag = _flag_from_fields(evidence)
        return None if flag is None else SUBJECT_BY_FLAG.get(flag)
    if evidence.kind == "categorical":
        field = _categorical_from_fields(evidence)
        return None if field is None else _SUBJECT_BY_CATEGORICAL.get(field)
    return SUBJECT_BY_CONDITION.get(evidence.kind)


def _flag_from_fields(evidence: ConditionEvidence) -> ProfileFlag | None:
    for name in evidence.profile_fields_used:
        if name.startswith("flags."):
            try:
                return ProfileFlag(name.removeprefix("flags."))
            except ValueError:
                return None
    return None


def _categorical_from_fields(evidence: ConditionEvidence) -> CategoricalProfileField | None:
    for name in evidence.profile_fields_used:
        try:
            return CategoricalProfileField(name)
        except ValueError:
            continue
    return None


def _relevant_evaluations(
    evaluations: Sequence[EligibilityEvaluation],
) -> Iterator[EligibilityEvaluation]:
    for evaluation in evaluations:
        if evaluation.status is not MatchStatus.UNLIKELY_MATCH:
            yield evaluation


@dataclass(slots=True)
class _Bucket:
    subject: str
    program_ids: dict[uuid.UUID, None]
    program_slugs: dict[str, None]
    condition_kinds: dict[str, None]


def required_documents(
    evaluations: Sequence[EligibilityEvaluation],
) -> list[DocumentRequirement]:
    """The deduplicated document list implied by a set of evaluations.

    Ordered by how many programs need each document (most shared first), then
    by ``dedup_key``, so the list is deterministic and the one errand that
    serves four applications is at the top.
    """
    buckets: dict[str, _Bucket] = {}

    for evaluation in _relevant_evaluations(evaluations):
        for root in evaluation.evidence:
            for node in leaf_evidence(root):
                subject = subject_for_evidence(node)
                if subject is None:
                    continue
                key = make_dedup_key(ActionItemKind.DOCUMENT, subject)
                bucket = buckets.get(key)
                if bucket is None:
                    bucket = _Bucket(subject, {}, {}, {})
                    buckets[key] = bucket
                bucket.program_ids[evaluation.program_id] = None
                bucket.program_slugs[evaluation.program_slug] = None
                bucket.condition_kinds[node.kind] = None

    requirements = [
        DocumentRequirement(
            dedup_key=key,
            subject=bucket.subject,
            title=_COPY[bucket.subject][0],
            detail=_COPY[bucket.subject][1],
            program_ids=tuple(bucket.program_ids),
            program_slugs=tuple(bucket.program_slugs),
            condition_kinds=tuple(sorted(bucket.condition_kinds)),
        )
        for key, bucket in buckets.items()
    ]
    requirements.sort(key=lambda item: (-len(item.program_ids), item.dedup_key))
    return requirements


def document_action_drafts(
    evaluations: Sequence[EligibilityEvaluation],
) -> list[ActionItemDraft]:
    """The same requirements as per-program drafts for the action-plan builder.

    The plan layer owns identity, ordering and grouping, so this hands it one
    draft per (program, subject) pair with the canonical ``subject`` already
    chosen. ``group_shared_documents`` then collapses them with the identical
    key :func:`required_documents` used, which is why the two views can never
    disagree about what is shared.
    """
    drafts: list[ActionItemDraft] = []
    for requirement in required_documents(evaluations):
        for slug in requirement.program_slugs:
            drafts.append(
                ActionItemDraft(
                    program_slug=slug,
                    kind=ActionItemKind.DOCUMENT,
                    subject=requirement.subject,
                    title=requirement.title,
                    detail=requirement.detail,
                )
            )
    return drafts
