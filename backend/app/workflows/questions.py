"""Deterministic clarification questions for the *first* pause in the graph.

The second pause is easy: by then the rule engine has produced evaluations and
:func:`app.eligibility.engine.missing_field_questions` derives the questions
from the conditions that actually resolved ``UNKNOWN``, ranked by how many
programs an answer would unblock. That function is the one the graph uses for
``request_targeted_clarification``, and it is the better source because it can
prove an answer would change something.

The first pause happens *before* retrieval, so there are no evaluations to
derive anything from. The engine's per-field copy lives in a private table
(``app.eligibility.engine._QUESTION_TEMPLATES``) and reaching into another
module's private name across an ownership boundary is worse than restating
three questions, so the small table below is written here. See the handoff
note asking retrieval to export a public ``question_for_field``, after which
this module becomes a one-line delegation.

Two properties are structural rather than reviewed:

* ``field`` is validated against
  :data:`~app.schemas.profile.PROFILE_FIELD_NAMES`, a closed set with no SSN,
  no bank details and no immigration status in it - so there is nowhere to put
  a question we must never ask.
* ``why_it_matters`` is required by
  :class:`~app.schemas.profile.ClarificationQuestion`. A question whose author
  cannot say what the answer would change should not be asked at all.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Final

from app.eligibility.engine import FIELD_PRIORITY, MAX_CLARIFICATION_QUESTIONS
from app.schemas import (
    PROFILE_FIELD_NAMES,
    ClarificationAnswer,
    ClarificationInputKind,
    ClarificationOption,
    ClarificationQuestion,
    StudentStatus,
    UserProfile,
    clarification_question_id,
)

__all__ = [
    "MATERIAL_FIELDS",
    "SENSITIVE_TERMS",
    "declines_for_unanswered",
    "intake_questions",
    "material_missing_fields",
    "not_already_declined",
]

#: Fields whose absence genuinely changes which programs can be assessed.
#: Almost every rule set in the catalog reads geography, household size or
#: income, so leaving one of these unknown leaves most programs stuck at
#: ``MORE_INFO_NEEDED``. Anything outside this tuple is not worth a question
#: before we even know which programs are relevant - that is what the second,
#: evidence-driven pass is for.
MATERIAL_FIELDS: Final[tuple[str, ...]] = ("state", "household_size", "income")

#: Belt and braces. Nothing in :data:`MATERIAL_FIELDS` could produce one of
#: these, so a failure here means somebody reworded a template badly.
SENSITIVE_TERMS: Final[frozenset[str]] = frozenset(
    {
        "social security number",
        "ssn",
        "bank account",
        "routing number",
        "account number",
        "credit card",
        "password",
        "immigration status",
        "date of birth",
    }
)


@dataclass(frozen=True, slots=True)
class _Template:
    question: str
    why_it_matters: str
    input_kind: ClarificationInputKind
    options: tuple[tuple[str, str], ...] = ()
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    affected: tuple[str, ...] = dataclass_field(default=())


_TEMPLATES: Final[dict[str, _Template]] = {
    "state": _Template(
        question="Which state do you live in?",
        why_it_matters="Nearly all of this help is only open to people living in New Jersey.",
        input_kind=ClarificationInputKind.TEXT,
    ),
    "county": _Template(
        question="Which New Jersey county do you live in?",
        why_it_matters="Some help is run county by county, so this changes what is available.",
        input_kind=ClarificationInputKind.TEXT,
    ),
    "household_size": _Template(
        question="How many people live in your household, counting yourself?",
        why_it_matters="Income limits rise with household size, so this can change the answer.",
        input_kind=ClarificationInputKind.NUMBER,
        unit="people",
        min_value=1,
        max_value=20,
    ),
    "income": _Template(
        question="Roughly how much money does your household bring in each month?",
        why_it_matters=(
            "A rough range is enough. Several programs set a monthly income limit and we "
            "cannot tell yet which side of it you are on."
        ),
        input_kind=ClarificationInputKind.RANGE,
        unit="dollars per month",
        min_value=0,
    ),
    "student_status": _Template(
        question="Are you enrolled as a student right now?",
        why_it_matters="Student rules change which food and emergency aid you can get.",
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=(
            (StudentStatus.NOT_A_STUDENT.value, "I am not a student"),
            (StudentStatus.UNDERGRAD_FULL_TIME.value, "Undergraduate, full time"),
            (StudentStatus.UNDERGRAD_PART_TIME.value, "Undergraduate, part time"),
            (StudentStatus.GRADUATE.value, "Graduate or professional student"),
            (StudentStatus.OTHER.value, "Something else"),
        ),
    ),
}


def _is_safe(template: _Template) -> bool:
    haystack = f"{template.question} {template.why_it_matters}".casefold()
    return not any(term in haystack for term in SENSITIVE_TERMS)


def material_missing_fields(profile: UserProfile) -> list[str]:
    """Material fields this profile does not determine and has not declined.

    A field the user was already given the chance to supply is in
    ``profile.unknown_fields`` and is deliberately not returned: asking the same
    question twice is how a product like this starts to feel like a government
    form.
    """
    missing: list[str] = []
    for name in MATERIAL_FIELDS:
        if name in profile.unknown_fields:
            continue
        if profile.is_unknown(name):
            missing.append(name)
    return missing


def intake_questions(
    profile: UserProfile,
    missing_fields: Sequence[str],
    *,
    run_id: uuid.UUID,
    limit: int = MAX_CLARIFICATION_QUESTIONS,
) -> list[ClarificationQuestion]:
    """Questions for the pre-retrieval pause, ranked and capped.

    Ranking follows :data:`app.eligibility.engine.FIELD_PRIORITY` so the order a
    user sees here matches the order the evidence-driven pass would use. The
    result is a pure function of ``(profile, missing_fields, run_id)``, which is
    what lets the resume test assert on the exact interrupt payload.
    """

    def rank(name: str) -> tuple[int, str]:
        return (
            FIELD_PRIORITY.index(name) if name in FIELD_PRIORITY else len(FIELD_PRIORITY),
            name,
        )

    wanted = [
        name
        for name in dict.fromkeys(missing_fields)
        if name in _TEMPLATES and name in PROFILE_FIELD_NAMES and profile.is_unknown(name)
    ]
    questions: list[ClarificationQuestion] = []
    for name in sorted(wanted, key=rank):
        template = _TEMPLATES[name]
        if not _is_safe(template):  # pragma: no cover - guards a bad reword
            continue
        questions.append(
            ClarificationQuestion(
                id=clarification_question_id(run_id, name),
                field=name,
                question=template.question,
                input_kind=template.input_kind,
                options=[
                    ClarificationOption(value=value, label=label)
                    for value, label in template.options
                ],
                why_it_matters=template.why_it_matters,
                affected_program_ids=list(template.affected),
                is_required=False,
                unit=template.unit,
                min_value=template.min_value,
                max_value=template.max_value,
            )
        )
        if len(questions) >= limit:
            break
    return questions


def declines_for_unanswered(
    questions: Sequence[ClarificationQuestion],
    answers: Sequence[ClarificationAnswer],
) -> list[ClarificationAnswer]:
    """Turn "shown it, did not answer it" into the decline it already is.

    Every clarification question carries ``is_required=False``, so a person who
    submits two of the three answers has *used* that affordance: they looked at
    the third question and moved on. The resume payload cannot express that on
    its own - a question nobody touched is simply absent from it - so a gate
    that re-derives its questions from the profile sees the field still missing
    and asks the identical question again, with the identical deterministic
    ``question_id``, in the very next round. From the user's seat, skipping did
    not stick.

    This closes that gap by writing the decline down in the one place that
    already means it: a ``skipped`` answer, which
    :func:`~app.workflows.profile_merge.assemble_profile` records in
    ``UserProfile.unknown_fields`` and which :func:`not_already_declined` then
    honours. Nothing is inferred about the *value* of the field - it stays
    unknown, and the programs it blocks stay at ``MORE_INFO_NEEDED``.

    An answer addresses a question when it carries that question's id or names
    that question's field. The field is enough on its own because
    :func:`~app.schemas.common.clarification_question_id` is a pure function of
    ``(run_id, field)``, so a client that answers the right field with a stale
    or hand-built id has still answered the question.

    :returns: one skipped answer per unaddressed question, in the order the
        questions were presented. Empty when every question was addressed.
    """
    addressed_ids = {answer.question_id for answer in answers}
    addressed_fields = {answer.field for answer in answers}
    return [
        ClarificationAnswer(question_id=question.id, field=question.field, skipped=True)
        for question in questions
        if question.id not in addressed_ids and question.field not in addressed_fields
    ]


def not_already_declined(
    questions: Sequence[ClarificationQuestion], profile: UserProfile
) -> list[ClarificationQuestion]:
    """Drop questions about fields the user has already declined to answer.

    ``UserProfile.unknown_fields`` records a field the user was asked about and
    skipped - whether they said so explicitly or simply left an optional
    question alone (see :func:`declines_for_unanswered`). Asking again is the
    behaviour that turns a two-question clarification into an interrogation,
    and the user already gave their answer - it was "I would rather not say".
    The affected programs stay at ``MORE_INFO_NEEDED``, which is the honest
    outcome.
    """
    declined = set(profile.unknown_fields)
    return [question for question in questions if question.field not in declined]
