"""Response schemas the model is allowed to emit, and their projection onto the
shared contract.

Why these exist at all
----------------------
The types in :mod:`app.schemas` are the *product's* types. They carry server
identity (``ClarificationQuestion.id``), derived state
(``ChatResponse.is_grounded``) and cross-field invariants that a model has no
business populating. Handing one of them to ``response_schema`` would invite the
model to fill in all of it.

So the model answers in the narrower types defined here, and each one knows how
to project itself onto the contract type. The projection is where the defensive
work happens: unknown profile field names, out-of-range numbers, citations to
sources that were never supplied and URLs that were not in the catalog are
dropped, and every drop is reported back to the caller as a reason string so it
can be counted in the logs.

Two structural guarantees
-------------------------
1. **No schema here can express a** :class:`~app.schemas.enums.MatchStatus`.
   Not a status field, not a status enum, not a free-text field whose value is
   fed into one. The rule engine decides status; the model only writes language.
   ``tests/ai/test_ai_schemas.py`` walks the generated JSON schema of every
   member of :data:`MODEL_RESPONSE_SCHEMAS` and asserts the three status values
   appear nowhere in it, so the guarantee is checked rather than asserted.
2. **"I cannot determine this" is always representable.** Every schema has a
   way to answer honestly without inventing anything:
   :attr:`ExtractionResponse.unknown_fields` and ``ambiguities``, an empty
   :attr:`ClarificationResponse.questions` with a ``rationale``,
   :attr:`ExplanationResponse.what_we_still_need`, an empty
   :attr:`ActionPlanResponse.items`, and
   :attr:`ChatAnswerResponse.can_answer` = ``False``.

Leniency policy
---------------
Numeric fields are declared unbounded and clamped or dropped during projection,
rather than being bounded in the schema. A bounded schema turns "the model said
``household_size: 0``" into a whole-call validation failure; projection turns it
into one dropped field and a reason string. The strictness lives where it
belongs - on the contract types, which still refuse to hold a bad value.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.assistant import ChatResponse, Citation, FactStatement
from app.schemas.enums import (
    ActionItemKind,
    ClarificationInputKind,
    ConfidenceLevel,
    EmploymentStatus,
    HousingSituation,
    IncomePeriod,
    NeedCategory,
    ProfileFlag,
    StudentStatus,
)
from app.schemas.matching import MatchExplanation
from app.schemas.plan import ActionItemDraft, ActionPlanDraft
from app.schemas.profile import (
    PROFILE_FIELD_NAMES,
    ClarificationOption,
    ClarificationQuestion,
    ClarificationQuestionSet,
    ExtractedFlag,
    ExtractedProfile,
    FieldEvidence,
    HouseholdComposition,
)
from app.schemas.program import ProgramSource

__all__ = [
    "BANNED_STATUS_VOCABULARY",
    "CANNOT_ANSWER_TEXT",
    "MODEL_OUTPUT_CONFIG",
    "MODEL_RESPONSE_SCHEMAS",
    "ActionItemDraftOut",
    "ActionPlanResponse",
    "ChatAnswerResponse",
    "CitationOut",
    "ClarificationOptionOut",
    "ClarificationQuestionOut",
    "ClarificationResponse",
    "ExplanationResponse",
    "ExtractedFlagOut",
    "ExtractionResponse",
    "FactStatementOut",
    "FieldEvidenceOut",
    "GeminiTask",
    "HouseholdCompositionOut",
]

#: Words the product never uses about a person's situation. The engine's three
#: statuses are the whole vocabulary; "eligible" and "ineligible" state a legal
#: conclusion this system is not entitled to state.
BANNED_STATUS_VOCABULARY: Final[tuple[str, ...]] = (
    "eligible",
    "ineligible",
    "ineligibility",
    "eligibility determination",
    "you qualify",
    "you do not qualify",
    "you don't qualify",
    "approved for",
    "denied",
    "guaranteed",
)

#: Used when the model answers that it cannot answer and supplies no wording.
CANNOT_ANSWER_TEXT: Final[str] = (
    "We could not find this in the official sources we have, so we cannot answer it. "
    "Contacting the program directly is the reliable way to find out."
)

_MAX_HOUSEHOLD_SIZE: Final[int] = 20
_MAX_AGE: Final[int] = 120
_MAX_PLAIN_TEXT: Final[int] = 4000
_MAX_QUESTION_CHARS: Final[int] = 300
_MIN_QUESTION_CHARS: Final[int] = 5
_MAX_SPAN_CHARS: Final[int] = 400
_MAX_OPTION_HELP_CHARS: Final[int] = 300
_MAX_RATIONALE_CHARS: Final[int] = 400
_MAX_UNIT_CHARS: Final[int] = 40
_MAX_STATE_INPUT_CHARS: Final[int] = 60
_MAX_PLAN_ITEMS: Final[int] = 60
_MAX_NEXT_STEPS: Final[int] = 5
_MAX_QUESTIONS: Final[int] = 5
_SELECT_KINDS: Final[frozenset[ClarificationInputKind]] = frozenset(
    {ClarificationInputKind.SINGLE_SELECT, ClarificationInputKind.MULTI_SELECT}
)
_STATE_CODE = re.compile(r"^[A-Za-z]{2}$")
_WHITESPACE = re.compile(r"\s+")


#: Shared config for every schema in this module.
#:
#: ``extra="ignore"`` rather than the ``extra="forbid"`` used everywhere else in
#: the codebase, for two reasons, the first of which is not negotiable:
#:
#: 1. ``extra="forbid"`` makes pydantic emit ``additionalProperties: false``,
#:    google-genai serialises that into ``generation_config.response_schema``,
#:    and the Gemini REST API rejects the whole request::
#:
#:      400 INVALID_ARGUMENT - Unknown name "additional_properties" at
#:      'generation_config.response_schema': Cannot find field.
#:
#:    Verified against the live API on 2026-10-10 with google-genai 2.29.0. A
#:    response schema carrying ``extra="forbid"`` cannot be used at all.
#: 2. It is the right policy anyway. An extra key from a chatty model is
#:    harmless - nothing reads an undeclared field - whereas forbidding it would
#:    fail the entire call over a stray ``"notes"``. Strictness belongs on the
#:    contract types, which still refuse to hold a bad value.
MODEL_OUTPUT_CONFIG: Final[ConfigDict] = ConfigDict(extra="ignore")


class GeminiTask(StrEnum):
    """The logical model calls, one per :class:`~app.schemas.contracts.GeminiPort`
    method.

    The member values double as the prompt family, the metrics label and the
    fixture file name under ``tests/fixtures/gemini/``, so the three can never
    drift apart.
    """

    EXTRACT_PROFILE = "extract_profile"
    PROPOSE_CLARIFICATIONS = "propose_clarifications"
    EXPLAIN_MATCH = "explain_match"
    DRAFT_ACTION_PLAN = "draft_action_plan"
    ANSWER_QUESTION = "answer_question"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clean(value: str | None, *, limit: int) -> str | None:
    """Collapse whitespace and truncate. ``None`` for anything empty."""
    if value is None:
        return None
    collapsed = _WHITESPACE.sub(" ", value).strip()
    if not collapsed:
        return None
    return collapsed[:limit]


def _clean_list(values: Iterable[str], *, limit: int = _MAX_PLAIN_TEXT) -> list[str]:
    out: list[str] = []
    for value in values:
        cleaned = _clean(value, limit=limit)
        if cleaned is not None and cleaned not in out:
            out.append(cleaned)
    return out


def _in_range(value: int | None, *, low: int, high: int) -> int | None:
    if value is None or not low <= value <= high:
        return None
    return value


def _normalise_for_match(value: str) -> str:
    """Case-folded, whitespace-collapsed form used for substring comparison."""
    return _WHITESPACE.sub(" ", value).strip().casefold()


def contains_banned_vocabulary(*values: str | Iterable[str]) -> str | None:
    """Return the first banned term found in ``values``, or ``None``.

    Applied to every piece of model-written prose before it is accepted. A
    partially scrubbed explanation is less trustworthy than no explanation, so
    callers treat a hit as invalid output and fall back to deterministic text.
    """
    for value in values:
        texts = [value] if isinstance(value, str) else list(value)
        for text in texts:
            lowered = text.casefold()
            for term in BANNED_STATUS_VOCABULARY:
                if term in lowered:
                    return term
    return None


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
class HouseholdCompositionOut(BaseModel):
    """Household counts, as the model reports them."""

    model_config = MODEL_OUTPUT_CONFIG

    adults: int | None = None
    children_under_18: int | None = None
    seniors_65_plus: int | None = None
    full_time_students: int | None = None
    members_with_disability: int | None = None

    def to_contract(self) -> HouseholdComposition:
        high = _MAX_HOUSEHOLD_SIZE
        return HouseholdComposition(
            adults=_in_range(self.adults, low=0, high=high),
            children_under_18=_in_range(self.children_under_18, low=0, high=high),
            seniors_65_plus=_in_range(self.seniors_65_plus, low=0, high=high),
            full_time_students=_in_range(self.full_time_students, low=0, high=high),
            members_with_disability=_in_range(self.members_with_disability, low=0, high=high),
        )


class ExtractedFlagOut(BaseModel):
    """One boolean the narrative stated explicitly."""

    model_config = MODEL_OUTPUT_CONFIG

    flag: ProfileFlag
    value: bool
    confidence: float = 1.0
    source_span: str | None = Field(
        default=None,
        description="The words in the narrative that state this. Quoted, not paraphrased.",
    )


class FieldEvidenceOut(BaseModel):
    """Where one extracted field came from, and how sure the model is."""

    model_config = MODEL_OUTPUT_CONFIG

    field: str = Field(description="Profile field name, exactly as listed in the prompt.")
    confidence: float = Field(description="0.0-1.0. Below 0.5 means 'the text hints at this'.")
    source_span: str | None = Field(default=None, description="Quoted span from the narrative.")
    reasoning: str | None = Field(default=None, description="One short clause. No speculation.")


class ExtractionResponse(BaseModel):
    """Structured profile fields read out of a narrative.

    Every field is optional because leaving a field unset is the correct answer
    far more often than guessing it. ``unknown_fields`` and ``ambiguities`` are
    how the model says so explicitly, which is what drives the clarification
    step instead of a silent default.
    """

    model_config = MODEL_OUTPUT_CONFIG

    state: str | None = Field(default=None, description="Two-letter US state code, e.g. 'NJ'.")
    county: str | None = Field(default=None, description="County name without the word 'County'.")
    student_status: StudentStatus | None = None
    household_size: int | None = Field(
        default=None, description="People in the household, including the speaker."
    )
    household_composition: HouseholdCompositionOut | None = None
    income_min_cents: int | None = Field(
        default=None, description="Low end of the stated income range, in cents."
    )
    income_max_cents: int | None = Field(
        default=None, description="High end of the stated income range, in cents."
    )
    income_period: IncomePeriod | None = Field(
        default=None, description="Required whenever an income amount is given."
    )
    employment_status: EmploymentStatus | None = None
    housing_situation: HousingSituation | None = None
    needs: list[NeedCategory] = Field(default_factory=list)
    age: int | None = None
    flags: list[ExtractedFlagOut] = Field(default_factory=list)
    field_evidence: list[FieldEvidenceOut] = Field(default_factory=list)
    unknown_fields: list[str] = Field(
        default_factory=list, description="Profile fields the narrative does not determine."
    )
    ambiguities: list[str] = Field(
        default_factory=list, description="Statements that could be read more than one way."
    )

    def to_extracted_profile(self) -> tuple[ExtractedProfile, list[str]]:
        """Project onto :class:`~app.schemas.profile.ExtractedProfile`.

        Returns the profile and a list of machine-readable drop reasons. The
        income rules are the ones that matter: an amount without a period, or a
        minimum above its maximum, is dropped rather than repaired, because
        either repair would be this layer inventing a number.
        """
        dropped: list[str] = []
        # Cleaned at full length *before* the shape check: truncating first would
        # turn "New Jersey" into "Ne", which validates as Nebraska.
        state = _clean(self.state, limit=_MAX_STATE_INPUT_CHARS)
        if state is not None and not _STATE_CODE.match(state):
            dropped.append("state:not_a_state_code")
            state = None
        county = _clean(self.county, limit=100)

        household_size = _in_range(self.household_size, low=1, high=_MAX_HOUSEHOLD_SIZE)
        if self.household_size is not None and household_size is None:
            dropped.append("household_size:out_of_range")
        age = _in_range(self.age, low=0, high=_MAX_AGE)
        if self.age is not None and age is None:
            dropped.append("age:out_of_range")

        income_min, income_max, income_dropped = self._income()
        dropped.extend(income_dropped)

        flags: list[ExtractedFlag] = []
        for flag in self.flags:
            flags.append(
                ExtractedFlag(
                    flag=flag.flag,
                    value=flag.value,
                    confidence=min(max(flag.confidence, 0.0), 1.0),
                    source_span=_clean(flag.source_span, limit=_MAX_SPAN_CHARS),
                )
            )

        evidence: list[FieldEvidence] = []
        for item in self.field_evidence:
            name = item.field.strip()
            if name not in PROFILE_FIELD_NAMES:
                dropped.append(f"field_evidence:{_normalise_for_match(name)[:40]}:unknown_field")
                continue
            evidence.append(
                FieldEvidence(
                    field=name,
                    confidence=min(max(item.confidence, 0.0), 1.0),
                    source_span=_clean(item.source_span, limit=_MAX_SPAN_CHARS),
                    reasoning=_clean(item.reasoning, limit=_MAX_QUESTION_CHARS),
                )
            )

        unknown: list[str] = []
        for name in self.unknown_fields:
            cleaned = name.strip()
            if cleaned in PROFILE_FIELD_NAMES:
                if cleaned not in unknown:
                    unknown.append(cleaned)
            else:
                dropped.append("unknown_fields:unknown_field")

        extracted = ExtractedProfile(
            state=state.upper() if state else None,
            county=county,
            student_status=self.student_status,
            household_size=household_size,
            household_composition=(
                self.household_composition.to_contract() if self.household_composition else None
            ),
            income_min_cents=income_min,
            income_max_cents=income_max,
            income_period=self.income_period if (income_min or income_max) else None,
            employment_status=self.employment_status,
            housing_situation=self.housing_situation,
            needs=list(dict.fromkeys(self.needs)),
            age=age,
            flags=flags,
            field_evidence=evidence,
            unknown_fields=unknown,
            ambiguities=_clean_list(self.ambiguities),
        )
        return extracted, dropped

    def _income(self) -> tuple[int | None, int | None, list[str]]:
        dropped: list[str] = []
        low = self.income_min_cents
        high = self.income_max_cents
        if low is not None and low < 0:
            dropped.append("income_min_cents:negative")
            low = None
        if high is not None and high < 0:
            dropped.append("income_max_cents:negative")
            high = None
        if low is not None and high is not None and low > high:
            dropped.append("income:min_above_max")
            return None, None, dropped
        if (low is not None or high is not None) and self.income_period is None:
            dropped.append("income:amount_without_period")
            return None, None, dropped
        return low, high, dropped


# ---------------------------------------------------------------------------
# clarification
# ---------------------------------------------------------------------------
class ClarificationOptionOut(BaseModel):
    """One choice offered for a select-style question."""

    model_config = MODEL_OUTPUT_CONFIG

    value: str = Field(description="Machine value; an enum member where one exists.")
    label: str = Field(description="Plain-language label the user reads.")
    help_text: str | None = None


class ClarificationQuestionOut(BaseModel):
    """One proposed question. The server assigns its id."""

    model_config = MODEL_OUTPUT_CONFIG

    field: str = Field(description="The profile field this question resolves.")
    question: str = Field(description="Plain language, one sentence, 6th-grade reading level.")
    input_kind: ClarificationInputKind
    options: list[ClarificationOptionOut] = Field(
        default_factory=list, description="Two or more for select kinds; empty otherwise."
    )
    why_it_matters: str = Field(description="What the answer would change, in the user's terms.")
    is_required: bool = False
    unit: str | None = Field(default=None, description='Unit for numeric input, e.g. "people".')
    min_value: float | None = None
    max_value: float | None = None
    affected_program_slugs: list[str] = Field(
        default_factory=list, description="Only slugs that appear in the supplied program data."
    )


class ClarificationResponse(BaseModel):
    """Questions worth asking, or an explicit "nothing worth asking"."""

    model_config = MODEL_OUTPUT_CONFIG

    questions: list[ClarificationQuestionOut] = Field(default_factory=list)
    rationale: str | None = Field(
        default=None, description="Internal note on why these questions. Not shown to the user."
    )

    def to_question_set(
        self,
        *,
        id_for: Callable[[str], str],
        allowed_fields: Sequence[str] | None = None,
        allowed_slugs: Sequence[str] = (),
        max_questions: int = 3,
    ) -> tuple[ClarificationQuestionSet, list[str]]:
        """Project onto :class:`~app.schemas.profile.ClarificationQuestionSet`.

        Drops anything the product cannot honour: a field that is not a profile
        field, a field that is not actually missing, a select question with
        fewer than two options, and any program slug that was not in the
        supplied data. Ids are assigned by ``id_for`` so the model cannot mint
        one.
        """
        dropped: list[str] = []
        permitted = set(allowed_fields) if allowed_fields is not None else set(PROFILE_FIELD_NAMES)
        slugs = set(allowed_slugs)
        seen: set[str] = set()
        questions: list[ClarificationQuestion] = []
        cap = max(0, min(max_questions, _MAX_QUESTIONS))

        for proposed in self.questions:
            if len(questions) >= cap:
                dropped.append("question:over_cap")
                continue
            field_name = proposed.field.strip()
            if field_name not in PROFILE_FIELD_NAMES:
                dropped.append("question:unknown_field")
                continue
            if field_name not in permitted:
                dropped.append("question:field_not_missing")
                continue
            if field_name in seen:
                dropped.append("question:duplicate_field")
                continue
            text = _clean(proposed.question, limit=_MAX_QUESTION_CHARS)
            why = _clean(proposed.why_it_matters, limit=_MAX_QUESTION_CHARS)
            if text is None or why is None:
                dropped.append("question:empty_text")
                continue
            if len(text) < _MIN_QUESTION_CHARS or len(why) < _MIN_QUESTION_CHARS:
                dropped.append("question:text_too_short")
                continue
            banned = contains_banned_vocabulary(text, why)
            if banned is not None:
                dropped.append("question:banned_vocabulary")
                continue
            options = self._options(proposed, dropped)
            if options is None:
                continue
            low, high = proposed.min_value, proposed.max_value
            if low is not None and high is not None and low > high:
                low, high = None, None
                dropped.append("question:bad_bounds")
            questions.append(
                ClarificationQuestion(
                    id=id_for(field_name),
                    field=field_name,
                    question=text,
                    input_kind=proposed.input_kind,
                    options=options,
                    why_it_matters=why,
                    affected_program_ids=[
                        slug
                        for slug in dict.fromkeys(proposed.affected_program_slugs)
                        if slug in slugs
                    ],
                    is_required=proposed.is_required,
                    unit=_clean(proposed.unit, limit=_MAX_UNIT_CHARS),
                    min_value=low,
                    max_value=high,
                )
            )
            seen.add(field_name)

        return (
            ClarificationQuestionSet(
                questions=questions,
                rationale=_clean(self.rationale, limit=_MAX_RATIONALE_CHARS),
            ),
            dropped,
        )

    @staticmethod
    def _options(
        proposed: ClarificationQuestionOut, dropped: list[str]
    ) -> list[ClarificationOption] | None:
        """Validate a question's options. ``None`` means drop the question."""
        options: list[ClarificationOption] = []
        values: set[str] = set()
        for option in proposed.options:
            value = _clean(option.value, limit=_MAX_UNIT_CHARS)
            label = _clean(option.label, limit=_MAX_QUESTION_CHARS)
            if value is None or label is None or value in values:
                continue
            values.add(value)
            options.append(
                ClarificationOption(
                    value=value,
                    label=label,
                    help_text=_clean(option.help_text, limit=_MAX_OPTION_HELP_CHARS),
                )
            )
        if proposed.input_kind in _SELECT_KINDS:
            if len(options) < 2:
                dropped.append("question:too_few_options")
                return None
            return options
        if options:
            dropped.append("question:options_on_non_select")
        return []


# ---------------------------------------------------------------------------
# explanation
# ---------------------------------------------------------------------------
class ExplanationResponse(BaseModel):
    """Plain-language wording for a status the engine already decided.

    There is deliberately no status field, no score and no recommendation. The
    model is given the decided status and the evidence and writes sentences
    about them.
    """

    model_config = MODEL_OUTPUT_CONFIG

    headline: str = Field(description="One short sentence. Never states a legal conclusion.")
    why_it_may_fit: list[str] = Field(
        default_factory=list, description="Only reasons backed by evidence with outcome TRUE."
    )
    what_we_still_need: list[str] = Field(
        default_factory=list, description="One line per missing field, phrased as a question."
    )
    caveats: list[str] = Field(
        default_factory=list, description="Limitations, copied. Never softened or dropped."
    )

    def to_match_explanation(
        self,
        *,
        prompt_version: str,
        required_caveats: Sequence[str] = (),
    ) -> MatchExplanation:
        """Project onto :class:`~app.schemas.matching.MatchExplanation`.

        ``required_caveats`` are the evaluation's own limitations and are merged
        in unconditionally, so a model that quietly omits one cannot make an
        assessment look more certain than it is.
        """
        headline = _clean(self.headline, limit=_MAX_PLAIN_TEXT)
        if headline is None:
            msg = "explanation has no headline"
            raise ValueError(msg)
        caveats = _clean_list([*required_caveats, *self.caveats])
        return MatchExplanation(
            headline=headline,
            why_it_may_fit=_clean_list(self.why_it_may_fit),
            what_we_still_need=_clean_list(self.what_we_still_need),
            caveats=caveats,
            generated_by="gemini",
            prompt_version=prompt_version,
        )

    def banned_term(self) -> str | None:
        """First banned term anywhere in the prose, or ``None``."""
        return contains_banned_vocabulary(
            self.headline, self.why_it_may_fit, self.what_we_still_need, self.caveats
        )


# ---------------------------------------------------------------------------
# action plan
# ---------------------------------------------------------------------------
class ActionItemDraftOut(BaseModel):
    """One proposed next step. Identity and ordering are server-side."""

    model_config = MODEL_OUTPUT_CONFIG

    program_slug: str = Field(description="Must be one of the slugs in the supplied data.")
    kind: ActionItemKind
    subject: str = Field(description='Canonical requirement, e.g. "proof of address".')
    title: str = Field(description="Imperative, one line.")
    detail: str = Field(description="What to do and why, in plain language.")
    url: str | None = Field(
        default=None, description="Only a URL that appears in the supplied data, or null."
    )


class ActionPlanResponse(BaseModel):
    """A proposed action plan. An empty plan is a valid answer."""

    model_config = MODEL_OUTPUT_CONFIG

    title: str
    summary: str = Field(description="Two or three sentences on what to do first.")
    items: list[ActionItemDraftOut] = Field(default_factory=list)

    def to_draft(
        self,
        *,
        allowed_slugs: Sequence[str],
        allowed_urls: Sequence[str] = (),
    ) -> tuple[ActionPlanDraft, list[str]]:
        """Project onto :class:`~app.schemas.plan.ActionPlanDraft`.

        An item for a program that was not supplied is dropped outright. A URL
        that is not in the catalog is stripped from an otherwise valid item,
        because a fabricated link is the one output a distressed user is most
        likely to follow.
        """
        dropped: list[str] = []
        slugs = set(allowed_slugs)
        urls = set(allowed_urls)
        title = _clean(self.title, limit=_MAX_PLAIN_TEXT)
        summary = _clean(self.summary, limit=_MAX_PLAIN_TEXT)
        if title is None or summary is None:
            msg = "action plan has no title or summary"
            raise ValueError(msg)
        if (banned := contains_banned_vocabulary(title, summary)) is not None:
            msg = f"action plan uses banned vocabulary: {banned!r}"
            raise ValueError(msg)

        items: list[ActionItemDraft] = []
        for proposed in self.items:
            if len(items) >= _MAX_PLAN_ITEMS:
                dropped.append("item:over_cap")
                continue
            slug = _clean(proposed.program_slug, limit=200)
            if slug is None or slug not in slugs:
                dropped.append("item:unknown_program")
                continue
            subject = _clean(proposed.subject, limit=_MAX_QUESTION_CHARS)
            item_title = _clean(proposed.title, limit=_MAX_PLAIN_TEXT)
            detail = _clean(proposed.detail, limit=_MAX_PLAIN_TEXT)
            if subject is None or item_title is None or detail is None:
                dropped.append("item:empty_text")
                continue
            if contains_banned_vocabulary(item_title, detail) is not None:
                dropped.append("item:banned_vocabulary")
                continue
            url = _clean(proposed.url, limit=2000)
            if url is not None and url not in urls:
                dropped.append("item:url_not_in_catalog")
                url = None
            items.append(
                ActionItemDraft(
                    program_slug=slug,
                    kind=proposed.kind,
                    subject=subject,
                    title=item_title,
                    detail=detail,
                    url=url,
                )
            )
        return ActionPlanDraft(title=title, summary=summary, items=items), dropped


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
class CitationOut(BaseModel):
    """A pointer into one of the supplied sources."""

    model_config = MODEL_OUTPUT_CONFIG

    source_id: str = Field(description="Exactly one of the source ids in the supplied data.")
    excerpt: str = Field(description="Verbatim words from that source. Never paraphrased.")


class FactStatementOut(BaseModel):
    """A sentence presented as fact. Needs at least one source id."""

    model_config = MODEL_OUTPUT_CONFIG

    text: str
    source_ids: list[str] = Field(description="Source ids from the supplied data only.")


class ChatAnswerResponse(BaseModel):
    """A source-grounded answer, or an honest refusal.

    ``can_answer`` is first in the schema on purpose: the model decides whether
    the supplied sources answer the question before it writes anything, which is
    the decision that keeps this from becoming a benefits chatbot.
    """

    model_config = MODEL_OUTPUT_CONFIG

    can_answer: bool = Field(
        description="False when the supplied sources do not answer the question."
    )
    answer: str = Field(description="Plain language. Say plainly when you do not know.")
    facts: list[FactStatementOut] = Field(default_factory=list)
    interpretation: list[str] = Field(
        default_factory=list, description="Reasoning that no source states outright."
    )
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    caveats: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)

    def banned_term(self) -> str | None:
        """First banned term anywhere in the user-visible prose, or ``None``.

        Checked by the caller before the answer is accepted. An injected
        instruction that persuades the model to say "you are eligible" is a
        product-rule violation even though chat carries no status, so the whole
        answer is rejected rather than scrubbed.
        """
        return contains_banned_vocabulary(
            self.answer,
            self.interpretation,
            self.caveats,
            self.suggested_next_steps,
            [fact.text for fact in self.facts],
        )

    def to_chat_response(
        self, *, known_sources: Sequence[ProgramSource]
    ) -> tuple[ChatResponse, list[str]]:
        """Project onto :class:`~app.schemas.assistant.ChatResponse`.

        This is the grounding gate, and it is deliberately unforgiving:

        * a citation whose ``source_id`` was not supplied is dropped, so a
          hallucinated or injected id such as ``source 999`` cannot survive;
        * ``url`` and ``title`` are taken from the stored source record, never
          from the model, so a poisoned source body cannot inject a link;
        * an excerpt that does not occur in the stored source text is dropped;
        * a fact left with no surviving citation is **demoted** into
          ``interpretation`` rather than deleted, so the content is still shown
          but is no longer presented as sourced.

        :class:`~app.schemas.assistant.ChatResponse` then derives
        ``is_grounded`` itself, so an answer that loses all its citations is
        automatically labelled ungrounded.
        """
        dropped: list[str] = []
        by_id: Mapping[str, ProgramSource] = {source.id: source for source in known_sources}

        citations: list[Citation] = []
        kept_ids: set[str] = set()
        for proposed in self.citations:
            source_id = _clean(proposed.source_id, limit=200)
            if source_id is None or source_id not in by_id:
                dropped.append("citation:unknown_source")
                continue
            if source_id in kept_ids:
                dropped.append("citation:duplicate")
                continue
            source = by_id[source_id]
            excerpt = _clean(proposed.excerpt, limit=_MAX_PLAIN_TEXT)
            if excerpt is None:
                dropped.append("citation:empty_excerpt")
                continue
            if _normalise_for_match(excerpt) not in _normalise_for_match(source.excerpt):
                dropped.append("citation:excerpt_not_found")
                continue
            citations.append(
                Citation(
                    source_id=source.id,
                    url=source.url,
                    title=source.title,
                    excerpt=excerpt,
                )
            )
            kept_ids.add(source_id)

        facts: list[FactStatement] = []
        demoted: list[str] = []
        for fact in self.facts:
            text = _clean(fact.text, limit=_MAX_PLAIN_TEXT)
            if text is None:
                dropped.append("fact:empty_text")
                continue
            supported = [sid for sid in dict.fromkeys(fact.source_ids) if sid in kept_ids]
            if not supported:
                dropped.append("fact:demoted_to_interpretation")
                demoted.append(text)
                continue
            facts.append(FactStatement(text=text, source_ids=supported))

        answer = _clean(self.answer, limit=_MAX_PLAIN_TEXT)
        if answer is None:
            answer = CANNOT_ANSWER_TEXT
        confidence = self.confidence
        if not citations and confidence in {ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM}:
            dropped.append("confidence:downgraded_ungrounded")
            confidence = ConfidenceLevel.LOW

        response = ChatResponse(
            answer=answer,
            facts=facts,
            interpretation=_clean_list([*self.interpretation, *demoted]),
            citations=citations,
            confidence=confidence,
            caveats=_clean_list(self.caveats),
            suggested_next_steps=_clean_list(self.suggested_next_steps)[:_MAX_NEXT_STEPS],
        )
        return response, dropped


#: Task -> the schema the model must answer in. Used by the client to build
#: ``GenerateContentConfig``, by the fake to validate fixtures, and by the
#: schema tests to prove no status can be emitted.
MODEL_RESPONSE_SCHEMAS: Final[dict[GeminiTask, type[BaseModel]]] = {
    GeminiTask.EXTRACT_PROFILE: ExtractionResponse,
    GeminiTask.PROPOSE_CLARIFICATIONS: ClarificationResponse,
    GeminiTask.EXPLAIN_MATCH: ExplanationResponse,
    GeminiTask.DRAFT_ACTION_PLAN: ActionPlanResponse,
    GeminiTask.ANSWER_QUESTION: ChatAnswerResponse,
}
