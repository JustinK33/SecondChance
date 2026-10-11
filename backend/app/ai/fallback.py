"""The non-model :class:`~app.schemas.contracts.GeminiPort`: templates and heuristics.

What this is
------------
A complete, deterministic implementation of the two
:class:`~app.ai.base.BaseGeminiPort` primitives, built so the product keeps
working - honestly - when no model call can succeed. That is not a hypothetical
state: ``docs/TOOLCHAIN.md`` records that ``generateContent`` currently answers
``403``/``404`` for every model this project is configured for, so this module
is the only language path the product has today.

It is *not* :class:`~app.ai.fake.FakeGeminiClient`. The fake replays recorded
fixtures, which is exactly right for a hermetic test and exactly wrong for a
real user: a replayed fixture describes the person whose narrative was recorded,
not the person who is typing. Everything this module emits is derived from the
request it was given, or is fixed template text written in this repository.

The three rules it is built around
----------------------------------
1. **Never invent a value.** A field is set only when the submitted text states
   it. Anything the text leaves open is reported in ``unknown_fields`` and
   turned into a question instead of a guess. Income is the sharpest case: an
   amount is kept only with the period the person themselves gave, and an
   hourly or weekly figure is left unset with an ambiguity recorded, because
   converting one would mean inventing the hours.
2. **Never decide anything.** The rule engine owns status. Explanations here
   word the ``decided_status`` they are handed, the response schemas cannot
   express a status at all, and no template contains the banned vocabulary.
3. **Never pass template text off as model text.** Every result reports
   ``usage.source = "fallback"`` (see :meth:`~app.schemas.ai.GeminiResult.is_fallback`),
   which is what :class:`~app.services.ai_port.DegradingGeminiPort` relabels
   on and what the interface renders its limitation from.

Why it reads the prompt
-----------------------
:class:`~app.services.ai_port.DegradingGeminiPort` degrades at the ``structured``
/ ``text`` seam, so the only thing a degraded call receives is the rendered
:class:`~app.ai.prompts.base.PromptSpec` - the same bytes the live client would
have sent. The inputs are all in there: the narrative and every payload sit in
the labelled ``<<<UNTRUSTED_DATA id="...">>>`` blocks that
:mod:`app.ai.prompts.base` writes, so this module reads them back out by id.

The upside of that seam is worth the parsing: a deterministic answer goes
through the *same* response schema, the same validation and the same projection
as a live one, so it cannot emit a citation, a URL or a status that a model
would have been refused. The blocks are sanitised before they are parsed - that
is what :func:`~app.ai.prompts.base.sanitise_untrusted` guarantees - and nothing
read out of one is ever treated as an instruction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from app.ai.base import BaseGeminiPort, CallContext, StructuredResult, TextResult
from app.ai.errors import ModelInvalidOutput
from app.ai.prompts.base import BLOCK_CLOSE, BLOCK_OPEN, PromptSpec
from app.ai.schemas import (
    CANNOT_ANSWER_TEXT,
    MODEL_RESPONSE_SCHEMAS,
    GeminiTask,
    contains_banned_vocabulary,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.ai import ModelUsage
from app.schemas.enums import (
    ActionItemKind,
    ClarificationInputKind,
    ConfidenceLevel,
    EmploymentStatus,
    HousingSituation,
    IncomePeriod,
    MatchStatus,
    NeedCategory,
    ProfileFlag,
    StudentStatus,
)

__all__ = ["FALLBACK_MODEL_ID", "DeterministicAi", "data_blocks"]

_log = get_logger(__name__)

#: What goes in ``ModelUsage.model`` for an answer no model produced.
#:
#: Not a model id, on purpose: the field is read back in logs and in the
#: ``model_calls`` table, and writing ``gemini-2.5-flash-lite`` there would
#: attribute template text to a model that was never reached.
FALLBACK_MODEL_ID: Final[str] = "deterministic-fallback"

#: Confidence floor used by ``assemble_profile`` (``min_confidence=0.5``).
#: Nothing below it is worth emitting, so every heuristic here scores at or
#: above it and anything weaker is left unset instead.
_MIN_USEFUL_CONFIDENCE: Final[float] = 0.6

_MAX_NEED_CATEGORIES: Final[int] = 6
_MAX_EXPLANATION_POINTS: Final[int] = 6
_MAX_EVIDENCE_DEPTH: Final[int] = 6
_MAX_CHAT_FACTS: Final[int] = 3
_MAX_NEXT_STEPS: Final[int] = 3
_MIN_KEYWORD_LEN: Final[int] = 4
_MIN_KEYWORD_HITS: Final[int] = 2
_MAX_HOUSEHOLD_COUNT: Final[int] = 20
_MAX_SPAN_CHARS: Final[int] = 240
_APPROXIMATE_BAND: Final[float] = 0.05


# ---------------------------------------------------------------------------
# reading the prompt back
# ---------------------------------------------------------------------------
_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    re.escape(BLOCK_OPEN) + r'\s+id="(?P<id>[^"]*)">>>\n(?P<body>.*?)\n' + re.escape(BLOCK_CLOSE),
    re.DOTALL,
)


def data_blocks(prompt_text: str) -> dict[str, str]:
    """The untrusted data blocks in a rendered prompt, keyed by their id.

    Public because the tests assert on it directly: if a prompt builder ever
    renames a block, the fallback must fail loudly in CI rather than quietly
    start reading an empty narrative.
    """
    found: dict[str, str] = {}
    for match in _BLOCK_RE.finditer(prompt_text):
        found.setdefault(match["id"], match["body"])
    return found


def _loads(raw: str | None) -> Any:
    """Parse a JSON data block, recovering the whole records of a truncated one.

    ``sanitise_untrusted`` caps every block at 8 KB, and a set of eighteen
    decided matches with their limitations is comfortably over that, so a
    truncated payload is the normal case rather than the edge case - the live
    model is handed the same truncated bytes. Refusing to parse it would mean
    an explanation with no reasons and an action plan with no steps, which
    reads like a broken product rather than a degraded one.

    So the complete records are recovered and the partial tail is discarded.
    Nothing is reconstructed: a record that did not arrive whole is dropped.
    """
    if raw is None or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    body = raw.split(_TRUNCATION_NOTE_PREFIX, 1)[0].lstrip()
    if body.startswith("["):
        return _recover_array(body)
    if body.startswith("{"):
        return _recover_object(body)
    return None


#: First words of ``app.ai.prompts.base._TRUNCATED_NOTE``. Matched as a prefix
#: so a reworded note still cuts in the right place.
_TRUNCATION_NOTE_PREFIX: Final[str] = "\n[content truncated"
_JSON_GAP: Final[str] = " \t\r\n,"


def _recover_array(body: str) -> list[Any]:
    """Every element of a truncated JSON array that can be recovered.

    An element that is itself cut off is recovered one level deeper rather than
    dropped. That matters for the evidence tree: an evaluation serialises as a
    single root condition whose ``children`` list is most of its size, so the
    root never arrives whole and dropping it would leave an explanation with no
    reasons at all. Its own ``description`` and ``outcome`` sit before
    ``children`` in the payload, so they are recoverable, and any key that did
    not arrive is simply absent - the callers treat absence as "not stated".
    """
    decoder = json.JSONDecoder()
    items: list[Any] = []
    position = 1
    while position < len(body):
        position = _skip_gap(body, position)
        if position >= len(body) or body[position] == "]":
            break
        try:
            value, position = decoder.raw_decode(body, position)
        except ValueError:
            partial = _recover_partial(body[position:])
            if partial is not None:
                items.append(partial)
            break
        items.append(value)
    return items


def _recover_partial(tail: str) -> Any:
    """Recover a cut-off JSON value, or ``None`` when it is not a container."""
    if tail.startswith("{"):
        recovered = _recover_object(tail)
        return recovered or None
    if tail.startswith("["):
        return _recover_array(tail) or None
    return None


def _recover_object(body: str) -> dict[str, Any]:
    """Every top-level key of a truncated JSON object that arrived complete."""
    decoder = json.JSONDecoder()
    out: dict[str, Any] = {}
    position = 1
    while position < len(body):
        position = _skip_gap(body, position)
        if position >= len(body) or body[position] == "}":
            break
        try:
            key, position = decoder.raw_decode(body, position)
        except ValueError:
            break
        position = _skip_gap(body, position)
        if position >= len(body) or body[position] != ":":
            break
        if not isinstance(key, str):
            break
        # ``raw_decode`` will not skip whitespace of its own, so the colon and
        # the indentation after it have to be stepped over explicitly.
        position = _skip_gap(body, position + 1)
        try:
            value, position = decoder.raw_decode(body, position)
        except ValueError:
            # The value is itself cut off. When it is a list - which is the
            # usual case, because the long keys here are ``evidence`` and
            # ``limitations`` - the elements that did arrive are still worth
            # keeping, so the same recovery runs one level down.
            partial = _recover_partial(body[position:])
            if partial is not None:
                out[key] = partial
            break
        out[key] = value
    return out


def _skip_gap(body: str, position: int) -> int:
    while position < len(body) and body[position] in _JSON_GAP:
        position += 1
    return position


def _block_mapping(blocks: Mapping[str, str], name: str) -> dict[str, Any]:
    loaded = _loads(blocks.get(name))
    return loaded if isinstance(loaded, dict) else {}


def _block_sequence(blocks: Mapping[str, str], name: str) -> list[Any]:
    loaded = _loads(blocks.get(name))
    return loaded if isinstance(loaded, list) else []


def _text_of(value: object, *, limit: int = 2_000) -> str | None:
    """A non-empty single-line string, or ``None``."""
    if not isinstance(value, str):
        return None
    collapsed = " ".join(value.split())
    if not collapsed:
        return None
    return collapsed[:limit]


def _clean_safe(values: Iterable[object], *, limit: int = 2_000) -> list[str]:
    """Strings that are non-empty, unique and free of the banned vocabulary.

    The filter is not cosmetic. ``BaseGeminiPort.explain_match`` and
    ``answer_question`` reject a whole response that contains one of those
    words, and curated program prose or a rule description may well contain
    "eligible". Dropping the sentence keeps the fallback usable; emitting it
    would turn an unreachable model into a failed request.
    """
    out: list[str] = []
    for value in values:
        text = _text_of(value, limit=limit)
        if text is None or text in out:
            continue
        if contains_banned_vocabulary(text) is not None:
            continue
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# extraction heuristics
# ---------------------------------------------------------------------------
_NUMBER_WORDS: Final[dict[str, int]] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_NUMBER: Final[str] = r"(?:\d{1,2}|" + "|".join(_NUMBER_WORDS) + r")"

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+|\n+")

#: An explicit "NJ" only. A city name is deliberately not enough - "I live in
#: New Brunswick" names no state, and guessing one would put a value the person
#: never gave in front of the rule engine. Every other state is left unset and
#: asked about, which is the honest reading of a narrative that does not say.
_STATE_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\bnew jersey\b", re.IGNORECASE), "NJ"),
    (re.compile(r"\bn\.?j\.?\b", re.IGNORECASE), "NJ"),
)

#: The 21 New Jersey counties. Geography, not program data, and matched only
#: when the narrative itself uses the word "county".
_NJ_COUNTIES: Final[tuple[str, ...]] = (
    "Atlantic",
    "Bergen",
    "Burlington",
    "Camden",
    "Cape May",
    "Cumberland",
    "Essex",
    "Gloucester",
    "Hudson",
    "Hunterdon",
    "Mercer",
    "Middlesex",
    "Monmouth",
    "Morris",
    "Ocean",
    "Passaic",
    "Salem",
    "Somerset",
    "Sussex",
    "Union",
    "Warren",
)
_COUNTY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:(?P<before>" + "|".join(_NJ_COUNTIES) + r")\s+county"
    r"|county\s+of\s+(?P<after>" + "|".join(_NJ_COUNTIES) + r"))\b",
    re.IGNORECASE,
)

_ALONE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:liv(?:e|es|ing)\s+(?:all\s+)?alone|by\s+myself|on\s+my\s+own|just\s+me)\b",
    re.IGNORECASE,
)
_HOUSEHOLD_TOTAL_RES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(rf"\b(?:household|family)\s+of\s+(?P<n>{_NUMBER})\b", re.IGNORECASE),
    re.compile(rf"\bthere\s+are\s+(?P<n>{_NUMBER})\s+of\s+us\b", re.IGNORECASE),
    re.compile(rf"\b(?P<n>{_NUMBER})\s+of\s+us\b", re.IGNORECASE),
    re.compile(
        rf"\b(?P<n>{_NUMBER})\s+(?:people|persons|adults)\s+(?:live|living|in)\b", re.IGNORECASE
    ),
)
#: "Me and my two kids" is a household of 3 - the speaker plus the children.
#: This is the one sum this module performs, and the extraction prompt states
#: the same reading, so it is the spec's arithmetic rather than an inference.
_WITH_CHILDREN_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:me\s+and\s+my|with\s+my|i\s+live\s+with\s+my|living\s+with\s+my)\s+"
    rf"(?P<n>{_NUMBER})\s+(?:kids?|children|child|sons?|daughters?)\b",
    re.IGNORECASE,
)
_CHILD_COUNT_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:my|have|raising|support)\s+(?P<n>{_NUMBER})\s+(?:kids?|children|child)\b",
    re.IGNORECASE,
)
_UNCOUNTED_CHILDREN_RE: Final[re.Pattern[str]] = re.compile(
    r"\bmy\s+(?:kids|children|son|daughter|baby|babies|toddler|grandkids)\b", re.IGNORECASE
)

_AGE_RES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b(?P<n>\d{1,3})[\s-]*(?:year|yr)s?[\s-]*old\b", re.IGNORECASE),
    re.compile(r"\bage[d]?\s*(?:of\s*)?(?P<n>\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+(?P<n>\d{1,3})(?=\s*(?:[,.;!?]|and\b|$))", re.IGNORECASE),
)

#: Ordered most specific first; the first pattern that matches wins.
_HOUSING_PATTERNS: Final[tuple[tuple[HousingSituation, re.Pattern[str], float], ...]] = (
    (
        HousingSituation.UNHOUSED,
        re.compile(r"\b(?:homeless|unhoused|sleeping in my car|on the street)\b", re.IGNORECASE),
        0.9,
    ),
    (
        HousingSituation.TEMPORARY_OR_SHELTER,
        re.compile(
            r"\b(?:a shelter|the shelter|in a motel|couch ?surf\w*|"
            r"staying with (?:a )?friends?|temporary housing|transitional housing)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        HousingSituation.UNIVERSITY_HOUSING,
        re.compile(
            r"\b(?:dorms?|dormitory|residence hall|on[- ]campus|university housing|"
            r"student housing)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        HousingSituation.LIVING_WITH_FAMILY,
        re.compile(
            r"\b(?:liv(?:e|ing)|stay(?:ing)?)\s+with\s+my\s+"
            r"(?:parents|mom|mother|dad|father|family|aunt|uncle|grandmother|"
            r"grandfather|grandparents|sister|brother)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        HousingSituation.OWN_HOME,
        re.compile(
            r"\b(?:my mortgage|i own my (?:home|house)|homeowner|own my own home)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        HousingSituation.RENTING,
        re.compile(
            r"\b(?:my rent\b|i rent\b|renting\b|my landlord|paying rent|rent is due|"
            r"rent money|behind on rent)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
)

_EMPLOYMENT_PATTERNS: Final[tuple[tuple[EmploymentStatus, re.Pattern[str], float], ...]] = (
    (
        EmploymentStatus.RETIRED,
        re.compile(r"\b(?:i am retired|i'm retired|retired\b)", re.IGNORECASE),
        0.9,
    ),
    (
        EmploymentStatus.UNABLE_TO_WORK,
        re.compile(
            r"\b(?:cannot work|can't work|unable to work|too sick to work|"
            r"on disability\b)",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        EmploymentStatus.UNEMPLOYED,
        re.compile(
            r"\b(?:unemployed|lost my job|laid off|out of work|no job right now|"
            r"looking for work|between jobs)\b",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        EmploymentStatus.SELF_EMPLOYED,
        re.compile(
            r"\b(?:self[- ]employed|my own business|freelance\w*|i run a business)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        EmploymentStatus.SEASONAL_OR_GIG,
        re.compile(
            r"\b(?:gig work|gig job|seasonal work|seasonal job|rideshare|"
            r"deliver(?:y|ing) app|odd jobs|temp work)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        EmploymentStatus.EMPLOYED_PART_TIME,
        re.compile(r"\bpart[- ]time\b", re.IGNORECASE),
        0.8,
    ),
    (
        EmploymentStatus.EMPLOYED_FULL_TIME,
        re.compile(r"\bfull[- ]time (?:job|work|position)\b|\bwork full[- ]time\b", re.IGNORECASE),
        0.8,
    ),
)

_STUDENT_PATTERNS: Final[tuple[tuple[StudentStatus, re.Pattern[str], float], ...]] = (
    (
        StudentStatus.NOT_A_STUDENT,
        re.compile(r"\b(?:not a student|i am not in school|not enrolled)\b", re.IGNORECASE),
        0.9,
    ),
    (
        StudentStatus.GRADUATE,
        re.compile(
            r"\b(?:grad(?:uate)? student|master'?s program|phd|doctoral|law school|"
            r"medical school)\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        StudentStatus.UNDERGRAD_PART_TIME,
        re.compile(r"\bpart[- ]time (?:student|undergrad\w*)\b", re.IGNORECASE),
        0.8,
    ),
    (
        StudentStatus.UNDERGRAD_FULL_TIME,
        re.compile(
            r"\bfull[- ]time (?:student|undergrad\w*)\b|\b(?:student )?taking \d+ credits\b",
            re.IGNORECASE,
        ),
        0.8,
    ),
)
#: "I am a student" with no course load. Not enough for a value, so it becomes
#: an ambiguity and a question instead.
_VAGUE_STUDENT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:i am a student|i'm a student|in college|in school|my classes|my semester|"
    r"my tuition)\b",
    re.IGNORECASE,
)

_NEED_PATTERNS: Final[tuple[tuple[NeedCategory, re.Pattern[str]], ...]] = (
    (
        NeedCategory.FOOD,
        re.compile(
            r"\b(?:food|groceries|grocery|hungry|meals?|food pantry|cannot eat|"
            r"keep food|formula)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.HEALTHCARE,
        re.compile(
            r"\b(?:doctor|medical bill\w*|health insurance|prescriptions?|medicine|"
            r"dental|dentist|hospital|health ?care)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.UTILITIES,
        re.compile(
            r"\b(?:utility|utilities|electric\w*|heating|heat bill|gas bill|water bill|"
            r"power bill|energy bill|shut[- ]?off|turned off my)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.HOUSING,
        re.compile(
            r"\b(?:rent\b|housing|evict\w*|mortgage|homeless|shelter|landlord|"
            r"security deposit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.CHILDCARE,
        re.compile(r"\b(?:child ?care|day ?care|babysit\w*|after[- ]school)\b", re.IGNORECASE),
    ),
    (
        NeedCategory.TRANSPORTATION,
        re.compile(
            r"\b(?:bus (?:fare|pass)|transit|transportation|car repair|"
            r"get(?:ting)? to work)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.EDUCATION_COSTS,
        re.compile(
            r"\b(?:tuition|textbooks?|school supplies|course fees|student loans?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.INCOME_SUPPORT,
        re.compile(
            r"\b(?:cash assistance|income support|help with money|lost my income)\b",
            re.IGNORECASE,
        ),
    ),
    (
        NeedCategory.INTERNET_PHONE,
        re.compile(r"\b(?:internet|wi-?fi|broadband|phone bill|cell ?phone)\b", re.IGNORECASE),
    ),
    (
        NeedCategory.LEGAL_HELP,
        re.compile(
            r"\b(?:lawyer|legal (?:help|aid|advice)|court (?:date|hearing))\b", re.IGNORECASE
        ),
    ),
    (
        NeedCategory.MENTAL_HEALTH,
        re.compile(r"\b(?:therapy|therapist|counsel(?:ing|or)|mental health)\b", re.IGNORECASE),
    ),
    (
        NeedCategory.EMERGENCY_EXPENSE,
        re.compile(
            r"\b(?:emergency|unexpected (?:bill|expense)|car broke down|funeral cost\w*)\b",
            re.IGNORECASE,
        ),
    ),
)

#: Flags, and the words that state them. ``value`` is only ever ``True``: the
#: schema has no way to say "not mentioned", and setting a flag ``False`` to
#: mean that would be a statement the person never made.
_FLAG_PATTERNS: Final[tuple[tuple[ProfileFlag, re.Pattern[str]], ...]] = (
    (
        ProfileFlag.RECEIVES_SNAP,
        re.compile(r"\b(?:food stamps|snap benefits|get snap|on snap)\b", re.IGNORECASE),
    ),
    (ProfileFlag.RECEIVES_WIC, re.compile(r"\bwic\b", re.IGNORECASE)),
    (
        ProfileFlag.RECEIVES_MEDICAID,
        re.compile(r"\b(?:medicaid|nj ?familycare)\b", re.IGNORECASE),
    ),
    (ProfileFlag.RECEIVES_TANF, re.compile(r"\b(?:tanf|work ?first nj)\b", re.IGNORECASE)),
    (
        ProfileFlag.RECEIVES_SSI,
        re.compile(r"\b(?:ssi\b|supplemental security income)\b", re.IGNORECASE),
    ),
    (ProfileFlag.RECEIVES_SSDI, re.compile(r"\bssdi\b", re.IGNORECASE)),
    (
        ProfileFlag.RECEIVES_UNEMPLOYMENT,
        re.compile(
            r"\b(?:unemployment (?:benefits|checks?)|collecting unemployment)\b", re.IGNORECASE
        ),
    ),
    (
        ProfileFlag.RECEIVES_FREE_SCHOOL_MEALS,
        re.compile(r"\b(?:free school (?:meals|lunch)|reduced[- ]price lunch)\b", re.IGNORECASE),
    ),
    (
        ProfileFlag.IS_PREGNANT,
        re.compile(r"\b(?:i am pregnant|i'm pregnant|expecting a baby)\b", re.IGNORECASE),
    ),
    (ProfileFlag.HAS_CHILDREN, _UNCOUNTED_CHILDREN_RE),
    (
        ProfileFlag.IS_VETERAN,
        re.compile(
            r"\b(?:i am a veteran|i'm a veteran|army veteran|navy veteran)\b", re.IGNORECASE
        ),
    ),
    (
        ProfileFlag.IS_DISABLED,
        re.compile(r"\b(?:i am disabled|i'm disabled|my disability)\b", re.IGNORECASE),
    ),
    (
        ProfileFlag.IS_SENIOR,
        re.compile(r"\b(?:senior citizen|i am elderly|i'm elderly)\b", re.IGNORECASE),
    ),
    (
        ProfileFlag.IS_CAREGIVER,
        re.compile(r"\b(?:i take care of my|caring for my|caregiver)\b", re.IGNORECASE),
    ),
    (ProfileFlag.IS_RUTGERS_AFFILIATED, re.compile(r"\brutgers\b", re.IGNORECASE)),
    (
        ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE,
        re.compile(
            r"\b(?:shut[- ]?off notice|disconnection notice|"
            r"turning off my (?:power|gas|electric))\b",
            re.IGNORECASE,
        ),
    ),
    (
        ProfileFlag.HAS_EVICTION_NOTICE,
        re.compile(r"\b(?:eviction notice|being evicted|court date for eviction)\b", re.IGNORECASE),
    ),
    (
        ProfileFlag.FILED_TAXES_LAST_YEAR,
        re.compile(r"\b(?:filed my taxes|filed a tax return)\b", re.IGNORECASE),
    ),
)

#: A sentence carrying one of these is not a statement that something is true
#: now. "I applied for food stamps" and "I do not get SNAP" must both leave
#: ``receives_snap`` unset.
_NOT_CURRENTLY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:appl(?:y|ied|ying)|applications?|waiting|pending|denied|rejected|"
    r"not|n't|never|no longer|used to|stopped|cancelled|canceled|lost my|"
    r"don't|dont|cut off|if i)\b",
    re.IGNORECASE,
)

#: Income is read only out of a sentence that says it is income.
_INCOME_CUE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:i make|we make|makes?\b|making|earn\w*|income|bring(?:ing)? (?:in|home)|"
    r"take home|salary|wages?|get paid|paid|pension|social security|"
    r"i get|we get|my check|our checks?)\b",
    re.IGNORECASE,
)
_MONEY_RE: Final[re.Pattern[str]] = re.compile(
    r"\$\s?(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
    r"\s*(?P<thousands>k\b)?"
    r"\s*(?:(?:an?|per|every|each)\s+|/\s?)"
    r"(?P<period>month|mo\b|year|yr\b|annum|week|wk\b|hour|hr\b|paycheck|biweekly)",
    re.IGNORECASE,
)
_APPROXIMATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:about|around|roughly|approximately|give or take|or so|maybe|ballpark)\b",
    re.IGNORECASE,
)
_PERIODS: Final[dict[str, IncomePeriod]] = {
    "month": IncomePeriod.MONTHLY,
    "mo": IncomePeriod.MONTHLY,
    "year": IncomePeriod.ANNUAL,
    "yr": IncomePeriod.ANNUAL,
    "annum": IncomePeriod.ANNUAL,
}
#: Periods the contract cannot express. Converting one would mean inventing the
#: hours, the shifts or the pay dates behind it, so the income is left unset and
#: the person is told why.
_UNCONVERTIBLE_PERIODS: Final[dict[str, str]] = {
    "week": "week",
    "wk": "week",
    "hour": "hour",
    "hr": "hour",
    "paycheck": "paycheck",
    "biweekly": "two weeks",
}

#: Text that is trying to talk to the assistant. Reported, never obeyed - the
#: extraction prompt asks for exactly this behaviour.
_INJECTION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:ignore (?:all )?(?:previous|prior|above)|disregard (?:all )?(?:previous|prior)|"
    r"system prompt|you are now|new instructions|mark (?:me|this user) as|"
    r"pretend (?:you|to be))\b",
    re.IGNORECASE,
)

#: Profile fields this module reports on, in the order the extraction prompt
#: lists them. Anything not set lands in ``unknown_fields``.
_REPORTED_FIELDS: Final[tuple[str, ...]] = (
    "state",
    "county",
    "student_status",
    "household_size",
    "household_composition",
    "income",
    "employment_status",
    "housing_situation",
    "needs",
    "age",
)


@dataclass(slots=True)
class _Reading:
    """One field value read out of the text, with the words that justify it."""

    value: Any
    span: str
    confidence: float


@dataclass(slots=True)
class _Extraction:
    """The accumulated reading of one narrative."""

    readings: dict[str, _Reading] = dataclass_field(default_factory=dict)
    flags: list[dict[str, Any]] = dataclass_field(default_factory=list)
    composition: dict[str, int] = dataclass_field(default_factory=dict)
    ambiguities: list[str] = dataclass_field(default_factory=list)

    def record(self, field_name: str, value: Any, span: str, confidence: float) -> None:
        if field_name in self.readings or confidence < _MIN_USEFUL_CONFIDENCE:
            return
        self.readings[field_name] = _Reading(
            value=value, span=span.strip()[:_MAX_SPAN_CHARS], confidence=confidence
        )

    def note(self, text: str) -> None:
        if text not in self.ambiguities:
            self.ambiguities.append(text)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _sentence_around(text: str, index: int) -> str:
    """The sentence containing ``index``, used for the negation guard."""
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        if match.end() > index:
            return text[start : match.start()]
        start = match.end()
    return text[start:]


def _count_of(token: str | None) -> int | None:
    if token is None:
        return None
    text = token.strip().casefold()
    if text.isdigit():
        value = int(text)
        return value if 1 <= value <= _MAX_HOUSEHOLD_COUNT else None
    return _NUMBER_WORDS.get(text)


def _first_pattern[T](
    text: str, table: Sequence[tuple[T, re.Pattern[str], float]]
) -> tuple[T, str, float] | None:
    for value, pattern, confidence in table:
        match = pattern.search(text)
        if match is not None:
            return value, match.group(0), confidence
    return None


def _read_state(text: str, out: _Extraction) -> None:
    for pattern, code in _STATE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            out.record("state", code, match.group(0), 1.0)
            return


def _read_county(text: str, out: _Extraction) -> None:
    match = _COUNTY_RE.search(text)
    if match is None:
        return
    name = match["before"] or match["after"]
    if name is None:  # pragma: no cover - one group always matches
        return
    out.record("county", name.title(), match.group(0), 1.0)


def _read_household(text: str, out: _Extraction) -> None:
    alone = _ALONE_RE.search(text)
    if alone is not None:
        out.record("household_size", 1, alone.group(0), 0.9)
        out.composition["adults"] = 1
        return
    for pattern in _HOUSEHOLD_TOTAL_RES:
        match = pattern.search(text)
        size = _count_of(match["n"]) if match is not None else None
        if match is not None and size is not None:
            out.record("household_size", size, match.group(0), 0.9)
            return
    with_children = _WITH_CHILDREN_RE.search(text)
    children = _count_of(with_children["n"]) if with_children is not None else None
    if with_children is not None and children is not None:
        out.record("household_size", children + 1, with_children.group(0), 0.8)
        out.composition["adults"] = 1
        out.composition["children_under_18"] = children
        return
    counted = _CHILD_COUNT_RE.search(text)
    only_children = _count_of(counted["n"]) if counted is not None else None
    if counted is not None and only_children is not None:
        out.composition["children_under_18"] = only_children
        out.note(
            "You told us how many children you have, but not how many people are in your "
            "household in total, so we left the household size blank."
        )
        return
    if _UNCOUNTED_CHILDREN_RE.search(text) is not None:
        out.note(
            "You mentioned children but not how many, so we left your household size blank "
            "rather than guess it."
        )


def _read_age(text: str, out: _Extraction) -> None:
    for pattern in _AGE_RES:
        match = pattern.search(text)
        if match is None:
            continue
        try:
            age = int(match["n"])
        except (TypeError, ValueError):  # pragma: no cover - the group is \d+
            continue
        if 0 <= age <= 120:
            out.record("age", age, match.group(0), 1.0)
            return


def _read_needs(text: str, out: _Extraction) -> None:
    found: list[str] = []
    span: str | None = None
    for category, pattern in _NEED_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        found.append(category.value)
        span = span or match.group(0)
        if len(found) >= _MAX_NEED_CATEGORIES:
            break
    if found:
        out.record("needs", found, span or "", 0.8)


def _read_flags(text: str, out: _Extraction) -> None:
    for flag, pattern in _FLAG_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        sentence = _sentence_around(text, match.start())
        if _NOT_CURRENTLY_RE.search(sentence) is not None:
            # "I applied for food stamps", "I do not get SNAP". Stated as a
            # plan, a refusal or a past fact - not as a current one.
            continue
        out.flags.append(
            {
                "flag": flag.value,
                "value": True,
                "confidence": 0.9,
                "source_span": sentence.strip()[:_MAX_SPAN_CHARS] or match.group(0),
            }
        )


def _read_income(text: str, out: _Extraction) -> None:
    """Read an income range, or record why none was read.

    Never converts. An hourly or weekly figure without the hours behind it
    cannot become a monthly one without inventing the hours, and an invented
    income is the single most harmful thing this module could emit: it flows
    straight into an income-limit rule and comes back out as a result the
    person never stated.
    """
    for match in _MONEY_RE.finditer(text):
        sentence = _sentence_around(text, match.start())
        if _INCOME_CUE_RE.search(sentence) is None:
            continue  # "my heating bill is $300 a month" is not an income.
        period_word = (match["period"] or "").casefold()
        period = _PERIODS.get(period_word)
        if period is None:
            label = _UNCONVERTIBLE_PERIODS.get(period_word, "pay period")
            out.note(
                f"You told us an amount for each {label}. Turning that into a monthly or "
                "yearly figure would mean guessing at the rest, so we left your income blank "
                "and asked about it instead."
            )
            continue
        amount = _cents(match["amount"], thousands=bool(match["thousands"]))
        if amount is None:
            continue
        approximate = _APPROXIMATE_RE.search(sentence) is not None
        if approximate:
            band = round(amount * _APPROXIMATE_BAND)
            low, high = max(amount - band, 0), amount + band
            out.note(
                f"You said {match.group(0).strip()}, so we read your income as a range around "
                "that figure rather than an exact amount."
            )
        else:
            low, high = amount, amount
        out.record("income_min_cents", low, match.group(0), 0.8)
        out.record("income_max_cents", high, match.group(0), 0.8)
        out.record("income_period", period.value, match.group(0), 0.8)
        return


def _cents(amount: str | None, *, thousands: bool) -> int | None:
    if amount is None:
        return None
    try:
        dollars = float(amount.replace(",", ""))
    except ValueError:  # pragma: no cover - the group is numeric
        return None
    if thousands:
        dollars *= 1_000
    if dollars <= 0 or dollars > 10_000_000:
        return None
    return round(dollars * 100)


def _read_narrative(narrative: str) -> _Extraction:
    """Everything the submitted text states, and nothing else."""
    out = _Extraction()
    _read_state(narrative, out)
    _read_county(narrative, out)
    _read_household(narrative, out)
    _read_age(narrative, out)
    _read_needs(narrative, out)
    _read_income(narrative, out)
    _read_flags(narrative, out)

    housing = _first_pattern(narrative, _HOUSING_PATTERNS)
    if housing is not None:
        out.record("housing_situation", housing[0].value, housing[1], housing[2])
    employment = _first_pattern(narrative, _EMPLOYMENT_PATTERNS)
    if employment is not None:
        out.record("employment_status", employment[0].value, employment[1], employment[2])
    student = _first_pattern(narrative, _STUDENT_PATTERNS)
    if student is not None:
        out.record("student_status", student[0].value, student[1], student[2])
    elif _VAGUE_STUDENT_RE.search(narrative) is not None:
        out.note(
            "You mentioned school, but not whether you are enrolled full time or part time, "
            "so we left your student status blank."
        )
    if _INJECTION_RE.search(narrative) is not None:
        out.note(
            "Part of what you wrote reads like an instruction to this assistant. We treated it "
            "as information about your situation and nothing more."
        )
    return out


def _extraction_payload(narrative: str, prefilled: Mapping[str, Any]) -> dict[str, Any]:
    """Build the :class:`~app.ai.schemas.ExtractionResponse` body."""
    out = _read_narrative(narrative)
    readings = out.readings
    payload: dict[str, Any] = {
        name: reading.value
        for name, reading in readings.items()
        if name not in {"needs", "household_composition"}
    }
    payload["needs"] = readings["needs"].value if "needs" in readings else []
    if out.composition:
        payload["household_composition"] = dict(out.composition)
    payload["flags"] = out.flags

    evidence: list[dict[str, Any]] = []
    for name, reading in readings.items():
        field_name = "income" if name.startswith("income_") else name
        if any(item["field"] == field_name for item in evidence):
            continue
        evidence.append(
            {
                "field": field_name,
                "confidence": reading.confidence,
                "source_span": reading.span,
                "reasoning": "read from the words you wrote",
            }
        )
    if "household_composition" in payload:
        evidence.append(
            {
                "field": "household_composition",
                "confidence": 0.8,
                "source_span": readings["household_size"].span
                if "household_size" in readings
                else None,
                "reasoning": "counted from the people you described",
            }
        )
    payload["field_evidence"] = evidence

    determined = {item["field"] for item in evidence}
    # A value the person typed into the form is not something we failed to
    # understand, so it is not reported back as unknown.
    determined |= {name for name, value in prefilled.items() if value not in (None, [], {})}
    if prefilled.get("income_period") is not None:
        determined.add("income")
    payload["unknown_fields"] = [name for name in _REPORTED_FIELDS if name not in determined]
    payload["ambiguities"] = out.ambiguities
    return payload


# ---------------------------------------------------------------------------
# clarification templates
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _QuestionTemplate:
    """One question this module can ask, in fixed, reviewed wording.

    Deliberately a closed table. There is no path here that turns program text
    or a narrative into a question, so the fallback cannot be talked into
    asking for a Social Security number, a bank account or an immigration
    document - none of them is in the table, and ``field`` is validated against
    ``PROFILE_FIELD_NAMES`` by the projection.
    """

    field: str
    question: str
    why_it_matters: str
    input_kind: ClarificationInputKind
    options: tuple[tuple[str, str], ...] = ()
    unit: str | None = None
    min_value: float | None = None
    max_value: float | None = None


#: Ordered by how much an answer tends to unblock, highest first: the
#: projection keeps the first questions that survive and caps the rest.
_QUESTION_TEMPLATES: Final[tuple[_QuestionTemplate, ...]] = (
    _QuestionTemplate(
        field="state",
        question="Which state do you live in?",
        why_it_matters="Nearly all of this help is only open to people living in New Jersey.",
        input_kind=ClarificationInputKind.TEXT,
    ),
    _QuestionTemplate(
        field="household_size",
        question="How many people live in your household, counting yourself?",
        why_it_matters="Income limits rise with household size, so this can change the answer.",
        input_kind=ClarificationInputKind.NUMBER,
        unit="people",
        min_value=1,
        max_value=20,
    ),
    _QuestionTemplate(
        field="income",
        question="Roughly how much money does your household bring in each month?",
        why_it_matters=(
            "A rough range is enough. Several programs set a monthly income limit and we "
            "cannot tell yet which side of it you are on."
        ),
        input_kind=ClarificationInputKind.RANGE,
        unit="dollars per month",
        min_value=0,
    ),
    _QuestionTemplate(
        field="county",
        question="Which New Jersey county do you live in?",
        why_it_matters="Some help is run county by county, so this changes what is available.",
        input_kind=ClarificationInputKind.TEXT,
    ),
    _QuestionTemplate(
        field="student_status",
        question="Are you enrolled as a student right now?",
        why_it_matters="Student rules change which food and emergency help you can get.",
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=(
            (StudentStatus.NOT_A_STUDENT.value, "I am not a student"),
            (StudentStatus.UNDERGRAD_FULL_TIME.value, "Undergraduate, full time"),
            (StudentStatus.UNDERGRAD_PART_TIME.value, "Undergraduate, part time"),
            (StudentStatus.GRADUATE.value, "Graduate or professional student"),
            (StudentStatus.OTHER.value, "Something else"),
        ),
    ),
    _QuestionTemplate(
        field="housing_situation",
        question="Do you rent, own your home, or stay with someone else?",
        why_it_matters="Housing help is split by this, so it changes which programs can help.",
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=(
            (HousingSituation.RENTING.value, "I rent"),
            (HousingSituation.OWN_HOME.value, "I own my home"),
            (HousingSituation.LIVING_WITH_FAMILY.value, "I stay with family or friends"),
            (HousingSituation.UNIVERSITY_HOUSING.value, "I live in university housing"),
            (HousingSituation.TEMPORARY_OR_SHELTER.value, "I am somewhere temporary"),
            (HousingSituation.UNHOUSED.value, "I do not have housing right now"),
        ),
    ),
    _QuestionTemplate(
        field="employment_status",
        question="Are you working right now?",
        why_it_matters=(
            "Some help is only for people in work, and some only for people who are not."
        ),
        input_kind=ClarificationInputKind.SINGLE_SELECT,
        options=(
            (EmploymentStatus.EMPLOYED_FULL_TIME.value, "Working full time"),
            (EmploymentStatus.EMPLOYED_PART_TIME.value, "Working part time"),
            (EmploymentStatus.SELF_EMPLOYED.value, "Self-employed"),
            (EmploymentStatus.SEASONAL_OR_GIG.value, "Seasonal or gig work"),
            (EmploymentStatus.UNEMPLOYED.value, "Not working right now"),
            (EmploymentStatus.RETIRED.value, "Retired"),
            (EmploymentStatus.UNABLE_TO_WORK.value, "Unable to work"),
        ),
    ),
    _QuestionTemplate(
        field="age",
        question="How old are you?",
        why_it_matters="A few programs are only for people over or under a certain age.",
        input_kind=ClarificationInputKind.NUMBER,
        unit="years",
        min_value=0,
        max_value=120,
    ),
)

#: What each missing field is asked as, for ``what_we_still_need``.
_FIELD_QUESTIONS: Final[dict[str, str]] = {
    template.field: template.question for template in _QUESTION_TEMPLATES
} | {
    "income_min_cents": "Roughly how much money does your household bring in each month?",
    "income_max_cents": "Roughly how much money does your household bring in each month?",
    "income_period": "Is the amount you gave us for a month or for a year?",
    "household_composition": "Who lives with you - how many adults, children and seniors?",
    "needs": "What kind of help do you need most right now?",
}


#: One plain question per flag, so a missing flag reads like something a person
#: can answer rather than like a field name. Reviewed copy: none of these asks
#: for an identifier, a document number or an immigration status.
_FLAG_QUESTIONS: Final[dict[ProfileFlag, str]] = {
    ProfileFlag.RECEIVES_SNAP: "Do you already get SNAP, sometimes called food stamps?",
    ProfileFlag.RECEIVES_WIC: "Do you already get WIC?",
    ProfileFlag.RECEIVES_MEDICAID: "Do you already have Medicaid or NJ FamilyCare?",
    ProfileFlag.RECEIVES_TANF: "Do you already get TANF or WorkFirst NJ cash assistance?",
    ProfileFlag.RECEIVES_SSI: "Do you already get SSI payments?",
    ProfileFlag.RECEIVES_SSDI: "Do you already get Social Security disability payments?",
    ProfileFlag.RECEIVES_UNEMPLOYMENT: "Are you getting unemployment payments right now?",
    ProfileFlag.RECEIVES_FREE_SCHOOL_MEALS: "Does a child in your home get free school meals?",
    ProfileFlag.IS_PREGNANT: "Is anyone in your household pregnant right now?",
    ProfileFlag.HAS_CHILDREN: "Are there children under 18 living with you?",
    ProfileFlag.IS_VETERAN: "Have you served in the military?",
    ProfileFlag.IS_DISABLED: "Do you have a disability?",
    ProfileFlag.IS_SENIOR: "Are you 60 or older?",
    ProfileFlag.IS_CAREGIVER: "Do you look after someone who needs daily help?",
    ProfileFlag.IS_RUTGERS_AFFILIATED: "Do you study or work at Rutgers?",
    ProfileFlag.HAS_UTILITY_SHUTOFF_NOTICE: (
        "Have you had a notice saying your gas or electric will be shut off?"
    ),
    ProfileFlag.HAS_EVICTION_NOTICE: "Have you been given an eviction notice?",
    ProfileFlag.FILED_TAXES_LAST_YEAR: "Did you file a tax return last year?",
}


def _question_for_field(name: str) -> str | None:
    if name in _FIELD_QUESTIONS:
        return _FIELD_QUESTIONS[name]
    if name.startswith("flags."):
        try:
            flag = ProfileFlag(name.removeprefix("flags."))
        except ValueError:
            return None
        return _FLAG_QUESTIONS.get(flag)
    return None


def _profile_determines(profile: Mapping[str, Any], name: str) -> bool:
    """Whether a profile payload already settles ``name``.

    ``profile_payload`` drops unset fields, so presence is the test. A field the
    person was already asked and skipped sits in ``unknown_fields`` and counts
    as settled: asking twice is how this starts to feel like the government form
    it exists to replace.
    """
    unknown = profile.get("unknown_fields")
    if isinstance(unknown, list) and name in unknown:
        return True
    if name == "income":
        return profile.get("income_period") is not None and (
            profile.get("income_min_cents") is not None
            or profile.get("income_max_cents") is not None
        )
    value = profile.get(name)
    return value not in (None, [], {})


def _clarification_payload(blocks: Mapping[str, str]) -> dict[str, Any]:
    profile = _block_mapping(blocks, "user_profile")
    questions: list[dict[str, Any]] = []
    for template in _QUESTION_TEMPLATES:
        if _profile_determines(profile, template.field):
            continue
        questions.append(
            {
                "field": template.field,
                "question": template.question,
                "input_kind": template.input_kind.value,
                "options": [{"value": value, "label": label} for value, label in template.options],
                "why_it_matters": template.why_it_matters,
                "is_required": False,
                "unit": template.unit,
                "min_value": template.min_value,
                "max_value": template.max_value,
                # Left empty on purpose: which program an answer would unblock
                # is a rule-engine fact, and this module has no evidence for it.
                "affected_program_slugs": [],
            }
        )
    rationale = (
        "Written from the deterministic question bank because the model was unreachable; "
        "ordered by how much an answer usually unblocks."
        if questions
        else "Nothing in the deterministic question bank is still missing from this profile."
    )
    return {"questions": questions, "rationale": rationale}


# ---------------------------------------------------------------------------
# explanation templates
# ---------------------------------------------------------------------------
#: One headline per decided status. Describes the decision, never makes one,
#: and avoids the banned vocabulary by construction - ``tests/unit`` asserts
#: that for every string in this module.
_HEADLINES: Final[dict[MatchStatus, str]] = {
    MatchStatus.POTENTIAL_MATCH: ("From what you told us, this one looks worth applying for."),
    MatchStatus.MORE_INFO_NEEDED: (
        "We need a little more information before we can check this one."
    ),
    MatchStatus.UNLIKELY_MATCH: (
        "What you told us does not line up with this program's rules, but it decides for itself."
    ),
}
_GENERIC_HEADLINE: Final[str] = "Here is what we can tell you about this program."

#: The decided status as ``app.ai.prompts.explanation.build`` states it in the
#: trusted instruction text: "...this program's decided result: MORE_INFO_NEEDED."
_STATUS_HEADER_RE: Final[re.Pattern[str]] = re.compile(r"decided result:\s*(?P<status>[A-Z_]+)")

#: Attached to every explanation this module writes. The interface also renders
#: a run-level limitation from ``usage.source``; this one travels with the
#: individual explanation so it cannot be separated from the words it is about.
_TEMPLATE_CAVEAT: Final[str] = (
    "This wording was put together from a template, not written by the AI model, because the "
    "model could not be reached. The result itself comes from the same rules either way."
)
_PROGRAM_DECIDES_CAVEAT: Final[str] = (
    "The program itself makes the final decision, and its rules can change."
)


def _decided_status(evaluation: Mapping[str, Any], prompt_text: str) -> MatchStatus | None:
    """The status the engine decided, from the payload or the prompt's header.

    Two readings of the same fact, because wording the wrong status is the
    worst thing this module could do. The payload is preferred; the header line
    ``app.ai.prompts.explanation.build`` writes is the backstop, and it is
    trusted text from this repository rather than anything a caller supplied.
    ``tests/unit/test_ai_fallback.py`` asserts the header still matches, so
    rewording the prompt cannot silently disable this.
    """
    candidates = [_text_of(evaluation.get("decided_status"))]
    header = _STATUS_HEADER_RE.search(prompt_text)
    candidates.append(header["status"] if header is not None else None)
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return MatchStatus(candidate)
        except ValueError:
            continue
    return None


def _satisfied_reasons(evidence: Sequence[Any], *, depth: int = 0) -> list[str]:
    """Descriptions of the requirements the engine resolved ``TRUE``.

    Walks to the leaves, because the engine's own leaf wording is the specific,
    plain-language sentence a person can act on - "You live in New Jersey - the
    statewide program covers every county" - while the node above it is an
    internal label like "SNAP preliminary eligibility screen". A branch whose
    leaves did not survive the prompt's truncation falls back to its own
    description, so a reason is never silently lost.

    ``TRUE`` only. A node the rules could not resolve is not a reason the
    program may fit, and this module never reads an outcome it was not given.
    """
    reasons: list[str] = []
    if depth > _MAX_EVIDENCE_DEPTH:  # pragma: no cover - the trees are shallow
        return reasons
    for item in evidence:
        if not isinstance(item, dict) or item.get("outcome") != "TRUE":
            continue
        children = item.get("children")
        nested = _satisfied_reasons(children, depth=depth + 1) if isinstance(children, list) else []
        if nested:
            reasons.extend(nested)
            continue
        description = _text_of(item.get("description"))
        if description is not None:
            reasons.append(description)
    return reasons


def _explanation_payload(blocks: Mapping[str, str], prompt_text: str) -> dict[str, Any]:
    evaluation = _block_mapping(blocks, "evaluation")
    status = _decided_status(evaluation, prompt_text)
    headline = _HEADLINES[status] if status is not None else _GENERIC_HEADLINE

    evidence = evaluation.get("evidence")
    why: list[str] = []
    if isinstance(evidence, list):
        why = _clean_safe(_satisfied_reasons(evidence))[:_MAX_EXPLANATION_POINTS]

    missing = evaluation.get("missing_fields")
    still_need: list[str] = []
    if isinstance(missing, list):
        still_need = _clean_safe(
            _question_for_field(name) for name in missing if isinstance(name, str)
        )[:_MAX_EXPLANATION_POINTS]

    # The evaluation's own limitations are merged in unconditionally by
    # ``ExplanationResponse.to_match_explanation``, so they are deliberately
    # not copied here: a limitation containing the word "eligible" would make
    # ``BaseGeminiPort.explain_match`` reject this whole response.
    return {
        "headline": headline,
        "why_it_may_fit": why,
        "what_we_still_need": still_need,
        "caveats": [_TEMPLATE_CAVEAT, _PROGRAM_DECIDES_CAVEAT],
    }


# ---------------------------------------------------------------------------
# action plan templates
# ---------------------------------------------------------------------------
_PLAN_TITLE: Final[str] = "Your next steps"
_PLAN_SUMMARY: Final[str] = (
    "Start at the top of this list. Each step says what to gather or who to contact, and the "
    "official page is linked where we have one. This list was put together from a template "
    "because the AI model could not be reached, so it is plainer than usual."
)
_EMPTY_PLAN_SUMMARY: Final[str] = (
    "We do not have a next step to give you yet. Once we know a little more about your "
    "situation, this list will fill in."
)


def _plan_items_for(match: Mapping[str, Any]) -> list[dict[str, Any]]:
    slug = _text_of(match.get("program_slug"), limit=200)
    if slug is None:
        return []
    name = _text_of(match.get("official_name")) or "this program"
    application_url = _text_of(match.get("application_url"), limit=2_000)
    official_url = _text_of(match.get("official_url"), limit=2_000)
    status_value = _text_of(match.get("decided_status"))
    raw_missing = match.get("missing_fields")
    missing = (
        [name for name in raw_missing if isinstance(name, str)]
        if isinstance(raw_missing, list)
        else []
    )

    if status_value == MatchStatus.POTENTIAL_MATCH.value:
        if application_url is not None:
            return [
                {
                    "program_slug": slug,
                    "kind": ActionItemKind.APPLY.value,
                    "subject": "application",
                    "title": f"Start the application for {name}",
                    "detail": (
                        "Open the application page and work through it. Have your address and "
                        "proof of who lives with you nearby, since most applications ask for "
                        "them at some point."
                    ),
                    "url": application_url,
                }
            ]
        return [
            {
                "program_slug": slug,
                "kind": ActionItemKind.CONTACT.value,
                "subject": "contact the program",
                "title": f"Contact {name} to ask how to apply",
                "detail": (
                    "We do not have an application link for this one, so the official page or "
                    "phone number is the reliable way in. Ask what they need from you before "
                    "you gather anything."
                ),
                "url": official_url,
            }
        ]
    if status_value == MatchStatus.MORE_INFO_NEEDED.value:
        wanted = _clean_safe(_question_for_field(field) for field in missing)[:3]
        detail = (
            "Answering this lets the rules finish for this program: " + " ".join(wanted)
            if wanted
            else (
                "Some details are still missing, so the rules could not finish for this "
                "program. The official page explains what it asks for."
            )
        )
        return [
            {
                "program_slug": slug,
                "kind": ActionItemKind.CONFIRM_INFO.value,
                "subject": "missing information",
                "title": f"Fill in what is still missing for {name}",
                "detail": detail,
                "url": official_url,
            }
        ]
    return []


def _plan_payload(blocks: Mapping[str, str]) -> dict[str, Any]:
    matches = [item for item in _block_sequence(blocks, "matches") if isinstance(item, dict)]
    ranked = sorted(
        matches,
        key=lambda match: (
            0 if match.get("decided_status") == MatchStatus.POTENTIAL_MATCH.value else 1
        ),
    )
    items: list[dict[str, Any]] = []
    for match in ranked:
        items.extend(_plan_items_for(match))
    return {
        "title": _PLAN_TITLE,
        "summary": _PLAN_SUMMARY if items else _EMPTY_PLAN_SUMMARY,
        "items": items,
    }


# ---------------------------------------------------------------------------
# chat templates
# ---------------------------------------------------------------------------
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "about",
        "after",
        "another",
        "because",
        "before",
        "could",
        "does",
        "doing",
        "else",
        "from",
        "have",
        "help",
        "here",
        "into",
        "just",
        "know",
        "like",
        "make",
        "many",
        "more",
        "much",
        "need",
        "only",
        "other",
        "over",
        "same",
        "should",
        "some",
        "tell",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "told",
        "very",
        "want",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "would",
        "your",
    }
)
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z'-]+")
_CHAT_FOUND: Final[str] = (
    "The AI assistant could not be reached, so instead of writing an answer we searched the "
    "official sources we hold and copied out the lines that look closest to your question. "
    "They are quoted below exactly as the source words them."
)
_CHAT_CAVEATS: Final[tuple[str, ...]] = (
    "This reply was assembled from a template, not written by the AI assistant, because the "
    "model could not be reached.",
    "Only the official sources we already hold were searched, so something may be missing.",
)


def _keywords(question: str) -> set[str]:
    return {
        word
        for word in _WORD_RE.findall(question.casefold())
        if len(word) >= _MIN_KEYWORD_LEN and word not in _STOPWORDS
    }


def _matching_sentences(
    sources: Sequence[Mapping[str, Any]], keywords: set[str]
) -> list[tuple[str, str, int]]:
    """Source sentences that share keywords with the question.

    Returns ``(source_id, sentence, hits)``. The sentence is a verbatim span of
    the stored excerpt, which is what lets it survive the citation gate in
    ``ChatAnswerResponse.to_chat_response`` - that gate re-checks every excerpt
    against the stored source text, so nothing invented could get through.
    """
    scored: list[tuple[str, str, int]] = []
    for source in sources:
        source_id = _text_of(source.get("source_id"), limit=200)
        excerpt = source.get("excerpt")
        if source_id is None or not isinstance(excerpt, str):
            continue
        for sentence in _sentences(excerpt):
            if contains_banned_vocabulary(sentence) is not None:
                continue
            hits = sum(1 for word in keywords if word in sentence.casefold())
            if hits >= _MIN_KEYWORD_HITS:
                scored.append((source_id, sentence, hits))
    scored.sort(key=lambda row: (-row[2], row[0], row[1]))
    return scored[:_MAX_CHAT_FACTS]


def _chat_payload(blocks: Mapping[str, str]) -> dict[str, Any]:
    sources = [item for item in _block_sequence(blocks, "sources") if isinstance(item, dict)]
    question = blocks.get("user_question", "")
    matches = _matching_sentences(sources, _keywords(question))

    next_steps = _clean_safe(
        f'Read "{_text_of(source.get("title"))}" on the official page.'
        for source in sources
        if _text_of(source.get("title")) is not None
    )[:_MAX_NEXT_STEPS]

    if not matches:
        return {
            "can_answer": False,
            "answer": CANNOT_ANSWER_TEXT,
            "facts": [],
            "interpretation": [],
            "citations": [],
            "confidence": ConfidenceLevel.UNKNOWN.value,
            "caveats": list(_CHAT_CAVEATS),
            "suggested_next_steps": next_steps,
        }
    return {
        "can_answer": True,
        "answer": _CHAT_FOUND,
        "facts": [
            {"text": sentence, "source_ids": [source_id]} for source_id, sentence, _ in matches
        ],
        "interpretation": [],
        "citations": [
            {"source_id": source_id, "excerpt": sentence} for source_id, sentence, _ in matches
        ],
        # LOW, never higher: a keyword match is not a reading of the question.
        "confidence": ConfidenceLevel.LOW.value,
        "caveats": list(_CHAT_CAVEATS),
        "suggested_next_steps": next_steps,
    }


# ---------------------------------------------------------------------------
# free-form text
# ---------------------------------------------------------------------------
_TEXT_ANSWERS: Final[dict[GeminiTask, str]] = {
    GeminiTask.EXTRACT_PROFILE: (
        "We read your description with a set of fixed rules rather than the AI model, which "
        "could not be reached, so we kept only what you said in your own words."
    ),
    GeminiTask.PROPOSE_CLARIFICATIONS: (
        "The questions below come from a fixed list, because the AI model could not be reached."
    ),
    GeminiTask.EXPLAIN_MATCH: (
        "This wording comes from a template because the AI model could not be reached. The "
        "result itself comes from the same rules either way."
    ),
    GeminiTask.DRAFT_ACTION_PLAN: (
        "These steps come from a template because the AI model could not be reached."
    ),
    GeminiTask.ANSWER_QUESTION: CANNOT_ANSWER_TEXT,
}


# ---------------------------------------------------------------------------
# the port
# ---------------------------------------------------------------------------
class DeterministicAi(BaseGeminiPort):
    """The product's non-model language path: heuristics in, templates out.

    Stateless and side-effect free, so one instance is shared process-wide and
    concurrent calls cannot interfere. Everything it returns is a function of
    the prompt it was handed.
    """

    __slots__ = ("_settings",)

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings

    # ------------------------------------------------------------- primitives
    async def structured[M: BaseModel](
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        schema: type[M],
        context: CallContext | None = None,
    ) -> StructuredResult[M]:
        """Answer ``task`` from the prompt's own data, with no model call."""
        blocks = data_blocks(prompt.prompt)
        payload = self._payload(task, prompt.prompt, blocks)
        try:
            value = schema.model_validate(payload)
        except ValidationError as exc:
            # A bug in this module, not a provider failure - but the caller
            # only has the model-error taxonomy to react to, and "we could not
            # produce an answer" is the honest thing to tell it.
            _log.error(
                "ai.fallback.invalid_payload",
                task=task.value,
                schema=schema.__name__,
                errors=exc.error_count(),
            )
            raise ModelInvalidOutput(
                task=task.value,
                model=FALLBACK_MODEL_ID,
                reason="fallback_payload_invalid",
                cause=exc,
            ) from exc
        _log.info(
            "ai.fallback.answered",
            task=task.value,
            blocks=sorted(blocks),
            fields=len(payload),
        )
        return StructuredResult(value=value, usage=self._usage(prompt), notes=[])

    async def text(
        self,
        task: GeminiTask,
        prompt: PromptSpec,
        context: CallContext | None = None,
    ) -> TextResult:
        """The fixed, honest sentence for ``task``."""
        return TextResult(text=_TEXT_ANSWERS[task], usage=self._usage(prompt))

    # ----------------------------------------------------------------- details
    def _payload(
        self, task: GeminiTask, prompt_text: str, blocks: Mapping[str, str]
    ) -> dict[str, Any]:
        """The response body for ``task``, derived from its own prompt."""
        if task is GeminiTask.EXTRACT_PROFILE:
            return _extraction_payload(
                blocks.get("user_narrative", ""),
                _block_mapping(blocks, "prefilled_profile"),
            )
        if task is GeminiTask.PROPOSE_CLARIFICATIONS:
            return _clarification_payload(blocks)
        if task is GeminiTask.EXPLAIN_MATCH:
            return _explanation_payload(blocks, prompt_text)
        if task is GeminiTask.DRAFT_ACTION_PLAN:
            return _plan_payload(blocks)
        if task is GeminiTask.ANSWER_QUESTION:
            return _chat_payload(blocks)
        raise AssertionError(f"unhandled task {task!r}")  # pragma: no cover - enum is exhaustive

    def _usage(self, prompt: PromptSpec) -> ModelUsage:
        """No tokens, no cost, and ``source="fallback"`` by construction."""
        return ModelUsage.for_fallback(FALLBACK_MODEL_ID, prompt.ref)


assert set(_TEXT_ANSWERS) == set(MODEL_RESPONSE_SCHEMAS), "every task needs a text answer"
