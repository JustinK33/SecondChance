"""Tests for the model-facing schemas and their projections.

Two guarantees are checked structurally rather than by inspection:

1. no schema the model answers in can express a
   :class:`~app.schemas.enums.MatchStatus`;
2. every schema survives the real google-genai conversion to a
   ``types.Schema``, which is what ``response_schema`` actually sends. That
   conversion rejects constructs pydantic is perfectly happy with, so a schema
   that is only checked by pydantic is a schema that fails on the first live
   call.

The rest covers the projections, where all the defensive dropping happens.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from google import genai
from google.genai import _transformers

from app.ai.schemas import (
    MODEL_RESPONSE_SCHEMAS,
    ActionItemDraftOut,
    ActionPlanResponse,
    ChatAnswerResponse,
    CitationOut,
    ClarificationOptionOut,
    ClarificationQuestionOut,
    ClarificationResponse,
    ExplanationResponse,
    ExtractedFlagOut,
    ExtractionResponse,
    FactStatementOut,
    FieldEvidenceOut,
    GeminiTask,
    HouseholdCompositionOut,
)
from app.schemas import (
    ActionItemKind,
    ClarificationInputKind,
    ClarificationQuestionSet,
    ConfidenceLevel,
    IncomePeriod,
    MatchStatus,
    NeedCategory,
    ProfileFlag,
    ProgramSource,
    SourceFreshness,
)
from app.schemas.profile import PROFILE_FIELD_NAMES
from tests.conftest import make_program

SCHEMAS = list(MODEL_RESPONSE_SCHEMAS.items())
SCHEMA_IDS = [task.value for task, _ in SCHEMAS]


def source() -> ProgramSource:
    return make_program().sources[0]


# ------------------------------------------------------------ the guarantees
def test_every_task_has_exactly_one_schema():
    assert set(MODEL_RESPONSE_SCHEMAS) == set(GeminiTask)


@pytest.mark.parametrize(("task", "schema"), SCHEMAS, ids=SCHEMA_IDS)
def test_no_schema_can_express_a_match_status(task, schema):
    rendered = json.dumps(schema.model_json_schema())
    for status in MatchStatus:
        assert status.value not in rendered, f"{schema.__name__} can emit {status.value}"
    assert "MatchStatus" not in rendered
    assert "status" not in schema.model_fields


@pytest.mark.parametrize(("task", "schema"), SCHEMAS, ids=SCHEMA_IDS)
def test_every_schema_converts_for_the_real_sdk(task, schema):
    """Exercises google-genai's own schema transformer, offline."""
    client = genai.Client(api_key="not-a-real-key")
    converted = _transformers.t_schema(client._api_client, schema)

    assert converted is not None
    assert converted.type is not None
    rendered = converted.model_dump_json(exclude_none=True)
    for status in MatchStatus:
        assert status.value not in rendered


@pytest.mark.parametrize(
    "value",
    [
        ExtractionResponse(unknown_fields=["age", "state"], ambiguities=["could mean two things"]),
        ClarificationResponse(questions=[], rationale="nothing is worth asking"),
        ExplanationResponse(headline="x", what_we_still_need=["How many people live with you?"]),
        ActionPlanResponse(title="t", summary="s", items=[]),
        ChatAnswerResponse(can_answer=False, answer="The sources here do not say."),
    ],
    ids=SCHEMA_IDS,
)
def test_not_knowing_is_always_representable(value):
    assert value is not None


# ------------------------------------------------------------- extraction
def test_extraction_projects_onto_the_contract_type():
    response = ExtractionResponse(
        state="nj",
        county="Middlesex",
        household_size=3,
        household_composition=HouseholdCompositionOut(adults=1, children_under_18=2),
        income_min_cents=190_000,
        income_max_cents=210_000,
        income_period=IncomePeriod.MONTHLY,
        needs=[NeedCategory.FOOD, NeedCategory.FOOD, NeedCategory.UTILITIES],
        age=34,
        flags=[ExtractedFlagOut(flag=ProfileFlag.HAS_CHILDREN, value=True, confidence=1.0)],
        field_evidence=[FieldEvidenceOut(field="state", confidence=1.0, source_span="NJ")],
        unknown_fields=["age"],
        ambiguities=["  about $2,000 a month  "],
    )

    extracted, dropped = response.to_extracted_profile()

    assert dropped == []
    assert extracted.state == "NJ"
    assert extracted.needs == [NeedCategory.FOOD, NeedCategory.UTILITIES]
    assert extracted.household_composition is not None
    assert extracted.household_composition.adults == 1
    assert extracted.ambiguities == ["about $2,000 a month"]
    assert extracted.to_profile_fields()["household_size"] == 3


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"household_size": 0}, "household_size:out_of_range"),
        ({"household_size": 999}, "household_size:out_of_range"),
        ({"age": -4}, "age:out_of_range"),
        ({"age": 400}, "age:out_of_range"),
        ({"state": "New Jersey"}, "state:not_a_state_code"),
        ({"income_min_cents": 100_000}, "income:amount_without_period"),
        (
            {
                "income_min_cents": 300_000,
                "income_max_cents": 100_000,
                "income_period": IncomePeriod.MONTHLY,
            },
            "income:min_above_max",
        ),
        (
            {"income_min_cents": -5, "income_period": IncomePeriod.MONTHLY},
            "income_min_cents:negative",
        ),
        ({"unknown_fields": ["ssn"]}, "unknown_fields:unknown_field"),
    ],
)
def test_extraction_drops_what_it_cannot_honour(kwargs, reason):
    response = ExtractionResponse(**kwargs)
    extracted, dropped = response.to_extracted_profile()

    assert reason in dropped
    assert extracted is not None


def test_extraction_drops_evidence_for_an_unknown_field():
    response = ExtractionResponse(
        field_evidence=[
            FieldEvidenceOut(field="eligibility_status", confidence=1.0),
            FieldEvidenceOut(field="household_size", confidence=0.9),
        ]
    )

    extracted, dropped = response.to_extracted_profile()

    assert [item.field for item in extracted.field_evidence] == ["household_size"]
    assert any("unknown_field" in reason for reason in dropped)
    assert all(item.field in PROFILE_FIELD_NAMES for item in extracted.field_evidence)


def test_extraction_clamps_confidence_into_range():
    response = ExtractionResponse(
        flags=[ExtractedFlagOut(flag=ProfileFlag.IS_VETERAN, value=True, confidence=4.2)],
        field_evidence=[FieldEvidenceOut(field="age", confidence=-1.0)],
    )

    extracted, _ = response.to_extracted_profile()

    assert extracted.flags[0].confidence == 1.0
    assert extracted.field_evidence[0].confidence == 0.0


# ---------------------------------------------------------- clarification
def question(**overrides: Any) -> ClarificationQuestionOut:
    base: dict[str, Any] = {
        "field": "household_size",
        "question": "How many people live with you?",
        "input_kind": ClarificationInputKind.NUMBER,
        "why_it_matters": "It decides what the program can look at.",
        "affected_program_slugs": ["example-grocery-help"],
    }
    base.update(overrides)
    return ClarificationQuestionOut(**base)


def project(
    response: ClarificationResponse, **kwargs: Any
) -> tuple[ClarificationQuestionSet, list[str]]:
    defaults: dict[str, Any] = {
        "id_for": lambda field: f"ask-{field}",
        "allowed_fields": ["household_size", "age"],
        "allowed_slugs": ["example-grocery-help"],
        "max_questions": 3,
    }
    defaults.update(kwargs)
    return response.to_question_set(**defaults)


def test_clarification_projection_assigns_ids_and_filters_slugs():
    response = ClarificationResponse(
        questions=[question(affected_program_slugs=["example-grocery-help", "program-999"])]
    )

    result, dropped = project(response)

    assert [q.id for q in result.questions] == ["ask-household_size"]
    assert result.questions[0].affected_program_ids == ["example-grocery-help"]
    assert dropped == []


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (ClarificationResponse(questions=[question(field="ssn")]), "question:unknown_field"),
        (ClarificationResponse(questions=[question(field="county")]), "question:field_not_missing"),
        (
            ClarificationResponse(questions=[question(), question()]),
            "question:duplicate_field",
        ),
        (
            ClarificationResponse(
                questions=[question(question="Are you eligible for this program?")]
            ),
            "question:banned_vocabulary",
        ),
        (
            ClarificationResponse(questions=[question(question="Hi?", why_it_matters="Yes")]),
            "question:text_too_short",
        ),
        (
            ClarificationResponse(
                questions=[
                    question(
                        input_kind=ClarificationInputKind.SINGLE_SELECT,
                        options=[ClarificationOptionOut(value="1", label="One")],
                    )
                ]
            ),
            "question:too_few_options",
        ),
        (
            ClarificationResponse(questions=[question(min_value=10, max_value=1)]),
            "question:bad_bounds",
        ),
    ],
)
def test_clarification_projection_drops_bad_questions(response, reason):
    _, dropped = project(response)
    assert reason in dropped


def test_clarification_projection_respects_the_cap():
    response = ClarificationResponse(
        questions=[question(), question(field="age", question="How old are you?")]
    )

    result, dropped = project(response, max_questions=1)

    assert len(result.questions) == 1
    assert "question:over_cap" in dropped


def test_clarification_projection_strips_options_from_a_numeric_question():
    response = ClarificationResponse(
        questions=[question(options=[ClarificationOptionOut(value="1", label="One")])]
    )

    result, dropped = project(response)

    assert result.questions[0].options == []
    assert "question:options_on_non_select" in dropped


# ------------------------------------------------------------- explanation
def test_explanation_merges_required_caveats_and_dedupes():
    response = ExplanationResponse(
        headline="  This program looks worth applying for.  ",
        why_it_may_fit=["You live in New Jersey.", "You live in New Jersey."],
        caveats=["Rules change."],
    )

    value = response.to_match_explanation(
        prompt_version="v1", required_caveats=["Rules change.", "The office decides."]
    )

    assert value.headline == "This program looks worth applying for."
    assert value.why_it_may_fit == ["You live in New Jersey."]
    assert value.caveats == ["Rules change.", "The office decides."]
    assert value.generated_by == "gemini"
    assert value.prompt_version == "v1"


@pytest.mark.parametrize(
    "response",
    [
        ExplanationResponse(headline="You are eligible for this program."),
        ExplanationResponse(headline="Good news.", why_it_may_fit=["You qualify."]),
        ExplanationResponse(headline="Good news.", caveats=["You were denied last year."]),
    ],
)
def test_explanation_detects_banned_vocabulary_anywhere(response):
    assert response.banned_term() is not None


def test_explanation_without_a_headline_is_unusable():
    with pytest.raises(ValueError, match="no headline"):
        ExplanationResponse(headline="   ").to_match_explanation(prompt_version="v1")


# ------------------------------------------------------------- action plan
def item(**overrides: Any) -> ActionItemDraftOut:
    base: dict[str, Any] = {
        "program_slug": "example-grocery-help",
        "kind": ActionItemKind.DOCUMENT,
        "subject": "proof of address",
        "title": "Find a utility bill",
        "detail": "The application asks where you live.",
        "url": None,
    }
    base.update(overrides)
    return ActionItemDraftOut(**base)


def test_action_plan_projection_keeps_catalog_urls_only():
    response = ActionPlanResponse(
        title="Your next steps",
        summary="Start with the document.",
        items=[
            item(url="https://example.org/example-grocery-help/apply"),
            item(subject="photo id", url="https://evil.example.net/phish"),
            item(program_slug="program-999"),
        ],
    )

    draft, dropped = response.to_draft(
        allowed_slugs=["example-grocery-help"],
        allowed_urls=["https://example.org/example-grocery-help/apply"],
    )

    assert len(draft.items) == 2
    assert draft.items[0].url == "https://example.org/example-grocery-help/apply"
    assert draft.items[1].url is None
    assert "item:url_not_in_catalog" in dropped
    assert "item:unknown_program" in dropped


def test_action_plan_projection_rejects_a_banned_summary():
    response = ActionPlanResponse(
        title="Your next steps", summary="You are eligible, so just apply."
    )

    with pytest.raises(ValueError, match="banned vocabulary"):
        response.to_draft(allowed_slugs=["example-grocery-help"])


def test_action_plan_projection_drops_a_banned_item():
    response = ActionPlanResponse(
        title="Your next steps",
        summary="Start with the document.",
        items=[item(detail="You qualify, so skip the paperwork."), item(subject="photo id")],
    )

    draft, dropped = response.to_draft(allowed_slugs=["example-grocery-help"])

    assert [i.subject for i in draft.items] == ["photo id"]
    assert "item:banned_vocabulary" in dropped


# --------------------------------------------------------------------- chat
def test_chat_projection_resolves_citations_against_the_stored_source():
    stored = source()
    response = ChatAnswerResponse(
        can_answer=True,
        answer="The source says it states no real requirement.",
        facts=[
            FactStatementOut(text="The source states no real requirement.", source_ids=[stored.id])
        ],
        citations=[
            CitationOut(source_id=stored.id, excerpt="  it DESCRIBES no real program  "),
        ],
        confidence=ConfidenceLevel.MEDIUM,
    )

    value, dropped = response.to_chat_response(known_sources=[stored])

    assert dropped == []
    assert value.is_grounded is True
    assert value.citations[0].url == stored.url
    assert value.citations[0].title == stored.title
    assert value.facts[0].source_ids == [stored.id]
    assert value.confidence is ConfidenceLevel.MEDIUM


def test_chat_projection_drops_unknown_sources_and_demotes_facts():
    stored = source()
    response = ChatAnswerResponse(
        can_answer=True,
        answer="There is a priority rule.",
        facts=[FactStatementOut(text="There is a priority rule.", source_ids=["999"])],
        citations=[CitationOut(source_id="999", excerpt="priority applicants first")],
        confidence=ConfidenceLevel.HIGH,
    )

    value, dropped = response.to_chat_response(known_sources=[stored])

    assert value.citations == []
    assert value.facts == []
    assert value.interpretation == ["There is a priority rule."]
    assert value.is_grounded is False
    assert value.ungrounded_notice is not None
    assert value.confidence is ConfidenceLevel.LOW
    assert "citation:unknown_source" in dropped
    assert "fact:demoted_to_interpretation" in dropped
    assert "confidence:downgraded_ungrounded" in dropped


def test_chat_projection_drops_an_excerpt_that_is_not_in_the_source():
    stored = source()
    response = ChatAnswerResponse(
        can_answer=True,
        answer="It says something else.",
        citations=[CitationOut(source_id=stored.id, excerpt="a sentence that is not there")],
    )

    value, dropped = response.to_chat_response(known_sources=[stored])

    assert value.citations == []
    assert "citation:excerpt_not_found" in dropped


def test_chat_projection_drops_a_duplicate_citation():
    stored = source()
    excerpt = "It describes no real program"
    response = ChatAnswerResponse(
        can_answer=True,
        answer="Twice.",
        citations=[
            CitationOut(source_id=stored.id, excerpt=excerpt),
            CitationOut(source_id=stored.id, excerpt=excerpt),
        ],
    )

    value, dropped = response.to_chat_response(known_sources=[stored])

    assert len(value.citations) == 1
    assert "citation:duplicate" in dropped


def test_chat_projection_supplies_wording_when_the_model_gives_none():
    response = ChatAnswerResponse(can_answer=False, answer="   ")

    value, _ = response.to_chat_response(known_sources=[])

    assert value.answer.startswith("We could not find this in the official sources")
    assert value.is_grounded is False


def test_chat_projection_caps_next_steps():
    stored = source()
    response = ChatAnswerResponse(
        can_answer=True,
        answer="Many steps.",
        suggested_next_steps=[f"step {index}" for index in range(10)],
    )

    value, _ = response.to_chat_response(known_sources=[stored])

    assert len(value.suggested_next_steps) == 5


def test_stale_source_freshness_is_not_invented_by_the_projection():
    """Freshness comes from the stored record, like the url and the title."""
    stored = source().model_copy(update={"freshness": SourceFreshness.STALE})
    response = ChatAnswerResponse(
        can_answer=True,
        answer="Old source.",
        citations=[CitationOut(source_id=stored.id, excerpt="It describes no real program")],
    )

    value, _ = response.to_chat_response(known_sources=[stored])

    assert value.citations[0].source_id == stored.id
