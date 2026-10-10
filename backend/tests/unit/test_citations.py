"""Unit tests for citation verification and for scrubbing model prose.

The whole point of this module under test is that it fails closed, so most of
these tests are negative: an id we do not hold, a source belonging to a
different program, a quotation that was altered, a link nobody published. Each
one must be *dropped and recorded*, never passed through, because a citation
that survives verification is shown to a nontechnical person as evidence.

The verifier is deterministic and synchronous at its core
(:func:`app.retrieval.citations.verify_against`), which is why these tests need
no database. ``tests/integration/test_retrieval_db.py`` covers the same rules
resolving against real rows.

The sources below are fictional. The excerpt text is invented, and the one
place it reads like official wording ("must meet certain requirements") is
there because real source excerpts contain the words "eligible" and "qualify" -
which our *own* prose may never contain, and the tests check that the
difference is respected.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from app.retrieval.citations import (
    EXCERPT_MATCH_RATIO,
    MIN_EXCERPT_CHARS,
    CatalogSource,
    CitationReport,
    SourceCitationVerifier,
    detect_guarantee_phrasing,
    excerpt_matches,
    has_guarantee_phrasing,
    normalise_excerpt,
    split_sentences,
    strip_unsupported_claims,
    verify_against,
    verify_citations,
    verify_claims,
)
from app.schemas import (
    ChatResponse,
    Citation,
    CitationVerificationReport,
    FactStatement,
    ProgramRepository,
    ProgramSource,
    SourceFreshness,
    program_uuid,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)

PROGRAM_A = program_uuid("example-grocery-help")
PROGRAM_B = program_uuid("example-rent-help")

STORED_EXCERPT = (
    "Your household must meet certain requirements, including the maximum "
    "allowable monthly income of $2,461 for a household of one. Applications "
    "are processed by the county office."
)


def _source(
    source_id: str = "example-grocery-help-source",
    *,
    url: str = "https://example.org/example-grocery-help/requirements",
    excerpt: str = STORED_EXCERPT,
    age_days: int = 5,
) -> ProgramSource:
    return ProgramSource(
        id=source_id,
        url=url,
        title="Example requirements page (fictional test fixture)",
        publisher="Example Test Publisher",
        excerpt=excerpt,
        retrieved_at=NOW - timedelta(days=age_days),
        content_hash=hashlib.sha256(source_id.encode()).hexdigest(),
        # Deliberately always FRESH: the verifier must recompute freshness from
        # retrieved_at rather than trust this snapshot.
        freshness=SourceFreshness.FRESH,
    )


def _catalog(
    source: ProgramSource | None = None, *, program_id: uuid.UUID | None = PROGRAM_A
) -> CatalogSource:
    return CatalogSource(source=source or _source(), program_id=program_id)


def _citation(
    *,
    source_id: str = "example-grocery-help-source",
    program_id: uuid.UUID | None = PROGRAM_A,
    url: str = "https://example.org/example-grocery-help/requirements",
    excerpt: str = "the maximum allowable monthly income of $2,461 for a household of one",
) -> Citation:
    return Citation(
        source_id=source_id,
        program_id=program_id,
        url=url,
        title="Example requirements page",
        excerpt=excerpt,
    )


def _verify(
    citations: Sequence[Citation],
    sources: Sequence[CatalogSource | ProgramSource] | None = None,
) -> CitationVerificationReport:
    return verify_against(citations, sources if sources is not None else [_catalog()], now=NOW)


class TestAValidCitation:
    def test_a_verbatim_quotation_of_a_real_source_passes(self) -> None:
        report = _verify([_citation()])
        assert report.ok is True
        assert [citation.source_id for citation in report.verified] == [
            "example-grocery-help-source"
        ]
        assert report.rejected == []

    def test_the_sources_it_checked_against_are_reported(self) -> None:
        """So a reviewer can see what the decision was made against."""
        report = _verify([_citation()])
        assert [source.id for source in report.checked_sources] == ["example-grocery-help-source"]

    def test_a_citation_with_no_program_attribution_is_still_checkable(self) -> None:
        """Attribution is only verified when the citation makes a claim about it."""
        report = _verify([_citation(program_id=None)], [_source()])
        assert report.ok is True

    def test_a_truncated_quotation_passes(self) -> None:
        report = _verify([_citation(excerpt="the maximum allowable monthly income of $2,461")])
        assert report.ok is True

    def test_re_wrapped_whitespace_passes(self) -> None:
        quoted = "the maximum allowable\n  monthly income   of $2,461"
        assert _verify([_citation(excerpt=quoted)]).ok is True

    def test_smart_quotes_and_dashes_are_folded_before_comparing(self) -> None:
        stored = _source(
            excerpt="The county office said \u201cbring proof of address\u2014today\u201d."
        )
        quoted = 'The county office said "bring proof of address-today".'
        assert _verify([_citation(excerpt=quoted)], [_catalog(stored)]).ok is True

    def test_case_differences_do_not_reject_a_real_quotation(self) -> None:
        quoted = "THE MAXIMUM ALLOWABLE MONTHLY INCOME OF $2,461"
        assert _verify([_citation(excerpt=quoted)]).ok is True

    def test_a_trailing_slash_or_host_case_does_not_reject_a_real_url(self) -> None:
        citation = _citation(url="https://EXAMPLE.org/example-grocery-help/requirements/")
        assert _verify([citation]).ok is True


class TestAnUnknownSource:
    def test_an_id_the_catalog_does_not_hold_is_rejected(self) -> None:
        report = _verify([_citation(source_id="totally-invented-source")])
        assert report.ok is False
        assert report.verified == []
        assert [rejection.reason for rejection in report.rejected] == ["unknown_source"]

    def test_the_rejection_names_the_id_that_failed(self) -> None:
        report = _verify([_citation(source_id="totally-invented-source")])
        assert report.rejected[0].source_id == "totally-invented-source"
        assert report.rejected[0].detail is not None

    def test_with_no_sources_at_all_nothing_is_let_through(self) -> None:
        report = _verify([_citation()], [])
        assert report.ok is False
        assert report.rejected[0].reason == "unknown_source"


class TestWrongProgramAttribution:
    def test_a_source_belonging_to_another_program_is_rejected(self) -> None:
        report = _verify([_citation(program_id=PROGRAM_B)], [_catalog(program_id=PROGRAM_A)])
        assert report.ok is False
        assert report.rejected[0].reason == "program_mismatch"

    def test_a_source_whose_program_is_unknown_cannot_be_attributed(self) -> None:
        """Fail closed: unverifiable attribution is not the same as correct."""
        report = _verify([_citation(program_id=PROGRAM_A)], [_catalog(program_id=None)])
        assert report.ok is False
        assert report.rejected[0].reason == "program_mismatch"


class TestAMangledExcerpt:
    def test_an_altered_number_is_rejected(self) -> None:
        """The single most dangerous kind of near-miss on a benefits page."""
        report = _verify(
            [_citation(excerpt="the maximum allowable monthly income of $9,999 for a household")]
        )
        assert report.ok is False
        assert report.rejected[0].reason == "excerpt_not_found"

    def test_an_invented_sentence_is_rejected(self) -> None:
        report = _verify(
            [_citation(excerpt="Everyone in New Jersey receives this benefit automatically.")]
        )
        assert report.ok is False
        assert report.rejected[0].reason == "excerpt_not_found"

    def test_reordered_words_are_rejected(self) -> None:
        report = _verify(
            [_citation(excerpt="income of $2,461 allowable the maximum monthly for a household")]
        )
        assert report.ok is False
        assert report.rejected[0].reason == "excerpt_not_found"

    def test_a_quotation_too_short_to_be_evidence_is_rejected(self) -> None:
        report = _verify([_citation(excerpt="income")])
        assert report.ok is False
        assert report.rejected[0].reason == "excerpt_not_found"
        assert str(MIN_EXCERPT_CHARS) in (report.rejected[0].detail or "")

    def test_the_match_threshold_is_strict_enough_to_catch_a_swapped_word(self) -> None:
        assert EXCERPT_MATCH_RATIO >= 0.9
        assert excerpt_matches("the county office processes applications", STORED_EXCERPT) is False


class TestUrlAndFreshness:
    def test_a_different_url_on_the_same_source_is_rejected(self) -> None:
        report = _verify([_citation(url="https://example.org/somewhere-else")])
        assert report.ok is False
        assert report.rejected[0].reason == "url_mismatch"

    def test_a_source_nobody_has_re_checked_in_a_year_is_rejected(self) -> None:
        report = _verify([_citation()], [_catalog(_source(age_days=400))])
        assert report.ok is False
        assert report.rejected[0].reason == "stale_source"

    def test_an_aging_source_is_still_usable(self) -> None:
        """Between 90 days and a year a source is caveated, not discarded."""
        report = _verify([_citation()], [_catalog(_source(age_days=120))])
        assert report.ok is True


class TestReportBookkeeping:
    def test_identical_citations_are_collapsed(self) -> None:
        report = _verify([_citation(), _citation()])
        assert len(report.verified) == 1

    def test_a_repeated_bad_citation_is_only_rejected_once(self) -> None:
        bad = _citation(source_id="totally-invented-source")
        report = _verify([bad, bad])
        assert len(report.rejected) == 1

    def test_ok_is_false_whenever_anything_was_rejected(self) -> None:
        report = _verify([_citation(), _citation(source_id="totally-invented-source")])
        assert report.ok is False
        assert len(report.verified) == 1
        assert len(report.rejected) == 1

    def test_no_citations_at_all_is_vacuously_ok(self) -> None:
        """Refusing to answer is sometimes correct; it is not a failure."""
        report = _verify([])
        assert report.ok is True
        assert report.unsupported_claim_rate == 0.0

    def test_the_report_alias_is_the_contract_type(self) -> None:
        assert CitationReport is CitationVerificationReport


def _response(
    *,
    answer: str = "The county office processes applications for this fictional program.",
    citations: Sequence[Citation] = (),
    facts: Sequence[FactStatement] = (),
) -> ChatResponse:
    return ChatResponse(
        answer=answer,
        facts=list(facts),
        citations=list(citations),
    )


class TestWholeResponses:
    def test_a_fact_resting_on_a_rejected_citation_becomes_unsupported(self) -> None:
        response = _response(
            answer="Here is what the official page says.",
            citations=[_citation(source_id="totally-invented-source")],
            facts=[
                FactStatement(
                    text="The monthly income limit is $2,461 for one person.",
                    source_ids=["totally-invented-source"],
                )
            ],
        )
        report = verify_claims(response, [_catalog()], now=NOW)
        assert report.ok is False
        assert report.unsupported_claims == ["The monthly income limit is $2,461 for one person."]

    def test_partial_support_counts_as_no_support(self) -> None:
        response = _response(
            citations=[_citation(), _citation(source_id="totally-invented-source")],
            facts=[
                FactStatement(
                    text="Two things are true at once.",
                    source_ids=["example-grocery-help-source", "totally-invented-source"],
                )
            ],
        )
        report = verify_claims(response, [_catalog()], now=NOW)
        assert report.unsupported_claims == ["Two things are true at once."]

    def test_a_fully_cited_fact_survives(self) -> None:
        response = _response(
            citations=[_citation()],
            facts=[
                FactStatement(
                    text="Applications are processed by the county office.",
                    source_ids=["example-grocery-help-source"],
                )
            ],
        )
        report = verify_claims(response, [_catalog()], now=NOW)
        assert report.ok is True
        assert report.unsupported_claims == []

    def test_guarantee_phrasing_in_the_answer_is_reported_even_when_cited(self) -> None:
        """No source can support a promise about an outcome."""
        response = _response(
            answer="You qualify for this program. The county office takes applications.",
            citations=[_citation()],
        )
        report = verify_claims(response, [_catalog()], now=NOW)
        assert report.ok is False
        assert report.unsupported_claims == ["You qualify for this program."]


class TestTheSessionlessPath:
    async def test_known_sources_are_used_when_there_is_no_session(self) -> None:
        report = await verify_citations(
            None, [_citation(program_id=None)], known_sources=[_source()], now=NOW
        )
        assert report.ok is True

    async def test_without_a_session_or_known_sources_everything_is_rejected(self) -> None:
        report = await verify_citations(None, [_citation()], now=NOW)
        assert report.ok is False
        assert report.rejected[0].reason == "unknown_source"

    async def test_a_whole_response_can_be_verified_without_a_session(self) -> None:
        response = _response(citations=[_citation(program_id=None)])
        report = await verify_citations(None, response, known_sources=[_source()], now=NOW)
        assert report.ok is True


class _StubRepository:
    """The one ``ProgramRepository`` method the verifier needs, and nothing else.

    Stands in for the in-memory repository to prove the no-session path really
    resolves through the repository - and that resolving through it settles
    attribution, because ``get_sources`` filters by ``program_id``.
    """

    def __init__(self, sources: dict[uuid.UUID, list[ProgramSource]]) -> None:
        self._sources = sources
        self.calls = 0

    async def get_sources(
        self, program_id: uuid.UUID, *, source_ids: Sequence[str] | None = None
    ) -> list[ProgramSource]:
        self.calls += 1
        found = self._sources.get(program_id, [])
        if source_ids is None:
            return list(found)
        wanted = set(source_ids)
        return [source for source in found if source.id in wanted]


class TestTheRepositoryPath:
    async def test_a_citation_is_resolved_through_the_repository(self) -> None:
        repository = _StubRepository({PROGRAM_A: [_source()]})
        verifier = SourceCitationVerifier(cast("ProgramRepository", repository))
        report = await verifier.verify([_citation()])
        assert report.ok is True
        assert repository.calls == 1

    async def test_attribution_to_the_wrong_program_finds_nothing(self) -> None:
        repository = _StubRepository({PROGRAM_A: [_source()]})
        verifier = SourceCitationVerifier(cast("ProgramRepository", repository))
        report = await verifier.verify([_citation(program_id=PROGRAM_B)])
        assert report.ok is False
        assert report.rejected[0].reason == "unknown_source"

    async def test_one_query_per_cited_program_not_per_citation(self) -> None:
        repository = _StubRepository(
            {PROGRAM_A: [_source(), _source("example-grocery-help-second")]}
        )
        verifier = SourceCitationVerifier(cast("ProgramRepository", repository))
        report = await verifier.verify(
            [_citation(), _citation(source_id="example-grocery-help-second")]
        )
        assert repository.calls == 1
        assert len(report.verified) == 2
        assert report.ok is True

    async def test_known_sources_fill_a_gap_the_repository_cannot(self) -> None:
        """Last resort, and only for a citation that attributes no program."""
        repository = _StubRepository({})
        verifier = SourceCitationVerifier(cast("ProgramRepository", repository))
        report = await verifier.verify([_citation(program_id=None)], known_sources=[_source()])
        assert report.ok is True

    async def test_a_response_is_verified_through_the_repository_too(self) -> None:
        repository = _StubRepository({PROGRAM_A: [_source()]})
        verifier = SourceCitationVerifier(cast("ProgramRepository", repository))
        report = await verifier.verify_response(_response(citations=[_citation()]))
        assert report.ok is True


class TestGuaranteeDetection:
    @pytest.mark.parametrize(
        "sentence",
        [
            "You qualify for SNAP based on what you told us.",
            "You will qualify once you apply.",
            "You are eligible for this program.",
            "You're eligible for help with rent.",
            "Most students are eligible for this grant.",
            "You are ineligible because of your income.",
            "Approval is guaranteed if you bring your documents.",
            "Your application is guaranteed to succeed.",
            "You have been pre-approved for this benefit.",
            "You will be approved within two weeks.",
            "This program is already approved for you.",
            "Households of three qualify for the maximum amount.",
        ],
    )
    def test_outcome_claims_are_flagged(self, sentence: str) -> None:
        flags = detect_guarantee_phrasing(sentence)
        assert len(flags) == 1
        assert flags[0].sentence == sentence
        assert flags[0].phrases
        assert has_guarantee_phrasing(sentence) is True

    @pytest.mark.parametrize(
        "sentence",
        [
            "You may be able to get help with groceries.",
            "The county office decides who receives benefits.",
            "Bring proof of address and a recent pay stub.",
            "We still need to know your household size.",
            "This program is run by the Example County Office of Testing.",
        ],
    )
    def test_hedged_product_language_is_not_flagged(self, sentence: str) -> None:
        assert detect_guarantee_phrasing(sentence) == []
        assert has_guarantee_phrasing(sentence) is False

    def test_only_the_offending_sentence_is_flagged(self) -> None:
        text = (
            "Bring proof of address. You qualify for this program. "
            "The county office is open on weekdays."
        )
        flags = detect_guarantee_phrasing(text)
        assert [flag.sentence for flag in flags] == ["You qualify for this program."]

    def test_the_matched_phrase_is_reported_lowercased(self) -> None:
        flags = detect_guarantee_phrasing("Approval Is GUARANTEED here.")
        assert "guaranteed" in flags[0].phrases


class TestSentenceSplitting:
    def test_terminal_punctuation_and_newlines_both_split(self) -> None:
        assert split_sentences("One. Two!\nThree?") == ["One.", "Two!", "Three?"]

    def test_text_without_terminal_punctuation_is_one_sentence(self) -> None:
        assert split_sentences("no full stop here") == ["no full stop here"]

    def test_empty_text_has_no_sentences(self) -> None:
        assert split_sentences("   \n  ") == []


class TestStrippingUnsupportedClaims:
    def test_a_guarantee_sentence_is_removed(self) -> None:
        result = strip_unsupported_claims(
            "You qualify for this program. Bring proof of address.", [_source()]
        )
        assert result.text == "Bring proof of address."
        assert result.removed == ["You qualify for this program."]
        assert result.guarantee_flags[0].phrases

    def test_a_sentence_citing_an_unknown_source_is_removed(self) -> None:
        result = strip_unsupported_claims(
            "The limit is $2,461 [invented-source]. The office is open on weekdays.",
            ["example-grocery-help-source"],
        )
        assert result.text == "The office is open on weekdays."
        assert result.removed_source_ids == ["invented-source"]

    def test_a_sentence_citing_an_allowed_source_is_kept(self) -> None:
        text = "The limit is $2,461 [example-grocery-help-source]."
        result = strip_unsupported_claims(text, [_source()])
        assert result.text == text
        assert result.changed is False

    def test_the_source_marker_form_with_a_label_is_understood(self) -> None:
        result = strip_unsupported_claims(
            "Applications go to the county office (source: invented-source).", [_source()]
        )
        assert result.text == ""
        assert result.removed_source_ids == ["invented-source"]

    def test_a_link_nobody_published_is_removed(self) -> None:
        result = strip_unsupported_claims(
            "Apply at https://not-a-real-site.example.com/apply now.", [_source()]
        )
        assert result.text == ""
        assert result.removed_urls == ["https://not-a-real-site.example.com/apply"]

    def test_a_link_an_allowed_source_published_is_kept(self) -> None:
        text = "Read the rules at https://example.org/example-grocery-help/requirements."
        result = strip_unsupported_claims(text, [_source()])
        assert result.text == text

    def test_extra_allowed_urls_cover_links_that_are_not_source_documents(self) -> None:
        text = "Apply at https://example.org/example-grocery-help/apply."
        result = strip_unsupported_claims(
            text, [_source()], allowed_urls=["https://example.org/example-grocery-help/apply"]
        )
        assert result.text == text

    def test_source_objects_and_bare_ids_are_both_accepted(self) -> None:
        text = "The limit is $2,461 [example-grocery-help-source]."
        from_objects = strip_unsupported_claims(text, [_source()])
        from_ids = strip_unsupported_claims(text, ["example-grocery-help-source"])
        assert from_objects.text == from_ids.text == text

    def test_removing_everything_is_an_allowed_outcome(self) -> None:
        result = strip_unsupported_claims("You are eligible. You are approved.", [_source()])
        assert result.text == ""
        assert len(result.removed) == 2

    def test_catalog_sources_are_accepted_as_an_allowlist(self) -> None:
        text = "The limit is $2,461 [example-grocery-help-source]."
        assert strip_unsupported_claims(text, [_catalog()]).text == text


class TestNormalisation:
    def test_whitespace_and_case_are_collapsed(self) -> None:
        assert normalise_excerpt("  The   COUNTY\n office ") == "the county office"

    def test_a_non_breaking_space_becomes_a_space(self) -> None:
        assert normalise_excerpt("county\u00a0office") == "county office"

    def test_an_ellipsis_character_is_folded(self) -> None:
        assert normalise_excerpt("income\u2026") == "income..."
