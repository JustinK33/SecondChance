"""Source-grounded chat.

The assistant is the most dangerous surface in this product. A user asking
"does my income count?" will act on the answer, and a fluent wrong answer is
worse than no answer at all - it costs them an application, a deadline, or a
month of waiting for a denial. So the pipeline is built so that an ungrounded
sentence cannot reach the user:

1. **Retrieve** official sources for the question. Scoped to one program when
   the client names one, and the ownership of a referenced run is checked
   server-side before it is used to scope anything.
2. **Generate** with the grounding prompt, which is handed those sources and
   told to answer from them alone.
3. **Verify** every citation deterministically through the
   :class:`~app.schemas.contracts.CitationVerifier`: the id must name a real
   stored source, the excerpt must actually appear in it, the URL must match
   and the source must not be stale. No model participates in that check,
   because asking the generator to grade itself is not verification.
4. **Strip** every sentence the verified sources do not support, along with
   anything that promises an outcome. Sentence-level, because deleting a
   citation marker from the middle of a sentence leaves fluent text that now
   says something nobody wrote.
5. **Separate** what is left into cited facts and explicitly-labelled
   interpretation, and add the caveats the stripping produced.

If nothing survives, the answer says so plainly. That is the behaviour under
test: an answer with no verified citation must state that we could not find it
in an official source, and must not assert anything in the user's situation.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from itertools import pairwise

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.concurrency import deadline, retry_async
from app.core.errors import AppError
from app.core.logging import get_logger
from app.core.metrics import get_metrics
from app.db.repositories import (
    AuditAction,
    AuditRepository,
    SqlProgramRepository,
    WorkflowRunRepository,
)
from app.retrieval.citations import SourceCitationVerifier, strip_unsupported_claims
from app.retrieval.search import HybridRetriever
from app.schemas import (
    UNGROUNDED_NOTICE,
    ChatRequest,
    ChatResponse,
    Citation,
    ConfidenceLevel,
    FactStatement,
    GeminiPort,
    ProgramSource,
    Retriever,
)
from app.services.ai_port import provide_ai_port

__all__ = ["SOURCE_LIMIT", "answer"]

_log = get_logger(__name__)

#: How many source passages to ground one answer in. Enough for a real answer,
#: few enough that the prompt stays small and every one of them is checkable.
SOURCE_LIMIT: int = 6

#: One retry on a transient provider failure, then an honest answer. A chat
#: turn is interactive: a user waiting on a spinner is better served by "try
#: again" than by a third attempt.
_ANSWER_ATTEMPTS: int = 2
#: Wall-clock budget for the whole generation, retries included.
_ANSWER_BUDGET_S: float = 25.0

#: Shortest token worth searching for.
_MIN_TERM_CHARS: int = 4
#: Searches per question. Each is a bounded substring scan.
_MAX_TERMS: int = 6

#: Words that match everything and therefore ground nothing.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "also",
        "anything",
        "apply",
        "because",
        "been",
        "being",
        "could",
        "does",
        "doing",
        "else",
        "even",
        "ever",
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
        "much",
        "must",
        "need",
        "only",
        "other",
        "really",
        "should",
        "some",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "this",
        "those",
        "very",
        "want",
        "well",
        "were",
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

#: Said when verification removed everything. Deliberately not an apology for a
#: technical failure: from the user's point of view the honest statement is that
#: we could not find this in an official source.
_NOTHING_VERIFIED = (
    "We could not confirm an answer to that in an official source, so we would rather not "
    "guess. The program's own page and phone number are the fastest way to get this "
    "confirmed."
)

#: Added when the model's prose had to be trimmed.
_TRIMMED_CAVEAT = (
    "We removed part of this answer because we could not trace it to an official source."
)


async def answer(
    session: AsyncSession,
    user_id: str,
    request: ChatRequest,
    *,
    request_id: str,
    gemini: GeminiPort | None = None,
) -> ChatResponse:
    """Answer one question from official sources only.

    ``request.run_id`` and ``request.program_id`` scope retrieval; neither is
    trusted as authorisation. A ``run_id`` the caller does not own is silently
    ignored rather than rejected, because the question is still answerable from
    the catalog and refusing would leak whether that run exists.
    """
    port = gemini if gemini is not None else provide_ai_port()
    repository = SqlProgramRepository(session)
    retriever = HybridRetriever(repository)

    program_id = await _scoped_program_id(session, user_id, request)
    sources = await _grounding_sources(
        retriever, repository, request.message, program_id=program_id
    )

    if not sources:
        _log.info("assistant.no_sources", scoped=program_id is not None)
        return _ungrounded(_NOTHING_VERIFIED)

    try:
        async with deadline(
            _ANSWER_BUDGET_S, operation_name="assistant.answer", observer=_observe_timeout
        ):
            result = await retry_async(
                lambda: port.answer_question(request, sources=sources, request_id=request_id),
                attempts=_ANSWER_ATTEMPTS,
                operation_name="assistant.answer",
                observer=_observe_retry,
            )
    except AppError:
        # The API layer maps this to 502/503/504. A half-generated answer is
        # never returned: there is nothing honest to say from one.
        _log.warning("assistant.generation_failed")
        raise

    drafted = result.value
    report = await SourceCitationVerifier(repository, session=session).verify_response(
        drafted, known_sources=sources
    )

    verified: list[Citation] = list(report.verified)
    verified_ids = {citation.source_id for citation in verified}
    if not verified_ids:
        _log.info(
            "assistant.answer.unverified",
            rejected=len(report.rejected),
            unsupported=len(report.unsupported_claims),
            source=result.usage.source,
        )
        await _record(session, user_id, request, request_id, grounded=False, citations=0)
        return _ungrounded(_NOTHING_VERIFIED)

    allowed_sources = [source for source in sources if source.id in verified_ids]
    scrubbed = strip_unsupported_claims(
        drafted.answer,
        allowed_sources,
        allowed_urls=[citation.url for citation in verified],
    )
    body = scrubbed.text.strip()

    facts = [
        FactStatement(text=fact.text, source_ids=supported)
        for fact in drafted.facts
        if (supported := [sid for sid in fact.source_ids if sid in verified_ids])
    ]
    interpretation = [
        line
        for line in (
            strip_unsupported_claims(
                entry, allowed_sources, allowed_urls=[c.url for c in verified]
            ).text.strip()
            for entry in drafted.interpretation
        )
        if line
    ]

    caveats = list(drafted.caveats)
    if scrubbed.removed or report.rejected or report.unsupported_claims:
        caveats.append(_TRIMMED_CAVEAT)
    if not body:
        # Everything in the prose was unsupported, but at least one citation
        # survived. Say the honest thing rather than returning an empty answer.
        body = _NOTHING_VERIFIED
        facts = []
        interpretation = []
        verified = []

    response = ChatResponse(
        answer=body,
        facts=facts,
        interpretation=interpretation,
        citations=verified,
        confidence=_confidence(drafted.confidence, report.ok, len(verified)),
        caveats=_dedupe(caveats),
        suggested_next_steps=drafted.suggested_next_steps[:5],
    )
    _log.info(
        "assistant.answered",
        source=result.usage.source,
        citations=len(response.citations),
        facts=len(response.facts),
        removed_sentences=len(scrubbed.removed),
        rejected=len(report.rejected),
        grounded=response.is_grounded,
    )
    await _record(
        session,
        user_id,
        request,
        request_id,
        grounded=response.is_grounded,
        citations=len(response.citations),
    )
    return response


async def _scoped_program_id(
    session: AsyncSession, user_id: str, request: ChatRequest
) -> uuid.UUID | None:
    """The program retrieval should be scoped to, after ownership checks."""
    if request.program_id is not None:
        return request.program_id
    if request.run_id is None:
        return None
    run = await WorkflowRunRepository(session).get(user_id, request.run_id)
    if run is None:
        # Not an error: the question stands on its own, and confirming whether
        # somebody else's run id exists is not information we owe the caller.
        _log.info("assistant.run_scope_ignored")
    return None


def _confidence(drafted: ConfidenceLevel, report_ok: bool, citation_count: int) -> ConfidenceLevel:
    """Confidence after verification, never above what verification supports."""
    if citation_count == 0:
        return ConfidenceLevel.UNKNOWN
    if not report_ok and drafted is ConfidenceLevel.HIGH:
        return ConfidenceLevel.MEDIUM
    return drafted


def _ungrounded(message: str) -> ChatResponse:
    """An answer that asserts nothing, with the standard notice attached."""
    return ChatResponse(
        answer=message,
        facts=[],
        interpretation=[],
        citations=[],
        confidence=ConfidenceLevel.UNKNOWN,
        caveats=[UNGROUNDED_NOTICE],
        suggested_next_steps=[],
    )


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value.strip()))


async def _record(
    session: AsyncSession,
    user_id: str,
    request: ChatRequest,
    request_id: str,
    *,
    grounded: bool,
    citations: int,
) -> None:
    """Audit that a question was answered. Counts and flags, never the text."""
    await AuditRepository(session).record(
        user_id,
        AuditAction.ASSISTANT_ANSWERED,
        resource_type="assistant",
        resource_id=request.program_id,
        run_id=request.run_id,
        request_id=request_id,
        metadata={"grounded": grounded, "citation_count": citations},
    )
    await session.commit()


# ---------------------------------------------------------------------------
# grounding retrieval
# ---------------------------------------------------------------------------
async def _grounding_sources(
    retriever: Retriever,
    repository: SqlProgramRepository,
    message: str,
    *,
    program_id: uuid.UUID | None,
) -> list[ProgramSource]:
    """Official sources to ground one question in.

    ``Retriever.search_sources`` matches a passage by substring, which is the
    right primitive - the assistant has to quote an excerpt back verbatim and
    the verifier has to find it again, so a stemmed match would return passages
    that do not contain the words the model was shown. But a whole question is
    never a substring of a government web page, so the question is reduced to
    its salient terms first and each is searched on its own.

    Sequentially, not concurrently: a ``Retriever`` is bound to one
    ``AsyncSession`` and a session is not safe for concurrent use. These are
    bounded substring scans over a small table, so the serial cost is
    negligible compared to opening a session per term.

    When the client named a program, that program's own sources are included
    even if none of them matched a term. A question asked on a program's page
    is about that program, and grounding it in the page the user is looking at
    is better than grounding it in nothing.
    """
    found: dict[str, ProgramSource] = {}
    for term in _search_terms(message):
        for source in await retriever.search_sources(
            term, program_id=program_id, limit=SOURCE_LIMIT
        ):
            found.setdefault(source.id, source)
        if len(found) >= SOURCE_LIMIT:
            break

    if program_id is not None and len(found) < SOURCE_LIMIT:
        for source in await repository.get_sources(program_id):
            found.setdefault(source.id, source)

    sources = list(found.values())[:SOURCE_LIMIT]
    _log.debug(
        "assistant.grounding",
        terms=len(_search_terms(message)),
        sources=len(sources),
        scoped=program_id is not None,
    )
    return sources


def _search_terms(message: str) -> list[str]:
    """Salient search terms from a question, longest-first.

    Adjacent informative words are kept as a phrase as well as individually,
    because "household income" matches a far more relevant passage than
    "income" alone, and a phrase that matches nothing simply contributes no
    sources.
    """
    words = [
        word
        for word in re.findall(r"[a-z0-9']+", message.lower())
        if len(word) >= _MIN_TERM_CHARS and word not in _STOPWORDS
    ]
    phrases = [f"{first} {second}" for first, second in pairwise(words)]
    ordered = sorted(dict.fromkeys([*phrases, *words]), key=len, reverse=True)
    return ordered[:_MAX_TERMS]


def _observe_retry(attempt: int, exc: BaseException, delay_s: float) -> None:
    get_metrics().increment_retry(operation="assistant.answer")


def _observe_timeout(operation: str) -> None:
    get_metrics().increment_timeout(operation=operation)
