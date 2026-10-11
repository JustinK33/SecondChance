"""The core flow, driven over real HTTP against a **running** API.

What this is for
----------------
``scripts.e2e_journey`` proves the product is correct: it builds the ASGI
application in-process, substitutes the model port for recorded fixtures and
asserts every transition of a 14-step journey. That is the right tool for CI
and the wrong tool for "is the thing I just deployed actually working", because
it never opens a socket and never touches the configuration the server was
started with.

This script is the other half. It talks to a server somebody else started, over
a real TCP connection, with whatever settings, database and model access that
process was given, and asserts that the core flow responds::

    liveness -> readiness -> unauthenticated access is refused -> catalog
    -> narrative becomes a profile -> a run is accepted and polled to
    completion -> every match carries an engine-decided status with evidence
    -> a plan with deduplicated steps -> a step is ticked -> a question is
    answered from cited sources -> the data is deleted and the deletion is
    verified

Run it with ``./do smoke``, after ``./do dev-api``::

    ./do smoke
    ./do smoke --base-url https://api.example.invalid --token "$ACCESS_TOKEN"

It is a *smoke* test, so the assertions are about shape, status codes and the
product's non-negotiable rules - not about particular values. The catalog it
reads is whatever the server has, the model may be live, and a correct server
is allowed to produce different wording every time. What it is not allowed to
do is claim eligibility, return a status the rule engine did not decide, cite a
source it did not retrieve, or leave a user's rows behind after a deletion.

Safety
------
The synthetic user is a fresh random UUID per run, its narrative is fictional,
and everything it creates is deleted through the API in the last step - so
running this against a real deployment adds no persistent data. Nothing a
person wrote and no amount is ever printed: the trace reports field *names*,
counts and statuses only.

Authentication
--------------
``AUTH_MODE=dev`` servers accept the ``X-Dev-User`` header, which is the
default here. Against an ``AUTH_MODE=supabase`` server, pass ``--token`` (or
``SMOKE_TOKEN``) with a real access token; the script sends it as a bearer
token and skips nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TextIO

import httpx

from app.core.security import DEV_USER_HEADER
from app.schemas import PROFILE_FIELD_NAMES, MatchStatus, WorkflowStatus
from app.workflows.state import MAX_CLARIFICATION_ROUNDS

__all__ = ["SmokeFailed", "SmokeResult", "main", "run_smoke"]

DEFAULT_BASE_URL: Final[str] = "http://127.0.0.1:8000"

#: Fictional, and phrased the way somebody actually writes it.
NARRATIVE: Final[str] = (
    "I live in New Brunswick, New Jersey with my two kids. I take classes at Rutgers and "
    "work part time, and money is tight this month. I need help with groceries and my "
    "electric bill."
)

QUESTION: Final[str] = "What documents do I need to apply, and where do I send them?"

#: Words the product may never use about a person's situation.
BANNED_WORDS: Final[tuple[str, ...]] = ("eligible", "ineligible", "you qualify", "guaranteed")

#: Keys whose value is copy *this product wrote* - a question, an explanation,
#: a step, an answer, a progress line, a caveat. These are what the vocabulary
#: rule governs. Kept deliberately in step with
#: ``scripts.e2e_journey.PRODUCT_COPY_KEYS``.
PRODUCT_COPY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ambiguities",
        "answer",
        "detail",
        "headline",
        "help_text",
        "interpretation",
        "label",
        "message",
        "progress_message",
        "question",
        "rationale",
        "retry_hint",
        "summary",
        "suggested_next_steps",
        "text",
        "title",
        "what_we_still_need",
        "why_it_matters",
        "why_it_may_fit",
    }
)

#: Subtrees that are *quoted* rather than written, and so are not scanned for
#: the status vocabulary:
#:
#: * ``program``, ``sources``, ``citations``, ``excerpt``, ``checked_sources``
#:   - the catalog record and the official passages behind it. A government
#:   page that says "you are eligible if..." has to be quotable verbatim;
#:   refusing to quote it would be a worse failure than using the word,
#:   because the alternative to a quote is a paraphrase nobody can check.
#: * ``limitations`` - passed through from the ``limitations`` a human curator
#:   wrote on the program's rule set (``backend/data/programs/*.json``), which
#:   restate the program's own published terms. "Households income eligible
#:   for USF are expected to apply for those first" is the state's phrasing of
#:   its own rule, not this product telling somebody they are eligible.
#: * ``caveats`` - by design a merge of the evaluation's curated limitations
#:   with the model's own caveats (``app/workflows/nodes/merge.py``), so it
#:   inherits the same quoted text. No coverage is lost by skipping it: the
#:   model-written half is rejected outright if it contains banned vocabulary,
#:   by ``app.ai.schemas.contains_banned_vocabulary``, before it is ever
#:   accepted.
#:
#: The catalog endpoints are quoted in their entirety and are not scanned at
#: all.
QUOTED_SUBTREE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "caveats",
        "checked_sources",
        "citations",
        "excerpt",
        "limitations",
        "program",
        "sources",
    }
)

_RUNNING: Final[frozenset[str]] = frozenset(
    {WorkflowStatus.QUEUED.value, WorkflowStatus.RUNNING.value}
)
_ALLOWED_STATUSES: Final[frozenset[str]] = frozenset(status.value for status in MatchStatus)


class SmokeFailed(AssertionError):  # noqa: N818 - it is an AssertionError, not an Error
    """A step did not hold. Carries the step it broke in."""


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Step:
    index: int
    name: str
    duration_ms: int
    notes: list[str] = field(default_factory=list)


class Trace:
    """A readable, timed step-by-step log. Deliberately the same shape as the
    end-to-end journey's trace, so the two read alike in a terminal."""

    def __init__(self, out: TextIO) -> None:
        self._out = out
        self._started = time.perf_counter()
        self.steps: list[Step] = []

    def banner(self, title: str) -> None:
        self._write("")
        self._write("=" * 78)
        self._write(f"  {title}")
        self._write("=" * 78)

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[Step]:
        index = len(self.steps) + 1
        self._write("")
        self._write(f"[{index:>2}] {name}")
        started = time.perf_counter()
        entry = Step(index=index, name=name, duration_ms=0)
        self.steps.append(entry)
        try:
            yield entry
        except AssertionError as exc:
            entry.duration_ms = int((time.perf_counter() - started) * 1000)
            self._write(f"     FAILED after {entry.duration_ms} ms: {exc}")
            raise SmokeFailed(f"step {index} ({name}): {exc}") from exc
        entry.duration_ms = int((time.perf_counter() - started) * 1000)
        for note in entry.notes:
            self._write(f"     - {note}")
        self._write(f"     ok ({entry.duration_ms} ms)")

    def line(self, message: str) -> None:
        """One unnumbered line, for the header."""
        self._write(message)

    def note(self, entry: Step, message: str) -> None:
        entry.notes.append(message)

    def summary(self, *, ok: bool = True) -> None:
        """The per-step table. ``ok=False`` says so instead of claiming a pass."""
        total = int((time.perf_counter() - self._started) * 1000)
        self._write("")
        self._write("-" * 78)
        for entry in self.steps:
            self._write(f"  {entry.index:>2}. {entry.name:<58} {entry.duration_ms:>6} ms")
        self._write("-" * 78)
        if ok:
            self._write(f"  {len(self.steps)} steps, all asserted, {total} ms total")
        else:
            self._write(
                f"  stopped after {len(self.steps)} step(s) in {total} ms - see the failure above"
            )
        self._write("")

    def _write(self, line: str) -> None:
        print(line, file=self._out, flush=True)


def require(condition: object, message: str) -> None:
    """Assert, with a message the trace can print."""
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SmokeResult:
    """What the smoke run observed, for a caller that wants to re-assert."""

    base_url: str
    user_id: str
    steps: list[Step]
    run_id: uuid.UUID
    match_statuses: dict[str, int]
    plan_item_count: int
    chat_was_grounded: bool
    deleted_counts: dict[str, int]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _body(response: httpx.Response, expected: int, what: str) -> dict[str, Any]:
    """The JSON body of a response, with the status code asserted first.

    A failure quotes the error envelope's ``code`` and ``request_id`` rather
    than the whole body, because those are the two things that let somebody
    find the request in the server's logs.
    """
    if response.status_code != expected:
        raise AssertionError(
            f"{what}: expected {expected}, got {response.status_code} {_hint(response)}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise AssertionError(f"{what}: response was not JSON ({exc})") from exc
    require(isinstance(payload, dict), f"{what}: expected a JSON object")
    assert isinstance(payload, dict)  # narrows for the type checker
    return payload


def _hint(response: httpx.Response) -> str:
    """A short, user-safe description of an unexpected response."""
    request_id = response.headers.get("X-Request-ID", "unknown")
    try:
        envelope = response.json()
    except ValueError:
        return f"(non-JSON body, request_id={request_id})"
    if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
        error = envelope["error"]
        return f"(code={error.get('code')!r}, request_id={error.get('request_id', request_id)!r})"
    return f"(request_id={request_id})"


def _product_copy(node: object, key: str | None = None) -> Iterator[str]:
    """Every string in ``node`` that this product wrote rather than quoted.

    Walking by key rather than flattening the payload is what makes the
    vocabulary assertion meaningful: a run detail embeds the catalog record and
    the official source excerpts behind it, so a blind substring search would
    flag a government page for using the word "eligible" about its own rules.
    """
    if isinstance(node, Mapping):
        for name, value in node.items():
            if str(name) in QUOTED_SUBTREE_KEYS:
                continue
            yield from _product_copy(value, str(name))
    elif isinstance(node, list | tuple):
        for item in node:
            yield from _product_copy(item, key)
    elif isinstance(node, str) and key in PRODUCT_COPY_KEYS:
        yield node


def _no_banned_language(payload: object, where: str) -> None:
    """No copy this product wrote may claim or deny eligibility."""
    for line in _product_copy(payload):
        lowered = line.lower()
        for word in BANNED_WORDS:
            if word in lowered:
                raise AssertionError(
                    f"{where} contains the banned phrase {word!r} in product copy: {line!r}"
                )


def _check_matches(matches: Sequence[Mapping[str, Any]], where: str) -> dict[str, int]:
    """The product rules every match has to satisfy, whatever its status."""
    counts: dict[str, int] = {}
    for match in matches:
        status = str(match["status"])
        require(
            status in _ALLOWED_STATUSES,
            f"{where}: {status!r} is not one of {sorted(_ALLOWED_STATUSES)}",
        )
        evaluation = match["evaluation"]
        require(
            status == evaluation["status"],
            f"{where}: the status a user reads must be the one the engine decided",
        )
        require(evaluation["evidence"], f"{where}: a status without evidence is an opinion")
        if status == MatchStatus.MORE_INFO_NEEDED.value:
            require(
                evaluation["missing_fields"],
                f"{where}: MORE_INFO_NEEDED must name the field it was missing",
            )
        require(
            match["program"]["administering_org"],
            f"{where}: a program must be shown next to the organisation that runs it",
        )
        counts[status] = counts.get(status, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
async def run_smoke(
    base_url: str,
    *,
    token: str | None = None,
    user_id: str | None = None,
    narrative: str = NARRATIVE,
    request_timeout_s: float = 30.0,
    poll_timeout_s: float = 180.0,
    delete_data: bool = True,
    out: TextIO | None = None,
) -> SmokeResult:
    """Drive the core flow against ``base_url`` and assert every response."""
    person = user_id or str(uuid.uuid4())
    trace = Trace(out if out is not None else sys.stdout)
    headers = {"Authorization": f"Bearer {token}"} if token else {DEV_USER_HEADER: person}
    mode = "bearer token" if token else f"{DEV_USER_HEADER} (AUTH_MODE=dev)"

    trace.banner(f"SecondChance smoke test against {base_url}")
    trace.line(f"  synthetic user, authenticated by {mode}")

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), timeout=request_timeout_s, follow_redirects=False
    ) as client:
        try:
            result = await _walk(
                client,
                trace=trace,
                person=person,
                headers=headers,
                narrative=narrative,
                poll_timeout_s=poll_timeout_s,
                delete_data=delete_data,
            )
        except BaseException:
            trace.summary(ok=False)
            raise
        trace.summary()
        return result


async def _walk(
    client: httpx.AsyncClient,
    *,
    trace: Trace,
    person: str,
    headers: dict[str, str],
    narrative: str,
    poll_timeout_s: float,
    delete_data: bool,
) -> SmokeResult:
    # ------------------------------------------------------------- step 1
    with trace.step("The process is alive") as entry:
        health = _body(await client.get("/healthz"), 200, "liveness")
        require(health["status"] == "ok", f"liveness is not ok: {health['status']}")
        trace.note(entry, f"app_env={health['app_env']}")

    # ------------------------------------------------------------- step 2
    with trace.step("The process is ready, and says which dependencies it checked") as entry:
        response = await client.get("/readyz")
        ready = _body(response, 200, "readiness")
        require(ready["ready"] is True, "the server reports itself unready")
        database = next(dep for dep in ready["dependencies"] if dep["name"] == "database")
        require(database["ok"] is True, "the database dependency is not ok")
        trace.note(
            entry,
            "dependencies: "
            + ", ".join(f"{dep['name']}={dep['ok']}" for dep in ready["dependencies"]),
        )

    # ------------------------------------------------------------- step 3
    with trace.step("A request with no credentials is refused with the error envelope") as entry:
        anonymous = await client.get("/api/v1/plans")
        require(
            anonymous.status_code == 401,
            f"an unauthenticated read returned {anonymous.status_code}, not 401",
        )
        envelope = _body(anonymous, 401, "unauthenticated read")
        require("error" in envelope, "a non-2xx body must use the error envelope")
        error = envelope["error"]
        for key in ("code", "message", "request_id"):
            require(error.get(key), f"the error envelope is missing {key}")
        _no_banned_language(envelope, "the 401 envelope")
        trace.note(entry, f"code={error['code']} and a quotable request_id")

    # ------------------------------------------------------------- step 4
    with trace.step("The catalog is served, and a slug resolves to a program") as entry:
        page = _body(await client.get("/api/v1/programs?limit=5"), 200, "list programs")
        require(page["total"] >= 1, "the catalog is empty - run './do seed'")
        require(page["items"], "a non-empty catalog returned no items")
        for item in page["items"]:
            require(item["administering_org"], f"{item['slug']} does not name who runs it")
            require(item["official_url"].startswith("https://"), f"{item['slug']}: no official URL")
        slug = str(page["items"][0]["slug"])
        detail = _body(await client.get(f"/api/v1/programs/{slug}"), 200, "program by slug")
        require(detail["program"]["slug"] == slug, "the slug lookup returned another program")
        require(detail["sources"], "a program must carry the official sources behind it")
        # Not scanned for the banned vocabulary: a catalog record and its
        # sources are quoted from official pages, which are allowed to use the
        # word "eligible" about their own rules. See QUOTED_SUBTREE_KEYS.
        trace.note(entry, f"{page['total']} programs, {slug} has {len(detail['sources'])} sources")

    # ------------------------------------------------------------- step 5
    with trace.step("A narrative becomes a structured profile") as entry:
        intake = _body(
            await client.post("/api/v1/intake", json={"narrative": narrative}, headers=headers),
            200,
            "intake",
        )
        profile = intake["profile"]
        require(profile["state"] == "NJ", "extraction lost the state")
        require(isinstance(intake["unknown_fields"], list), "unknown_fields must be a list")
        undetermined = set(intake["unknown_fields"])
        require(
            undetermined <= PROFILE_FIELD_NAMES,
            # A question can only be answered if it names a field the profile
            # actually has, so an unknown name here is a dead end in the UI.
            f"undetermined names that are not profile fields: "
            f"{sorted(undetermined - PROFILE_FIELD_NAMES)}",
        )
        _no_banned_language(intake, "the intake response")
        determined = sorted(
            name
            for name, value in profile.items()
            if value not in (None, [], {}) and name not in {"version", "updated_at"}
        )
        trace.note(entry, f"determined fields: {', '.join(determined)}")
        trace.note(
            entry,
            f"undetermined={intake['unknown_fields']} "
            f"ambiguities={len(intake['ambiguities'])} "
            f"source={intake['extraction_source']}",
        )

    # ------------------------------------------------------------- step 6
    with trace.step("Starting a check returns 202 and a durable id to poll") as entry:
        key = f"smoke-{person[:8]}"
        created = await client.post(
            "/api/v1/workflows",
            json={"narrative": narrative, "idempotency_key": key},
            headers=headers,
        )
        accepted = _body(created, 202, "create a run")
        run_id = uuid.UUID(accepted["run_id"])
        require(accepted["deduplicated"] is False, "a first create cannot be a replay")
        require(
            created.headers.get("Location") == accepted["poll_url"],
            "the Location header must match poll_url",
        )
        require("X-Request-ID" in created.headers, "every response carries a request id")
        replay = _body(
            await client.post(
                "/api/v1/workflows",
                json={"narrative": narrative, "idempotency_key": key},
                headers=headers,
            ),
            200,
            "replay the same idempotency key",
        )
        require(replay["deduplicated"] is True, "a replayed key must report deduplicated")
        require(replay["run_id"] == str(run_id), "a replayed key must return the original run")
        trace.note(entry, f"run accepted at {accepted['poll_url']}, replay deduplicated")

    # ------------------------------------------------------------- step 7
    with trace.step("The run is polled to completion, answering or skipping questions") as entry:
        detail = await _poll(
            client, headers, run_id, trace=trace, entry=entry, timeout_s=poll_timeout_s
        )
        rounds = 0
        while detail["status"] == WorkflowStatus.AWAITING_INPUT.value:
            require(
                rounds < MAX_CLARIFICATION_ROUNDS,
                f"the run asked for input more than {MAX_CLARIFICATION_ROUNDS} times",
            )
            questions = detail["clarification_questions"]
            require(questions, "AWAITING_INPUT with no question is a contract violation")
            _no_banned_language(questions, "a clarification question")
            for question in questions:
                require(
                    "ssn" not in str(question["field"]).lower(),
                    "the product must never ask for an SSN",
                )
            # Skipping is a real answer: it records the field as declined, which
            # is the honest thing for a script to do - a smoke test has no facts
            # about this fictional person beyond the narrative, and inventing
            # one would change what the rule engine decides.
            answers = [
                {"question_id": question["id"], "field": question["field"], "skipped": True}
                for question in questions
            ]
            trace.note(
                entry,
                f"round {rounds + 1}: skipped {[q['field'] for q in questions]}",
            )
            resumed = _body(
                await client.post(
                    f"/api/v1/workflows/{run_id}/resume",
                    json={"answers": answers},
                    headers=headers,
                ),
                200,
                "resume the run",
            )
            require(
                resumed["status"] != WorkflowStatus.AWAITING_INPUT.value,
                "a resumed run must stop telling the user it is waiting",
            )
            rounds += 1
            detail = await _poll(
                client, headers, run_id, trace=trace, entry=entry, timeout_s=poll_timeout_s
            )
        require(
            detail["status"] == WorkflowStatus.COMPLETED.value,
            f"the run finished as {detail['status']}: {(detail.get('error') or {}).get('code')}",
        )
        require(detail["completed_at"], "a terminal run must carry completed_at")
        _no_banned_language(detail, "the completed run")

    # ------------------------------------------------------------- step 8
    with trace.step("Every match carries an engine-decided status and its evidence") as entry:
        matches = detail["matches"]
        require(matches, "the run completed with no matches at all")
        counts = _check_matches(matches, "a match")
        breakdown = ", ".join(f"{status}={counts[status]}" for status in sorted(counts))
        trace.note(entry, f"{len(matches)} matches: {breakdown}")
        trace.note(entry, f"run-level limitations: {len(detail['limitations'])}")

    # ------------------------------------------------------------- step 9
    with trace.step("The plan lists steps once, with official links") as entry:
        plan = detail["plan"]
        require(plan is not None, "a completed run must carry a plan")
        require(plan["items"], "a plan with no steps is not a plan")
        keys = [item["dedup_key"] for item in plan["items"]]
        require(len(keys) == len(set(keys)), "a shared requirement must be collapsed, not repeated")
        shared = {doc["dedup_key"] for doc in plan["shared_documents"]}
        require(
            len(shared) == len(plan["shared_documents"]),
            "a shared document must appear once",
        )
        _no_banned_language(plan, "the action plan")
        trace.note(
            entry,
            f"plan v{plan['version']}: {len(plan['items'])} steps, "
            f"{len(plan['shared_documents'])} shared documents",
        )

    # ------------------------------------------------------------ step 10
    with trace.step("A step can be ticked off and stays ticked") as entry:
        target = plan["items"][0]
        ticked = _body(
            await client.patch(
                f"/api/v1/action-items/{target['id']}",
                json={"is_complete": True},
                headers=headers,
            ),
            200,
            "tick a step",
        )
        require(ticked["is_complete"] is True, "the step did not come back complete")
        reread = _body(
            await client.get(f"/api/v1/plans/{plan['id']}", headers=headers),
            200,
            "re-read the plan",
        )
        again = next(item for item in reread["items"] if item["id"] == target["id"])
        require(again["is_complete"] is True, "the completion did not persist")
        done = sum(1 for item in reread["items"] if item["is_complete"])
        trace.note(entry, f"re-read the plan: {done} of {len(reread['items'])} steps complete")

    # ------------------------------------------------------------ step 11
    with trace.step("A follow-up question is answered, and labelled by its grounding") as entry:
        chat = _body(
            await client.post(
                "/api/v1/assistant/chat",
                json={"message": QUESTION, "run_id": str(run_id)},
                headers=headers,
            ),
            200,
            "assistant chat",
        )
        require(chat["answer"], "the assistant returned an empty answer")
        declared = {citation["source_id"] for citation in chat["citations"]}
        for fact in chat["facts"]:
            require(
                set(fact["source_ids"]) <= declared,
                "a fact cites a source the user is never shown",
            )
        if chat["is_grounded"]:
            require(chat["citations"], "is_grounded is true with no citations")
            require(chat["ungrounded_notice"] is None, "a grounded answer carries no notice")
        else:
            require(
                chat["ungrounded_notice"],
                "an answer with no citations must say so, not read as official",
            )
        _no_banned_language(chat, "the assistant answer")
        trace.note(
            entry,
            f"grounded={chat['is_grounded']} citations={len(chat['citations'])} "
            f"facts={len(chat['facts'])} confidence={chat['confidence']}",
        )

    # ------------------------------------------------------------ step 12
    with trace.step("The metrics scrape serves counters, and no user content") as entry:
        scrape = await client.get("/internal/metrics")
        require(scrape.status_code == 200, f"the metrics scrape returned {scrape.status_code}")
        text = scrape.text
        require(text.strip(), "the metrics scrape is empty")
        for fragment in ("New Brunswick", "Rutgers", "groceries"):
            require(fragment not in text, f"the scrape leaked narrative content: {fragment!r}")
        trace.note(
            entry,
            f"{sum(1 for line in text.splitlines() if line and not line.startswith('#'))} series",
        )

    # ------------------------------------------------------------ step 13
    deleted_counts: dict[str, int] = {}
    if delete_data:
        with trace.step("The synthetic user's data is deleted, and the deletion verified") as entry:
            receipt = _body(
                await client.delete("/api/v1/account/data", headers=headers), 200, "delete data"
            )
            deleted_counts = {name: int(value) for name, value in receipt["deleted_counts"].items()}
            require(deleted_counts.get("workflow_runs", 0) >= 1, "the run was not deleted")
            require(receipt["retained"], "a deletion receipt must say what it kept and why")
            remaining = _body(
                await client.get("/api/v1/workflows", headers=headers), 200, "list runs after"
            )
            require(remaining["total"] == 0, f"{remaining['total']} runs survived the deletion")
            plans_left = _body(
                await client.get("/api/v1/plans", headers=headers), 200, "list plans after"
            )
            require(plans_left["total"] == 0, f"{plans_left['total']} plans survived the deletion")
            trace.note(
                entry,
                "deleted " + ", ".join(f"{k}={v}" for k, v in sorted(deleted_counts.items()) if v),
            )
            trace.note(entry, f"retained: {len(receipt['retained'])} stated reasons")
    else:
        with trace.step("Deletion skipped (--keep-data), so the data is left in place") as entry:
            trace.note(entry, "nothing was deleted; remove --keep-data to verify deletion")

    return SmokeResult(
        base_url=str(client.base_url),
        user_id=person,
        steps=trace.steps,
        run_id=run_id,
        match_statuses=counts,
        plan_item_count=len(plan["items"]),
        chat_was_grounded=bool(chat["is_grounded"]),
        deleted_counts=deleted_counts,
    )


async def _poll(
    client: httpx.AsyncClient,
    headers: Mapping[str, str],
    run_id: uuid.UUID,
    *,
    trace: Trace,
    entry: Step,
    timeout_s: float,
) -> dict[str, Any]:
    """Poll a run the way the frontend does, until it stops advancing."""
    deadline = time.perf_counter() + timeout_s
    seen: list[str] = []
    detail: dict[str, Any] = {}
    while time.perf_counter() < deadline:
        detail = _body(
            await client.get(f"/api/v1/workflows/{run_id}", headers=dict(headers)),
            200,
            "poll a run",
        )
        phase = f"{detail['status']}/{detail['phase']}"
        if phase not in seen:
            seen.append(phase)
        if detail["status"] not in _RUNNING:
            trace.note(entry, f"phases seen: {' -> '.join(seen)}")
            trace.note(
                entry,
                f'progress: {detail["progress_fraction"]:.2f} "{detail["progress_message"]}"',
            )
            return detail
        await asyncio.sleep(0.25)
    raise AssertionError(f"the run never left {detail.get('status')!r} within {timeout_s}s")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.smoke",
        description="Drive the core flow against a running API and assert it responds.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SMOKE_BASE_URL", DEFAULT_BASE_URL),
        help=f"Root URL of the running API. Default $SMOKE_BASE_URL or {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SMOKE_TOKEN"),
        help=(
            "Access token for an AUTH_MODE=supabase server, sent as a bearer token. "
            "Default $SMOKE_TOKEN. Without it the script uses the dev-user header."
        ),
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="UUID to act as in AUTH_MODE=dev. Default: a fresh random one per run.",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=30.0, help="Per-request timeout in seconds."
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=180.0,
        help="How long a single run may take before the smoke test gives up.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help=(
            "Do not delete the synthetic user's data at the end. Leaves rows behind, "
            "so the deletion step is not verified either."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        asyncio.run(
            run_smoke(
                args.base_url,
                token=args.token,
                user_id=args.user_id,
                request_timeout_s=args.request_timeout,
                poll_timeout_s=args.poll_timeout,
                delete_data=not args.keep_data,
            )
        )
    except SmokeFailed as exc:
        print(f"\nSMOKE FAILED: {exc}\n", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        # A connection error is the common case: the server is not running.
        print(
            f"\nSMOKE FAILED: could not reach {args.base_url} ({type(exc).__name__}).\n"
            "Start the API first: ./do dev-api\n",
            file=sys.stderr,
        )
        return 1
    print(f"SMOKE PASSED - the core flow works against {args.base_url}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
