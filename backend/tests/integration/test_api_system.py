"""``/api/v1/ready`` - and the rule that a probe may not claim health it cannot see.

This file exists because of a real outage shape. ``_check_model_access`` used
to answer ``{"name": "gemini", "ok": true, "note": "configured"}`` whenever
``GEMINI_API_KEY`` held a non-empty string. With this project's key every
``generateContent`` call fails - the 2.5 generation is retired for new projects
and the project is not authorised for 3.x - so readiness advertised a
dependency that was failing 100% of calls. A probe that reports ``ok=true`` for
a dead dependency does not merely fail to help; it hides the outage it exists
to surface.

The fix is not a live call from the probe. ``generateContent`` on every
load-balancer poll would be billable and would time out exactly when the model
is slow. :class:`~app.services.ai_port.DegradingGeminiPort` already observes
every live failure and latches into a cool-off, so the evidence is sitting in
the process and costs nothing to read. The tests below pin both halves:

* a configured key alone is reported as *unprobed*, never as reachable;
* once a live call has actually failed, readiness says so - while model access
  stays a non-required dependency, so the instance is not drained for it.

Everything model-facing here is driven through ``FakeGeminiClient``. No test in
this file reaches the network, and none of them needs a real API key: the
"configured" state is produced with the visibly fake string below.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.fake import FailureMode, FakeGeminiClient
from app.api.v1.deps import provide_gemini
from app.schemas import DependencyStatus, ReadinessResponse
from app.services.ai_port import DegradingGeminiPort, reset_ai_port
from tests.security.harness import ApiHarness, api_harness, dev_settings

pytestmark = pytest.mark.db

#: Not a credential. ``Settings.gemini_enabled`` is ``bool(key)`` and nothing
#: in these tests calls Google, so the only requirement is that it is
#: non-empty and unmistakably not a key.
PLACEHOLDER_KEY = "not-a-real-key-these-tests-make-no-network-call"

NARRATIVE = "I lost my job in Newark last month and cannot cover rent for my two kids."


async def _gemini_status(harness: ApiHarness, path: str = "/api/v1/ready") -> DependencyStatus:
    """The ``gemini`` entry of a readiness response, parsed through the schema."""
    response = await harness.client.get(path)
    assert response.status_code == 200, response.text
    ready = ReadinessResponse.model_validate(response.json())
    assert ready.ready is True, "model access must never be a required dependency"
    return next(entry for entry in ready.dependencies if entry.name == "gemini")


@pytest.fixture
async def degrading_api(
    session: AsyncSession,
) -> AsyncIterator[tuple[ApiHarness, DegradingGeminiPort]]:
    """A harness whose injected port is the real degrading port.

    The live half is a ``FakeGeminiClient`` in ``unavailable`` mode, which
    raises the same :class:`~app.ai.errors.ModelError` subclass the live client
    raises on an upstream 503. That is what makes this an honest stand-in for
    the 403 the project actually gets: the port cannot tell the difference, and
    neither can readiness.
    """
    settings = dev_settings(gemini_api_key=PLACEHOLDER_KEY)
    port = DegradingGeminiPort(
        live=FakeGeminiClient(settings, failure_mode=FailureMode.UNAVAILABLE),
        deterministic=FakeGeminiClient(settings),
    )
    async with api_harness(settings=settings, session=session) as harness:
        harness.app.dependency_overrides[provide_gemini] = lambda: port
        yield harness, port


class TestModelAccessReadiness:
    async def test_a_configured_key_alone_is_reported_as_unprobed(
        self, degrading_api: tuple[ApiHarness, DegradingGeminiPort]
    ) -> None:
        """A configured key is a fact about a string, not about reachability."""
        harness, port = degrading_api
        assert port.has_live_client and not port.is_degraded

        gemini = await _gemini_status(harness)

        assert gemini.ok is True, "no failure has been observed yet"
        assert gemini.degraded is False
        assert gemini.failures == 0
        assert gemini.note is not None
        # The note has to say the probe did not happen. "configured" on its own
        # is what readers took for "reachable", which is the whole bug.
        assert gemini.note != "configured"
        assert "no live call attempted" in gemini.note, gemini.note

    async def test_readiness_reports_the_degraded_state_after_a_live_failure(
        self, degrading_api: tuple[ApiHarness, DegradingGeminiPort]
    ) -> None:
        """The regression this file was written for.

        One real failed call is enough: the port latches into its cool-off, and
        the probe has to stop advertising the model from that point on.
        """
        harness, port = degrading_api
        before = await _gemini_status(harness)
        assert before.ok is True

        # A genuine call through the port, exactly as a workflow node makes it.
        # The live half raises, the deterministic half answers, and the result
        # is relabelled as non-model output.
        result = await port.extract_profile(NARRATIVE)
        assert result.is_fallback, "a degraded answer must never look live"
        assert port.is_degraded and port.failures == 1

        after = await _gemini_status(harness)
        assert after.ok is False, "the process has evidence the model is unusable"
        assert after.degraded is True
        assert after.failures == 1
        assert after.note

    async def test_a_degraded_model_does_not_drain_the_instance(
        self, degrading_api: tuple[ApiHarness, DegradingGeminiPort]
    ) -> None:
        """Non-required means non-required: still 200, still ``ready: true``.

        A load balancer must not pull an instance out of rotation because the
        wording went deterministic. Every status, every program and every
        official link is still produced.
        """
        harness, port = degrading_api
        await port.extract_profile(NARRATIVE)
        assert port.is_degraded

        response = await harness.client.get("/api/v1/ready")
        assert response.status_code == 200
        body = ReadinessResponse.model_validate(response.json())
        assert body.ready is True
        database = next(entry for entry in body.dependencies if entry.name == "database")
        assert database.ok is True
        gemini = next(entry for entry in body.dependencies if entry.name == "gemini")
        assert gemini.ok is False

    async def test_an_unconfigured_key_is_reported_as_not_provisioned(
        self, session: AsyncSession
    ) -> None:
        """The developer-machine case: no key, no model, product still ready.

        Checked on the root alias too. ``/readyz`` is the path the deployment
        platform probes, and it reaches :func:`app.api.v1.routes.system.ready`
        by a direct call rather than through dependency injection, so it is a
        genuinely separate code path through the same probe.
        """
        try:
            async with api_harness(settings=dev_settings(), session=session) as harness:
                for path in ("/api/v1/ready", "/readyz"):
                    gemini = await _gemini_status(harness, path)
                    assert gemini.ok is False, path
                    assert gemini.degraded is True, path
                    assert gemini.note and "not provisioned" in gemini.note, path
        finally:
            # ``/readyz`` resolves the process-wide port, which memoises. Leave
            # no port built from this test's settings behind for the next one.
            reset_ai_port()

    async def test_no_readiness_note_carries_an_upstream_error(
        self, degrading_api: tuple[ApiHarness, DegradingGeminiPort]
    ) -> None:
        """An unauthenticated endpoint is not a place to publish upstream detail.

        The real failures are ``404 NOT_FOUND`` and ``403 PERMISSION_DENIED``
        naming specific model ids. None of that may reach the response: it
        tells an attacker about the deployment and tells an operator nothing
        that ``degraded`` does not already say.
        """
        harness, port = degrading_api
        await port.extract_profile(NARRATIVE)
        body = (await harness.client.get("/api/v1/ready")).text.lower()

        leaks = ("gemini-", "403", "404", "permission_denied", "not_found", "api_key", "key=")
        for leaked in leaks:
            assert leaked not in body, f"readiness leaked {leaked!r}"
