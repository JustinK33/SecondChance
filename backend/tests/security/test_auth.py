"""Token verification, exercised against real signatures.

The question these tests answer is not "does the happy path work" but "which
bad tokens get in". So every case here is a token that is *almost* right -
signed with the wrong key, minted for another project, expired by a second,
carrying an algorithm the project does not use - and the assertion is a 401 in
the standard envelope.

No live Supabase project exists yet, so the JWKS is served by an in-process
``httpx.MockTransport`` and the tokens are signed with a locally generated
ES256 keypair. That is the whole of the substitution: ``verify_token``,
``JwksCache``, PyJWT and the real ``jwt.decode`` options all run unmodified,
which is the only way a test like this is worth anything.

The second half is about ``AUTH_MODE=dev``, the unsigned ``X-Dev-User`` path
that exists because Supabase is not provisioned. It is gated twice -
``Settings`` refuses to construct it outside ``APP_ENV=local``, and
``assert_auth_mode_allowed`` re-checks at request time - and both gates are
asserted here, because the failure mode is every user's data readable by
anyone who can set a header.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security as security_module
from app.core.config import Settings
from app.core.security import AUDIENCE, DEV_USER_HEADER, issuer_for, reset_jwks_cache
from app.db import models
from tests.security.harness import (
    ApiHarness,
    api_harness,
    dev_settings,
    jwks_shim,
    rolled_back_api,
    supabase_settings,
)

# ``rolled_back_api`` is a fixture, imported for the two tests below that need
# a dev-mode application. Re-exported so pytest can find it in this namespace.
__all__ = ["rolled_back_api"]

pytestmark = pytest.mark.db

NARRATIVE = "I need help with groceries and my electric bill this month, things are tight."


# ---------------------------------------------------------------------------
# local key material
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Signer:
    """An ES256 keypair and the JWK a JWKS document would publish for it."""

    kid: str
    private_key: Any
    jwk: dict[str, Any]

    def sign(self, claims: dict[str, Any], *, kid: str | None = None, alg: str = "ES256") -> str:
        return jwt.encode(claims, self.private_key, algorithm=alg, headers={"kid": kid or self.kid})


def make_signer(kid: str) -> Signer:
    """Generate an ES256 keypair. Ephemeral, per test, never written down."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk: dict[str, Any] = dict(ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True))
    jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return Signer(kid=kid, private_key=private_key, jwk=jwk)


def claims_for(
    settings: Settings,
    subject: str,
    *,
    audience: str | None = None,
    issuer: str | None = None,
    expires_in_s: int = 3600,
) -> dict[str, Any]:
    """A claim set shaped the way Supabase Auth mints one."""
    now = datetime.now(UTC)
    return {
        "sub": subject,
        "aud": audience if audience is not None else AUDIENCE,
        "iss": issuer if issuer is not None else issuer_for(settings),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in_s)).timestamp()),
        "role": "authenticated",
    }


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def signer() -> Signer:
    return make_signer("primary-test-key")


@pytest.fixture
def jwks_calls(signer: Signer, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Serve a JWKS in process, and count how often it is fetched."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"keys": [signer.jwk]})

    monkeypatch.setattr(security_module, "httpx", jwks_shim(handler))
    reset_jwks_cache()
    try:
        yield calls
    finally:
        reset_jwks_cache()


@pytest.fixture
def supabase_config() -> Settings:
    return supabase_settings()


@pytest.fixture
async def supabase_api(
    supabase_config: Settings, session: AsyncSession
) -> Any:  # AsyncIterator[ApiHarness]
    async with api_harness(settings=supabase_config, session=session) as harness:
        yield harness


def envelope(response: httpx.Response) -> dict[str, Any]:
    """The error envelope, asserted to have the documented shape."""
    body = response.json()
    assert set(body) == {"error"}, f"non-standard error body: {body}"
    error = body["error"]
    assert set(error) >= {"code", "message", "request_id"}, error
    assert error["message"], "an error message must be renderable verbatim"
    return error


# ---------------------------------------------------------------------------
# the tokens that must not get in
# ---------------------------------------------------------------------------
class TestRejectedTokens:
    """Every case is a 401 with the standard envelope and no detail leaked."""

    async def test_a_missing_authorization_header_is_rejected(
        self, supabase_api: ApiHarness
    ) -> None:
        response = await supabase_api.client.get("/api/v1/workflows")
        assert response.status_code == 401
        error = envelope(response)
        assert error["code"] == "unauthorized"

    async def test_a_bearer_header_with_no_token_is_rejected(
        self, supabase_api: ApiHarness
    ) -> None:
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers={"Authorization": "Bearer "}
        )
        assert response.status_code == 401
        assert envelope(response)["code"] == "unauthorized"

    async def test_a_malformed_token_is_rejected(self, supabase_api: ApiHarness) -> None:
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer("not.a.jwt")
        )
        assert response.status_code == 401
        assert envelope(response)["code"] == "unauthorized"

    async def test_a_token_signed_with_the_wrong_key_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        """Right ``kid``, right claims, wrong private key. The signature is all."""
        impostor = make_signer("ignored")
        token = impostor.sign(claims_for(supabase_api.settings, str(uuid.uuid4())), kid=signer.kid)
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401
        assert envelope(response)["code"] == "unauthorized"
        assert jwks_calls, "the JWKS was never consulted, so nothing was verified"

    async def test_a_token_naming_an_unknown_key_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        token = signer.sign(
            claims_for(supabase_api.settings, str(uuid.uuid4())), kid="rotated-away"
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401

    async def test_an_expired_token_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        token = signer.sign(claims_for(supabase_api.settings, str(uuid.uuid4()), expires_in_s=-1))
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401
        error = envelope(response)
        assert error["code"] == "unauthorized"
        assert "sign in" in error["message"].lower(), "an expired session must say what to do"

    async def test_a_token_for_the_wrong_audience_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        token = signer.sign(
            claims_for(supabase_api.settings, str(uuid.uuid4()), audience="service_role")
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401

    async def test_a_token_from_another_project_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        """Same key, same audience, different ``iss``. This is the quiet one.

        PyJWT only checks ``iss`` when it is passed, so omitting it would
        accept a token minted by any Supabase project at all.
        """
        token = signer.sign(
            claims_for(
                supabase_api.settings,
                str(uuid.uuid4()),
                issuer="https://someone-elses-project.supabase.co/auth/v1",
            )
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401

    async def test_a_token_without_a_subject_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        claims = claims_for(supabase_api.settings, "ignored")
        del claims["sub"]
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(signer.sign(claims))
        )
        assert response.status_code == 401

    async def test_an_unsupported_algorithm_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        """Algorithm confusion: the header does not get to choose the family."""
        token = jwt.encode(
            claims_for(supabase_api.settings, str(uuid.uuid4())),
            "an-attacker-chosen-key-long-enough-to-avoid-a-length-warning",
            algorithm="HS384",
            headers={"kid": signer.kid},
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401

    async def test_an_unsigned_token_is_rejected(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        token = jwt.encode(
            claims_for(supabase_api.settings, str(uuid.uuid4())), key=None, algorithm="none"
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401

    async def test_a_legacy_hs256_token_with_the_wrong_secret_is_rejected(
        self, supabase_api: ApiHarness, jwks_calls: list[str]
    ) -> None:
        token = jwt.encode(
            claims_for(supabase_api.settings, str(uuid.uuid4())),
            "not-the-configured-secret-but-long-enough-to-avoid-a-warning",
            algorithm="HS256",
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 401


class TestAcceptedTokens:
    """The counterpart, so the rejections above are not vacuous."""

    async def test_a_valid_token_authenticates_its_subject(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str], session
    ) -> None:
        subject = str(uuid.uuid4())
        token = signer.sign(claims_for(supabase_api.settings, subject))

        response = await supabase_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers=supabase_api.bearer(token),
        )
        assert response.status_code == 202, response.text

        owner = await session.scalar(
            select(models.WorkflowRun.user_id).where(
                models.WorkflowRun.id == uuid.UUID(response.json()["run_id"])
            )
        )
        assert owner == subject, "the row must belong to the token's subject"

    async def test_the_jwks_is_fetched_once_for_a_burst(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        token = signer.sign(claims_for(supabase_api.settings, str(uuid.uuid4())))
        for _ in range(4):
            response = await supabase_api.client.get(
                "/api/v1/workflows", headers=supabase_api.bearer(token)
            )
            assert response.status_code == 200
        assert len(jwks_calls) == 1, f"the JWKS cache is not holding: {jwks_calls}"

    async def test_an_unknown_kid_cannot_amplify_fetches(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str]
    ) -> None:
        """A crafted ``kid`` is free to produce; a JWKS fetch is not."""
        for index in range(5):
            token = signer.sign(
                claims_for(supabase_api.settings, str(uuid.uuid4())), kid=f"made-up-{index}"
            )
            response = await supabase_api.client.get(
                "/api/v1/workflows", headers=supabase_api.bearer(token)
            )
            assert response.status_code == 401
        assert len(jwks_calls) <= 2, f"an unknown kid became a fetch amplifier: {jwks_calls}"

    async def test_a_legacy_hs256_token_with_the_configured_secret_is_accepted(
        self, supabase_api: ApiHarness, jwks_calls: list[str]
    ) -> None:
        """Supabase is retiring HS256 but still issues it, so it still verifies."""
        subject = str(uuid.uuid4())
        token = jwt.encode(
            claims_for(supabase_api.settings, subject),
            supabase_api.settings.supabase_jwt_secret.get_secret_value(),
            algorithm="HS256",
        )
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers=supabase_api.bearer(token)
        )
        assert response.status_code == 200
        assert jwks_calls == [], "a symmetric token must not touch the JWKS endpoint"


class TestTheDevelopmentHeader:
    """``X-Dev-User`` is unsigned, so the gates around it are the product."""

    async def test_it_is_ignored_entirely_in_supabase_mode(self, supabase_api: ApiHarness) -> None:
        response = await supabase_api.client.get(
            "/api/v1/workflows", headers={DEV_USER_HEADER: str(uuid.uuid4())}
        )
        assert response.status_code == 401
        assert envelope(response)["code"] == "unauthorized"

    async def test_it_cannot_override_the_token_it_arrives_with(
        self, supabase_api: ApiHarness, signer: Signer, jwks_calls: list[str], session
    ) -> None:
        subject = str(uuid.uuid4())
        other = str(uuid.uuid4())
        token = signer.sign(claims_for(supabase_api.settings, subject))
        response = await supabase_api.client.post(
            "/api/v1/workflows",
            json={"narrative": NARRATIVE},
            headers={**supabase_api.bearer(token), DEV_USER_HEADER: other},
        )
        assert response.status_code == 202
        owner = await session.scalar(
            select(models.WorkflowRun.user_id).where(
                models.WorkflowRun.id == uuid.UUID(response.json()["run_id"])
            )
        )
        assert owner == subject
        assert owner != other

    def test_settings_refuses_to_construct_dev_mode_outside_local(self) -> None:
        with pytest.raises(ValidationError, match="APP_ENV"):
            dev_settings(app_env="staging")

    async def test_a_misconfigured_process_fails_closed_rather_than_open(
        self, rolled_back_api: ApiHarness
    ) -> None:
        """The second gate, for a process that got past the first.

        ``Settings`` already refuses this combination, so the only way to
        reach the runtime check is to mutate the object - which is exactly the
        shape of the accident being guarded against: a deployment that still
        carries a developer configuration. It must refuse the request, not
        authenticate it.
        """
        rolled_back_api.settings.app_env = "staging"
        response = await rolled_back_api.client.get(
            "/api/v1/workflows", headers=rolled_back_api.dev(str(uuid.uuid4()))
        )
        assert response.status_code == 500
        assert envelope(response)["code"] == "internal_error"
        assert rolled_back_api.runner.scheduled == []

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "app.core.security.assert_auth_mode_allowed raises InternalError with an "
            "operator diagnostic as the message, and the envelope renders message "
            "verbatim, so APP_ENV and the configured mode reach the client. Handoff "
            "note: the diagnostic belongs in the log line and detail=None, with the "
            "class default message going to the user."
        ),
    )
    async def test_the_misconfiguration_detail_does_not_reach_the_client(
        self, rolled_back_api: ApiHarness
    ) -> None:
        rolled_back_api.settings.app_env = "staging"
        response = await rolled_back_api.client.get(
            "/api/v1/workflows", headers=rolled_back_api.dev(str(uuid.uuid4()))
        )
        assert "APP_ENV" not in response.text

    async def test_a_missing_dev_header_is_rejected_in_dev_mode(
        self, rolled_back_api: ApiHarness
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/workflows")
        assert response.status_code == 401
        assert envelope(response)["code"] == "unauthorized"

    async def test_a_dev_header_that_is_not_a_uuid_is_rejected(
        self, rolled_back_api: ApiHarness
    ) -> None:
        """Supabase subjects are UUIDs, so development data must be shaped alike."""
        response = await rolled_back_api.client.get(
            "/api/v1/workflows", headers=rolled_back_api.dev("alice")
        )
        assert response.status_code == 401
        assert envelope(response)["code"] == "unauthorized"


class TestPublicRoutes:
    """The two routes that are unauthenticated on purpose, and no others."""

    async def test_the_catalog_needs_no_token(self, supabase_api: ApiHarness) -> None:
        assert (await supabase_api.client.get("/api/v1/programs")).status_code == 200

    async def test_the_probes_need_no_token(self, supabase_api: ApiHarness) -> None:
        assert (await supabase_api.client.get("/healthz")).status_code == 200
        assert (await supabase_api.client.get("/readyz")).status_code == 200

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/api/v1/intake", {"narrative": NARRATIVE}),
            ("POST", "/api/v1/workflows", {"narrative": NARRATIVE}),
            ("GET", "/api/v1/workflows", None),
            ("GET", f"/api/v1/workflows/{uuid.uuid4()}", None),
            ("POST", f"/api/v1/workflows/{uuid.uuid4()}/resume", {"answers": []}),
            ("POST", f"/api/v1/workflows/{uuid.uuid4()}/cancel", None),
            ("POST", "/api/v1/matches", {"run_id": str(uuid.uuid4()), "changes": {"age": 30}}),
            ("GET", "/api/v1/plans", None),
            ("GET", f"/api/v1/plans/{uuid.uuid4()}", None),
            ("PATCH", f"/api/v1/action-items/{uuid.uuid4()}", {"is_complete": True}),
            ("POST", "/api/v1/assistant/chat", {"message": "Who can help me with food?"}),
            ("DELETE", "/api/v1/account/data", None),
        ],
    )
    async def test_every_other_route_requires_a_token(
        self, supabase_api: ApiHarness, method: str, path: str, body: dict[str, Any] | None
    ) -> None:
        response = await supabase_api.client.request(method, path, json=body)
        assert response.status_code == 401, f"{method} {path} answered {response.status_code}"
        assert envelope(response)["code"] == "unauthorized"


class TestNothingIsCreatedByARejectedRequest:
    """A 401 must be inert: no row, no side effect."""

    async def test_a_rejected_create_writes_nothing(
        self, supabase_api: ApiHarness, session: AsyncSession
    ) -> None:
        before = await session.scalar(select(func.count()).select_from(models.WorkflowRun))
        for headers in ({}, {"Authorization": "Bearer not.a.jwt"}):
            response = await supabase_api.client.post(
                "/api/v1/workflows", json={"narrative": NARRATIVE}, headers=headers
            )
            assert response.status_code == 401
        after = await session.scalar(select(func.count()).select_from(models.WorkflowRun))
        assert after == before
        assert supabase_api.runner.scheduled == [], "a rejected request scheduled work"
