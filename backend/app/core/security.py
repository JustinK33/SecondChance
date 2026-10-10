"""Authentication: Supabase JWT verification and the local development path.

Two modes, chosen by ``AUTH_MODE`` and nothing else.

``supabase``
    A bearer token signed by the project's Supabase Auth instance. Asymmetric
    tokens (ES256 / RS256) are verified against the project's JWKS, fetched
    over HTTPS and cached; symmetric tokens (HS256) are verified against
    ``SUPABASE_JWT_SECRET``, which Supabase is retiring but which still issues
    today. Expiry, issuer and audience are all verified explicitly, because
    PyJWT only checks ``aud`` and ``iss`` when you pass them - omitting them is
    a silent downgrade that accepts a token minted for another project.

``dev``
    An ``X-Dev-User: <uuid>`` header naming the principal, with no signature at
    all. This exists because Supabase is not provisioned yet and the product
    has to be fully exercisable today. It is gated twice: ``Settings`` refuses
    to construct with ``AUTH_MODE=dev`` unless ``APP_ENV=local``, and
    :func:`assert_auth_mode_allowed` re-checks it during startup so a
    misconfigured deployment crashes on boot rather than serving every user's
    data to anyone who can set a header.

The identity always comes from the token, never from the request. A ``user_id``
in a body or a query string is ignored - not rejected, *ignored* - so an
endpoint cannot accidentally become an authorisation bypass by accepting a
field it was supposed to overwrite.

Nothing here logs a token, a claim set or a raw user id. A log line carries
``hash_user_id`` output and the algorithm name, which is enough to debug a key
rotation and not enough to impersonate anyone.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import httpx
import jwt
from jwt import PyJWK

from app.core.config import Settings
from app.core.errors import InternalError, UnauthorizedError, UpstreamUnavailableError
from app.core.logging import get_logger, hash_user_id

__all__ = [
    "ALLOWED_ASYMMETRIC_ALGORITHMS",
    "AUDIENCE",
    "DEV_USER_HEADER",
    "AuthenticatedUser",
    "JwksCache",
    "assert_auth_mode_allowed",
    "dev_user_from_header",
    "get_jwks_cache",
    "issuer_for",
    "jwks_url_for",
    "reset_jwks_cache",
    "verify_token",
]

_log = get_logger(__name__)

#: Supabase mints every end-user token with this audience.
AUDIENCE: Final[str] = "authenticated"
#: Supabase signs with ES256 by default and RS256 for projects migrated from
#: the legacy symmetric key. Nothing else is accepted: an attacker-chosen
#: algorithm list is the classic JWT confusion attack.
ALLOWED_ASYMMETRIC_ALGORITHMS: Final[tuple[str, ...]] = ("ES256", "RS256")
_LEGACY_ALGORITHM: Final[str] = "HS256"

#: Header naming the principal in ``AUTH_MODE=dev``.
DEV_USER_HEADER: Final[str] = "X-Dev-User"

#: Claims that must be present. ``sub`` is the user id, so a token without one
#: identifies nobody and cannot be authorised.
_REQUIRED_CLAIMS: Final[list[str]] = ["exp", "iss", "aud", "sub"]

#: JWKS cache lifetime. Long enough that a request burst costs one fetch,
#: short enough that a rotated key is picked up without a restart.
JWKS_TTL_S: Final[float] = 600.0
#: Floor between forced refreshes, so an unknown ``kid`` - which an attacker
#: can produce at will - cannot be turned into a fetch amplifier.
JWKS_MIN_REFRESH_INTERVAL_S: Final[float] = 30.0
JWKS_TIMEOUT_S: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """The verified principal for one request.

    ``user_id`` is the token's ``sub`` claim, which is the only identity the
    rest of the application is allowed to use.
    """

    user_id: str
    source: Literal["dev", "supabase"]

    @property
    def log_identity(self) -> str:
        """The hashed id to put in a log line. Never the raw id."""
        return hash_user_id(self.user_id)


def assert_auth_mode_allowed(settings: Settings) -> None:
    """Fail closed if the configured auth mode is not safe for this environment.

    ``Settings`` already enforces this, but the check is cheap and this is the
    one invariant where defence in depth is clearly worth it: the failure mode
    is "every user's data is readable by anyone".
    """
    if settings.auth_mode == "dev" and not settings.is_local:
        msg = (
            "AUTH_MODE=dev accepts an unsigned X-Dev-User header and is only "
            f"permitted when APP_ENV=local (APP_ENV={settings.app_env!r})"
        )
        raise InternalError(msg)
    if settings.auth_mode == "supabase" and not settings.supabase_url:
        msg = (
            "AUTH_MODE=supabase requires SUPABASE_URL so the token issuer and "
            "JWKS endpoint can be derived and verified"
        )
        raise InternalError(msg)


def issuer_for(settings: Settings) -> str:
    """The ``iss`` claim Supabase puts in this project's tokens."""
    return f"{settings.supabase_url.rstrip('/')}/auth/v1"


def jwks_url_for(settings: Settings) -> str:
    """The project's JWKS endpoint."""
    return f"{issuer_for(settings)}/.well-known/jwks.json"


class JwksCache:
    """Fetches and caches a JWKS, with a TTL and a single-flight guard.

    The guard is the interesting part. Without it, a burst of requests arriving
    with a cold cache each start their own HTTPS fetch; with a rotated key that
    burst is every request in flight, and the auth provider sees a stampede
    exactly when it is least able to absorb one. One lock, plus a re-check
    after acquiring it, turns N fetches into one.
    """

    __slots__ = ("_fetched_at", "_keys", "_last_attempt_at", "_lock", "_ttl_s", "_url")

    def __init__(self, url: str, *, ttl_s: float = JWKS_TTL_S) -> None:
        self._url = url
        self._ttl_s = ttl_s
        self._keys: dict[str, PyJWK] = {}
        self._fetched_at = 0.0
        self._last_attempt_at = 0.0
        self._lock = asyncio.Lock()

    @property
    def url(self) -> str:
        return self._url

    @property
    def is_fresh(self) -> bool:
        return bool(self._keys) and (time.monotonic() - self._fetched_at) < self._ttl_s

    async def key_for(self, kid: str | None) -> PyJWK:
        """The signing key for ``kid``, fetching or refreshing as needed.

        An unknown ``kid`` triggers at most one forced refresh per
        :data:`JWKS_MIN_REFRESH_INTERVAL_S`, which covers a genuine key
        rotation without letting a crafted token become a denial-of-service
        amplifier against the auth provider.
        """
        if not self.is_fresh:
            await self._refresh()
        if kid is None:
            if len(self._keys) == 1:
                return next(iter(self._keys.values()))
            raise UnauthorizedError("We could not verify your session. Please sign in again.")
        key = self._keys.get(kid)
        if key is not None:
            return key
        if (time.monotonic() - self._last_attempt_at) >= JWKS_MIN_REFRESH_INTERVAL_S:
            await self._refresh(force=True)
            key = self._keys.get(kid)
        if key is None:
            _log.warning("auth.jwks.unknown_kid", known_keys=len(self._keys))
            raise UnauthorizedError("We could not verify your session. Please sign in again.")
        return key

    async def _refresh(self, *, force: bool = False) -> None:
        async with self._lock:
            # Re-check inside the lock: whoever held it may already have done
            # the work this caller was about to duplicate.
            if not force and self.is_fresh:
                return
            if force and (time.monotonic() - self._last_attempt_at) < JWKS_MIN_REFRESH_INTERVAL_S:
                return
            self._last_attempt_at = time.monotonic()
            document = await self._fetch()
            keys: dict[str, PyJWK] = {}
            for entry in document.get("keys", []):
                if not isinstance(entry, dict):
                    continue
                kid = entry.get("kid")
                if not isinstance(kid, str):
                    continue
                try:
                    keys[kid] = PyJWK(entry)
                except Exception:
                    _log.warning("auth.jwks.key_rejected")
            if not keys:
                _log.error("auth.jwks.empty")
                raise UpstreamUnavailableError(
                    "We cannot verify sign-ins right now. Please try again shortly."
                )
            self._keys = keys
            self._fetched_at = time.monotonic()
            _log.info("auth.jwks.refreshed", key_count=len(keys), forced=force)

    async def _fetch(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=JWKS_TIMEOUT_S) as client:
                response = await client.get(self._url)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # The URL is configuration, not user input, but the exception text
            # can quote a response body, so only the type is recorded.
            _log.error("auth.jwks.fetch_failed", error_type=type(exc).__name__)
            raise UpstreamUnavailableError(
                "We cannot verify sign-ins right now. Please try again shortly.", cause=exc
            ) from exc
        if not isinstance(payload, dict):
            raise UpstreamUnavailableError(
                "We cannot verify sign-ins right now. Please try again shortly."
            )
        return payload


_jwks_cache: JwksCache | None = None


def get_jwks_cache(settings: Settings) -> JwksCache:
    """The process-wide JWKS cache for the configured project.

    One per process so the TTL and the single-flight guard are actually shared.
    Rebuilt if the configured URL changes, which only happens in tests.
    """
    global _jwks_cache
    url = jwks_url_for(settings)
    if _jwks_cache is None or _jwks_cache.url != url:
        _jwks_cache = JwksCache(url)
    return _jwks_cache


def reset_jwks_cache() -> None:
    """Drop the cached keys. For tests and for a configuration reload."""
    global _jwks_cache
    _jwks_cache = None


def _user_id_from(claims: dict[str, Any]) -> str:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise UnauthorizedError("We could not verify your session. Please sign in again.")
    return subject.strip()


def _decode_options() -> dict[str, Any]:
    return {
        "require": list(_REQUIRED_CLAIMS),
        "verify_exp": True,
        "verify_aud": True,
        "verify_iss": True,
        "verify_signature": True,
    }


async def verify_token(token: str, settings: Settings) -> AuthenticatedUser:
    """Verify a Supabase bearer token and return the principal it names.

    Raises :class:`~app.core.errors.UnauthorizedError` for anything wrong with
    the token, and :class:`~app.core.errors.UpstreamUnavailableError` when the
    JWKS endpoint cannot be reached - the distinction matters, because the
    first is the user's problem to fix by signing in again and the second is
    ours.
    """
    cleaned = token.strip()
    if not cleaned:
        raise UnauthorizedError()

    try:
        header = jwt.get_unverified_header(cleaned)
    except jwt.PyJWTError as exc:
        _log.info("auth.token.malformed", error_type=type(exc).__name__)
        raise UnauthorizedError(cause=exc) from exc

    algorithm = header.get("alg")
    issuer = issuer_for(settings) if settings.supabase_url else None
    if issuer is None:
        # Cannot verify ``iss`` without knowing the project. Refusing is the
        # only safe answer: accepting would mean trusting a token from any
        # Supabase project at all.
        _log.error("auth.config.missing_supabase_url")
        raise UpstreamUnavailableError(
            "Sign-in is not fully configured yet. Please try again later."
        )

    key: PyJWK | str
    if algorithm == _LEGACY_ALGORITHM:
        secret = settings.supabase_jwt_secret.get_secret_value()
        if not secret:
            _log.warning("auth.token.hs256_without_secret")
            raise UnauthorizedError()
        key = secret
        algorithms = [_LEGACY_ALGORITHM]
    elif algorithm in ALLOWED_ASYMMETRIC_ALGORITHMS:
        key = await get_jwks_cache(settings).key_for(header.get("kid"))
        algorithms = [str(algorithm)]
    else:
        _log.warning("auth.token.unsupported_algorithm", algorithm=str(algorithm))
        raise UnauthorizedError()

    try:
        claims = jwt.decode(
            cleaned,
            key=key,
            algorithms=algorithms,
            audience=AUDIENCE,
            issuer=issuer,
            # PyJWT types ``options`` as a closed TypedDict; the keys here are
            # all real members and the cast keeps mypy strict without widening
            # the call site.
            options=cast(Any, _decode_options()),
        )
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError(
            "Your session has expired. Please sign in again.", cause=exc
        ) from exc
    except jwt.PyJWTError as exc:
        _log.info("auth.token.rejected", error_type=type(exc).__name__, algorithm=str(algorithm))
        raise UnauthorizedError(cause=exc) from exc

    user = AuthenticatedUser(user_id=_user_id_from(claims), source="supabase")
    _log.debug("auth.token.accepted", user=user.log_identity, algorithm=str(algorithm))
    return user


def dev_user_from_header(value: str | None, settings: Settings) -> AuthenticatedUser:
    """The principal named by ``X-Dev-User``, in local development only.

    The value must be a UUID. Requiring one is not decoration: every
    user-scoped table stores ``user_id`` as the token subject, and Supabase
    subjects are UUIDs, so accepting ``"alice"`` here would let development
    data take a shape production can never produce.
    """
    assert_auth_mode_allowed(settings)
    if settings.auth_mode != "dev":  # pragma: no cover - guarded by the caller
        raise UnauthorizedError()
    if not value or not value.strip():
        raise UnauthorizedError(
            f"Development authentication requires a {DEV_USER_HEADER} header.",
            detail={"header": DEV_USER_HEADER},
        )
    try:
        parsed = uuid.UUID(value.strip())
    except ValueError as exc:
        raise UnauthorizedError(
            f"{DEV_USER_HEADER} must be a UUID.",
            detail={"header": DEV_USER_HEADER},
            cause=exc,
        ) from exc
    return AuthenticatedUser(user_id=str(parsed), source="dev")
