"""Application settings.

This is the **only** module in the codebase permitted to read the process
environment. Everything else takes a :class:`Settings` instance (or calls
:func:`get_settings`) so that configuration is explicit, typed and testable.

Loading rules
-------------
* Values come from the process environment first, then ``backend/.env``.
* Field names are case-insensitive, so ``APP_ENV`` populates ``app_env``.
* Every incoming string is stripped. The repository-root ``.env`` is CRLF, and
  an unstripped ``"gemini-2.5-flash-lite\\r"`` makes httpx raise
  ``InvalidURL: Invalid non-printable ASCII character in URL``.
* Unknown variables are ignored so one shared ``.env`` can serve several
  processes.

Secrets are typed :class:`~pydantic.SecretStr` so that an accidental ``repr``
or log call prints ``**********`` instead of the value.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "AppEnv",
    "AuthMode",
    "CheckpointerKind",
    "LogLevel",
    "Settings",
    "get_settings",
]

AppEnv = Literal["local", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
AuthMode = Literal["dev", "supabase"]
CheckpointerKind = Literal["postgres", "memory"]

#: ``backend/`` - this file is ``backend/app/core/config.py``.
BACKEND_DIR: Final[Path] = Path(__file__).resolve().parents[2]

_ASYNC_DRIVER: Final[str] = "postgresql+asyncpg"
_SYNC_DRIVER: Final[str] = "postgresql+psycopg"
_LIBPQ_SCHEME: Final[str] = "postgresql"


def _rewrite_scheme(dsn: str, scheme: str) -> str:
    """Return ``dsn`` with its scheme replaced by ``scheme``.

    Postgres is reached through three different URL dialects in this codebase:

    * ``postgresql+asyncpg://`` for the SQLAlchemy async engine,
    * ``postgresql+psycopg://`` for Alembic's synchronous engine,
    * ``postgresql://`` (plain libpq) for ``AsyncPostgresSaver``, which rejects
      SQLAlchemy driver suffixes.

    Only one value is configured and the other two are derived here, so the
    three can never drift apart.
    """
    _, _, remainder = dsn.partition("://")
    if not remainder:
        msg = f"not a valid postgres URL: {dsn!r}"
        raise ValueError(msg)
    return f"{scheme}://{remainder}"


class Settings(BaseSettings):
    """Typed view of every variable documented in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
        validate_default=True,
    )

    # ---------------------------------------------------------------- app
    app_env: AppEnv = "local"
    log_level: LogLevel = "INFO"
    #: ``dev`` trusts a stub user and is rejected outside ``app_env=local``.
    auth_mode: AuthMode = "supabase"
    #: Free-text log values longer than this are truncated by the redactor.
    log_max_text_chars: Annotated[int, Field(ge=16, le=4096)] = 200

    # ----------------------------------------------------------- database
    database_url: str = "postgresql://secondchance:secondchance@localhost:5433/secondchance"
    test_database_url: str = (
        "postgresql://secondchance:secondchance@localhost:5433/secondchance_test"
    )
    db_pool_size: Annotated[int, Field(ge=1, le=100)] = 10
    db_max_overflow: Annotated[int, Field(ge=0, le=100)] = 5
    db_pool_timeout_s: Annotated[float, Field(gt=0, le=120)] = 10.0
    db_statement_timeout_s: Annotated[float, Field(gt=0, le=300)] = 15.0

    # ---------------------------------------------------------- langgraph
    checkpointer: CheckpointerKind = "postgres"

    # ------------------------------------------------------------- gemini
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_model_extraction: str | None = None
    gemini_model_explanation: str | None = None
    gemini_model_chat: str | None = None
    gemini_timeout_s: Annotated[float, Field(gt=0, le=300)] = 30.0
    gemini_max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    gemini_price_input_per_mtok: Annotated[float, Field(ge=0)] = 0.10
    gemini_price_output_per_mtok: Annotated[float, Field(ge=0)] = 0.40

    # ----------------------------------------------------------- workflow
    workflow_timeout_s: Annotated[float, Field(gt=0, le=3600)] = 120.0
    max_concurrent_workflows: Annotated[int, Field(ge=1, le=256)] = 8

    # --------------------------------------------- supabase (server side)
    supabase_url: str = ""
    supabase_service_role_key: SecretStr = SecretStr("")
    supabase_jwt_secret: SecretStr = SecretStr("")

    # --------------------------------------------------------------- urls
    frontend_url: str = "http://localhost:3000"
    next_public_api_url: str = "http://localhost:8000"

    # ----------------------------------------- supabase (browser, public)
    next_public_supabase_url: str = ""
    next_public_supabase_anon_key: str = ""

    # ---------------------------------------------------------------------
    # normalisation
    # ---------------------------------------------------------------------
    @field_validator("*", mode="before")
    @classmethod
    def _strip_strings(cls, value: Any) -> Any:
        """Strip whitespace (including a trailing ``\\r``) from every raw string.

        ``str_strip_whitespace`` only applies to fields annotated as ``str``, so
        it misses ``SecretStr`` and the numeric fields. Doing it here covers all
        of them before coercion, which is what makes CRLF ``.env`` files safe.
        """
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("database_url", "test_database_url")
    @classmethod
    def _validate_postgres_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if not parts.scheme.startswith("postgres"):
            msg = f"expected a postgres URL, got scheme {parts.scheme!r}"
            raise ValueError(msg)
        if not parts.hostname:
            msg = "postgres URL is missing a host"
            raise ValueError(msg)
        if not parts.path.lstrip("/"):
            msg = "postgres URL is missing a database name"
            raise ValueError(msg)
        return value

    @field_validator("frontend_url", "next_public_api_url")
    @classmethod
    def _validate_http_base(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            msg = f"expected an absolute http(s) URL, got {value!r}"
            raise ValueError(msg)
        return value.rstrip("/")

    # ---------------------------------------------------------------------
    # cross-field rules
    # ---------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_auth_mode(self) -> Self:
        """``AUTH_MODE=dev`` is only ever allowed on a developer machine.

        ``dev`` mode accepts an unauthenticated stub user, so letting it reach a
        deployed environment would expose every user's data. The check fails
        closed: the default ``auth_mode`` is ``supabase``, and a staging or
        production process that still carries a developer ``.env`` refuses to
        start rather than silently disabling authentication.
        """
        if self.auth_mode == "dev" and self.app_env != "local":
            msg = (
                "AUTH_MODE=dev is only permitted when APP_ENV=local "
                f"(APP_ENV={self.app_env!r}); set AUTH_MODE=supabase"
            )
            raise ValueError(msg)
        if self.auth_mode == "supabase" and not self.supabase_jwt_secret.get_secret_value():
            msg = "AUTH_MODE=supabase requires SUPABASE_JWT_SECRET to be set"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_production_hardening(self) -> Self:
        """Refuse to run a deployed environment in a degraded configuration."""
        if self.app_env == "local":
            return self
        problems: list[str] = []
        if self.checkpointer != "postgres":
            problems.append("CHECKPOINTER must be 'postgres' outside APP_ENV=local")
        if not self.gemini_api_key.get_secret_value():
            problems.append("GEMINI_API_KEY must be set outside APP_ENV=local")
        if not self.supabase_url:
            problems.append("SUPABASE_URL must be set outside APP_ENV=local")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @model_validator(mode="after")
    def _check_gemini_budget(self) -> Self:
        """A single model call must be able to finish inside the run budget."""
        worst_case = self.gemini_timeout_s * (self.gemini_max_retries + 1)
        if worst_case > self.workflow_timeout_s:
            msg = (
                "GEMINI_TIMEOUT_S * (GEMINI_MAX_RETRIES + 1) = "
                f"{worst_case:g}s exceeds WORKFLOW_TIMEOUT_S={self.workflow_timeout_s:g}s; "
                "one retrying model call could consume the whole workflow budget"
            )
            raise ValueError(msg)
        return self

    # ---------------------------------------------------------------------
    # derived values
    # ---------------------------------------------------------------------
    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def debug(self) -> bool:
        """Whether to expose developer affordances such as ``/docs``."""
        return self.app_env == "local"

    @property
    def database_url_async(self) -> str:
        """SQLAlchemy async engine URL (``postgresql+asyncpg://``)."""
        return _rewrite_scheme(self.database_url, _ASYNC_DRIVER)

    @property
    def database_url_sync(self) -> str:
        """Synchronous URL for Alembic (``postgresql+psycopg://``)."""
        return _rewrite_scheme(self.database_url, _SYNC_DRIVER)

    @property
    def database_dsn(self) -> str:
        """Plain libpq DSN, required by ``AsyncPostgresSaver``."""
        return _rewrite_scheme(self.database_url, _LIBPQ_SCHEME)

    @property
    def test_database_url_async(self) -> str:
        return _rewrite_scheme(self.test_database_url, _ASYNC_DRIVER)

    @property
    def test_database_url_sync(self) -> str:
        return _rewrite_scheme(self.test_database_url, _SYNC_DRIVER)

    @property
    def test_database_dsn(self) -> str:
        return _rewrite_scheme(self.test_database_url, _LIBPQ_SCHEME)

    @property
    def extraction_model(self) -> str:
        """Model for narrative -> structured profile extraction."""
        return self.gemini_model_extraction or self.gemini_model

    @property
    def explanation_model(self) -> str:
        """Model for plain-language wording of an already-decided status."""
        return self.gemini_model_explanation or self.gemini_model

    @property
    def chat_model(self) -> str:
        """Model for source-grounded follow-up conversation turns."""
        return self.gemini_model_chat or self.gemini_model

    @property
    def gemini_timeout_ms(self) -> int:
        """``types.HttpOptions(timeout=...)`` is in milliseconds, not seconds."""
        return math.ceil(self.gemini_timeout_s * 1000)

    @property
    def gemini_enabled(self) -> bool:
        """Whether a live model call is even possible.

        When this is ``False`` the AI layer must use its deterministic fallback
        implementation; the product stays fully usable without model access.
        """
        return bool(self.gemini_api_key.get_secret_value())

    @property
    def cors_origins(self) -> tuple[str, ...]:
        """Exact allowed browser origins. Never ``"*"`` - requests are credentialed."""
        origins = [self.frontend_url]
        if self.is_local:
            origins.append("http://127.0.0.1:3000")
        return tuple(dict.fromkeys(origins))

    def estimate_cost_usd(self, prompt_tokens: int, output_tokens: int) -> float:
        """Estimated USD cost of one model call at the configured prices.

        Thinking tokens are billed as output by Gemini, so callers should fold
        ``thoughts_token_count`` into ``output_tokens``.
        """
        per_mtok = 1_000_000.0
        return (
            prompt_tokens * self.gemini_price_input_per_mtok
            + output_tokens * self.gemini_price_output_per_mtok
        ) / per_mtok


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, loaded once.

    Tests that need a different configuration should construct ``Settings(...)``
    directly, or call ``get_settings.cache_clear()`` after patching the
    environment. Never mutate the cached instance.
    """
    return Settings()
