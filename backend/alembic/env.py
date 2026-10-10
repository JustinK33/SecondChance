"""Alembic environment for SecondChance.

Three things are deliberate here.

**The URL is never in a file.** ``alembic.ini`` has no ``sqlalchemy.url``; it is
read from :class:`app.core.config.Settings`, which is the only module allowed to
touch the environment. So ``make migrate`` migrates the app database, and
``DATABASE_URL=... alembic upgrade head`` migrates whichever database that
variable names - which is exactly how the test database is migrated - and no
connection string, and therefore no password, ever lands in a committed file.

**Migrations run synchronously.** The application uses asyncpg, but Alembic's
own machinery is synchronous and gains nothing from being driven through a
greenlet shim, so migrations go over ``postgresql+psycopg`` via
``Settings.database_url_sync``. Both URLs are derived from the one configured
value, so they cannot drift.

**The LangGraph checkpoint tables are not ours.** ``checkpoints``,
``checkpoint_blobs``, ``checkpoint_writes`` and ``checkpoint_migrations`` are
created by ``await AsyncPostgresSaver.setup()`` and their schema belongs to that
library. They live in the same database, so without :func:`include_object`
below, every autogenerate run would helpfully offer to drop them.
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any, Final, Literal

from sqlalchemy import engine_from_config, pool
from sqlalchemy.schema import SchemaItem

from alembic import context
from app.core.config import get_settings
from app.db import models  # noqa: F401  - registers every table on the metadata
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata

#: Tables owned by ``langgraph-checkpoint-postgres``, not by these migrations.
EXTERNAL_TABLE_PREFIXES: Final[tuple[str, ...]] = ("checkpoint",)

#: Schemas Alembic must ignore entirely. ``auth`` holds Supabase's own objects
#: (and, locally, the fallback ``auth.uid()`` the RLS migration installs).
EXTERNAL_SCHEMAS: Final[frozenset[str]] = frozenset({"auth", "extensions", "storage", "realtime"})


def include_object(
    obj: SchemaItem,
    name: str | None,
    type_: Literal[
        "schema", "table", "column", "index", "unique_constraint", "foreign_key_constraint"
    ],
    reflected: bool,
    compare_to: SchemaItem | None,
) -> bool:
    """Keep autogenerate from touching objects these migrations do not own."""
    if type_ == "table" and name is not None:
        if name.startswith(EXTERNAL_TABLE_PREFIXES):
            return False
        schema = getattr(obj, "schema", None)
        if schema in EXTERNAL_SCHEMAS:
            return False
    return True


def include_name(
    name: str | None,
    type_: Literal["schema", "table", "column", "index", "unique_constraint"],
    parent_names: dict[str, str | None],
) -> bool:
    """Restrict reflection to the default schema."""
    if type_ == "schema":
        return name is None or name not in EXTERNAL_SCHEMAS
    return True


def _common_options() -> dict[str, Any]:
    return {
        "target_metadata": target_metadata,
        "include_object": include_object,
        "include_name": include_name,
        # Without these two, a widened column or a changed server default is
        # invisible to autogenerate and the model and the database drift apart
        # silently.
        "compare_type": True,
        "compare_server_default": True,
        # Name every constraint from the shared convention, so a generated
        # downgrade can actually drop what the upgrade created.
        "render_as_batch": False,
        "transaction_per_migration": True,
    }


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of connecting.

    Used to review a migration, and to hand DDL to someone with privileges the
    application role does not have.
    """
    context.configure(
        url=get_settings().database_url_sync,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **_common_options(),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run migrations against the configured database."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = get_settings().database_url_sync
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, **_common_options())
            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
