"""enable row level security

Revision ID: 61054d279612
Revises: f683fcfdca94
Create date: 2026-10-10 19:30:00.000000+00:00

Why this migration exists
-------------------------
Authorisation that lives only in application code is one forgotten ``WHERE``
clause away from a cross-user data leak. Putting it in the database makes the
isolation a property of the data rather than of the discipline of every future
query, and keeping it in a migration makes it reviewable, version-controlled
and - crucially - testable.

The policy is the same on all six user-owned tables::

    FOR ALL USING (user_id = auth.uid()::text)
            WITH CHECK (user_id = auth.uid()::text)

``USING`` controls which rows are visible to ``SELECT``, ``UPDATE`` and
``DELETE``; ``WITH CHECK`` controls what ``INSERT`` and ``UPDATE`` may write.
Both are needed: ``USING`` alone would let a user insert a row stamped with
somebody else's id, which they could then never see but which would still be
there.

``program_matches`` and ``action_items`` carry a denormalised ``user_id`` rather
than reaching their owner through ``run_id`` / ``plan_id``. A policy with a
subquery to the parent runs that subquery for every candidate row and, more
importantly, is a second place the ownership rule can be got wrong. The column
is written by the repository that creates the row, from the same verified
``user_id`` as the parent.

Catalog tables get RLS with a ``SELECT``-only policy and *no* write policy at
all. With RLS enabled and no permissive policy for a command, that command is
denied - so writes are available only to a role that bypasses RLS, which is
Supabase's ``service_role`` and the table owner. That is "public read,
service-role write" without naming a role that may not exist yet.
``model_calls`` gets RLS and no policies whatsoever: it is operator telemetry.

The ``auth.uid()`` problem
--------------------------
``auth.uid()`` is Supabase's function; on plain Postgres the ``auth`` schema
does not exist and every policy here would fail to parse. Since Supabase is not
provisioned yet and the isolation has to be *tested today* against local Docker
Postgres, this migration installs a compatible fallback when, and only when,
``auth.uid()`` is missing:

* ``auth.uid()`` reads ``current_setting('request.jwt.claims', true)``, parses
  it as JSON and returns the ``sub`` claim as a ``uuid`` - exactly what
  Supabase's own implementation does, and exactly what PostgREST sets.
* It returns ``NULL`` rather than raising when the setting is absent, empty or
  unparseable. ``user_id = NULL`` is ``NULL``, which is not ``true``, so a
  connection with no claims sees nothing. Failing closed is the only acceptable
  direction for this function.

The fallback is tagged with a comment marker and the downgrade removes only a
function carrying that marker, so running this migration against a real
Supabase project never touches Supabase's own ``auth`` schema.

Note that RLS does not apply to a table's owner unless ``FORCE ROW LEVEL
SECURITY`` is set, and it is deliberately not set here. The local
``AUTH_MODE=dev`` path connects as the owner, so these policies are inert in
local development - which is why ``tests/security/`` must also assert cross-user
isolation at the service layer, and why ``tests/integration/test_rls.py``
switches to a non-owner role before asserting anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from alembic import op

revision: str = "61054d279612"
down_revision: str | None = "f683fcfdca94"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Tables whose rows belong to exactly one user.
USER_OWNED_TABLES: Final[tuple[str, ...]] = (
    "profiles",
    "workflow_runs",
    "program_matches",
    "action_plans",
    "action_items",
    "audit_events",
)

#: Public catalog tables: readable by anyone, writable only by a role that
#: bypasses RLS.
CATALOG_TABLES: Final[tuple[str, ...]] = (
    "assistance_programs",
    "program_sources",
    "program_rule_versions",
)

#: Tables with RLS on and no policies at all - reachable only by a bypassing
#: role. Operator telemetry, never user-facing.
SERVICE_ONLY_TABLES: Final[tuple[str, ...]] = ("model_calls",)

#: Marker identifying the local-only fallback, so the downgrade can tell it
#: apart from Supabase's real ``auth.uid()`` and leave the latter alone.
FALLBACK_MARKER: Final[str] = "secondchance-local-auth-uid-fallback"

#: Creates ``auth.uid()`` only when it does not already exist.
#:
#: ``STABLE`` (not ``IMMUTABLE``): the value depends on a session setting, so it
#: is constant within a statement but not across statements - which is exactly
#: what a policy needs and what lets the planner cache it per statement.
CREATE_AUTH_UID_FALLBACK: Final[str] = f"""
DO $do$
BEGIN
    IF to_regprocedure('auth.uid()') IS NOT NULL THEN
        RAISE NOTICE 'auth.uid() already exists; leaving it alone';
        RETURN;
    END IF;

    CREATE SCHEMA IF NOT EXISTS auth;

    CREATE FUNCTION auth.uid() RETURNS uuid
    LANGUAGE plpgsql
    STABLE
    AS $fn$
    DECLARE
        claims text;
    BEGIN
        claims := current_setting('request.jwt.claims', true);
        IF claims IS NULL OR claims = '' THEN
            RETURN NULL;
        END IF;
        RETURN (claims::jsonb ->> 'sub')::uuid;
    EXCEPTION
        WHEN others THEN
            -- Malformed claims must not become an error that a caller could
            -- mistake for a permissive result. Fail closed.
            RETURN NULL;
    END
    $fn$;

    COMMENT ON FUNCTION auth.uid() IS '{FALLBACK_MARKER}';
    EXECUTE 'GRANT USAGE ON SCHEMA auth TO PUBLIC';
    EXECUTE 'GRANT EXECUTE ON FUNCTION auth.uid() TO PUBLIC';
END
$do$;
"""

#: Drops the fallback, and only the fallback.
DROP_AUTH_UID_FALLBACK: Final[str] = f"""
DO $do$
BEGIN
    IF to_regprocedure('auth.uid()') IS NULL THEN
        RETURN;
    END IF;
    IF obj_description('auth.uid()'::regprocedure, 'pg_proc') IS DISTINCT FROM
       '{FALLBACK_MARKER}' THEN
        RAISE NOTICE 'auth.uid() was not installed by this migration; leaving it alone';
        RETURN;
    END IF;
    DROP FUNCTION auth.uid();
    -- Only removes the schema if this migration's function was the last thing
    -- in it, so a real Supabase auth schema is never dropped.
    IF NOT EXISTS (
        SELECT 1 FROM pg_class
        WHERE relnamespace = 'auth'::regnamespace
    ) AND NOT EXISTS (
        SELECT 1 FROM pg_proc
        WHERE pronamespace = 'auth'::regnamespace
    ) THEN
        DROP SCHEMA auth;
    END IF;
END
$do$;
"""  # noqa: S608 - the only interpolation is this module's own FALLBACK_MARKER

#: Grants the Supabase roles need, applied only to roles that actually exist.
#: Without these a Supabase deployment would be blocked by table privileges
#: before RLS was ever consulted; locally there is nothing to do.
GRANT_SUPABASE_ROLES: Final[str] = """
DO $do$
DECLARE
    user_tables text := '{user_tables}';
    catalog_tables text := '{catalog_tables}';
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO authenticated';
        EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON ' || user_tables || ' TO authenticated';
        EXECUTE 'GRANT SELECT ON ' || catalog_tables || ' TO authenticated';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO anon';
        EXECUTE 'GRANT SELECT ON ' || catalog_tables || ' TO anon';
    END IF;
END
$do$;
"""

REVOKE_SUPABASE_ROLES: Final[str] = """
DO $do$
DECLARE
    all_tables text := '{all_tables}';
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        EXECUTE 'REVOKE ALL ON ' || all_tables || ' FROM authenticated';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        EXECUTE 'REVOKE ALL ON ' || all_tables || ' FROM anon';
    END IF;
END
$do$;
"""


def _policy_name(table: str, suffix: str) -> str:
    return f"{table}_{suffix}"


def upgrade() -> None:
    op.execute(CREATE_AUTH_UID_FALLBACK)

    for table in USER_OWNED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {_policy_name(table, 'owner_all')} ON {table} "
            f"FOR ALL "
            f"USING (user_id = auth.uid()::text) "
            f"WITH CHECK (user_id = auth.uid()::text)"
        )

    for table in CATALOG_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # SELECT only. The absence of an INSERT/UPDATE/DELETE policy is what
        # restricts writes to a bypassing role; it is not an omission.
        op.execute(
            f"CREATE POLICY {_policy_name(table, 'public_read')} ON {table} FOR SELECT USING (true)"
        )

    for table in SERVICE_ONLY_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")

    op.execute(
        GRANT_SUPABASE_ROLES.format(
            user_tables=", ".join(USER_OWNED_TABLES),
            catalog_tables=", ".join(CATALOG_TABLES),
        )
    )


def downgrade() -> None:
    op.execute(
        REVOKE_SUPABASE_ROLES.format(
            all_tables=", ".join(USER_OWNED_TABLES + CATALOG_TABLES + SERVICE_ONLY_TABLES)
        )
    )

    for table in SERVICE_ONLY_TABLES:
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    for table in CATALOG_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {_policy_name(table, 'public_read')} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    for table in USER_OWNED_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {_policy_name(table, 'owner_all')} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    op.execute(DROP_AUTH_UID_FALLBACK)
