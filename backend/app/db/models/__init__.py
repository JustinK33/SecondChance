"""SQLAlchemy ORM models. Owned by the backend-infrastructure agent.

Importing this package is what registers every table on
``app.db.base.Base.metadata``, so Alembic's ``env.py`` and any test that
creates the schema import it for that side effect alone.

A naming note. Several of these classes share a name with a Pydantic model in
``app.schemas`` - ``AssistanceProgram``, ``ProgramSource``, ``ProgramMatch``,
``ActionPlan``, ``ActionItem``. That is deliberate: an ORM row and the
validated object it serialises to describe the same thing and should be called
the same thing. To keep the two unambiguous, code that touches both imports the
module rather than the names::

    from app.db import models
    from app.schemas import ActionPlan

    row = models.ActionPlan(...)      # a database row
    plan = ActionPlan.model_validate(...)   # the validated API object

Every repository in ``app.db.repositories`` follows that convention.
"""

from __future__ import annotations

from app.db.models.audit import AuditEvent, ModelCall
from app.db.models.match import ProgramMatch
from app.db.models.plan import ActionItem, ActionPlan
from app.db.models.profile import Profile
from app.db.models.program import (
    SEARCH_VECTOR_EXPRESSION,
    TEXT_ARRAY_JOIN_FUNCTION_DDL,
    AssistanceProgram,
    ProgramRuleVersion,
    ProgramSource,
)
from app.db.models.workflow import WorkflowRun

__all__ = [
    "SEARCH_VECTOR_EXPRESSION",
    "TEXT_ARRAY_JOIN_FUNCTION_DDL",
    "USER_OWNED_TABLES",
    "ActionItem",
    "ActionPlan",
    "AssistanceProgram",
    "AuditEvent",
    "ModelCall",
    "Profile",
    "ProgramMatch",
    "ProgramRuleVersion",
    "ProgramSource",
    "WorkflowRun",
]

#: Tables that carry a ``user_id`` and are therefore protected by row-level
#: security and removed by the account-deletion flow. Deletion order matters
#: (children first) and is encoded here so the migration, the repository and
#: the tests all agree on one list.
USER_OWNED_TABLES: tuple[str, ...] = (
    "action_items",
    "action_plans",
    "program_matches",
    "workflow_runs",
    "profiles",
    "audit_events",
)
