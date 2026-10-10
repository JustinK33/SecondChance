"""Thin async repositories over the ORM models.

One rule governs this whole package: **a user-scoped query always filters by
``user_id``, and that ``user_id`` always comes from the verified token.** An id
in a request body or a URL path is an input, not an authorisation. See
``app.db.repositories.base`` for why that filter is the first line of defence
and row-level security the second.

Two further conventions:

* **Repositories never commit.** They flush when they need an id back, and the
  caller owns the transaction boundary - only the caller knows whether a write
  is the whole unit of work or one step of several.
* **A repository is bound to one session, and a session is never shared across
  concurrent tasks.** Open a ``session_scope()`` per parallel branch and build
  a repository inside it; see ``app.db.session``.

What each one returns is chosen rather than uniform. Plain reads and writes
hand back ORM rows, which keeps them thin. Anything that needs aggregation or
a join to be correct - a run summary, a full ``ProgramMatch``, a page of plans -
returns validated Pydantic objects instead, so that the projection lives in one
place and two screens cannot disagree about it.
"""

from __future__ import annotations

from app.db.repositories.account import CHECKPOINT_TABLES, AccountDataRepository
from app.db.repositories.audit import AuditAction, AuditRepository
from app.db.repositories.base import Repository, UserScopedRepository
from app.db.repositories.matches import MatchRepository
from app.db.repositories.plans import PlanRepository
from app.db.repositories.profiles import ProfileRepository
from app.db.repositories.programs import SqlProgramRepository, to_program_summary
from app.db.repositories.runs import WorkflowRunRepository, hash_narrative

__all__ = [
    "CHECKPOINT_TABLES",
    "AccountDataRepository",
    "AuditAction",
    "AuditRepository",
    "MatchRepository",
    "PlanRepository",
    "ProfileRepository",
    "Repository",
    "SqlProgramRepository",
    "UserScopedRepository",
    "WorkflowRunRepository",
    "hash_narrative",
    "to_program_summary",
]
