"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create date: ${create_date}

Why this migration exists
-------------------------
TODO: one paragraph. What changed in the model, and why. A reader three months
from now needs the reason, not a restatement of the DDL below.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = ${repr(up_revision)}
down_revision: str | None = ${repr(down_revision)}
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
