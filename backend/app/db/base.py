"""SQLAlchemy 2.x declarative base, naming conventions and shared mixins.

Two things are fixed here and must not be overridden per model.

**Constraint naming.** Postgres generates names like ``programs_slug_key`` for
anonymous constraints, and Alembic's autogenerate cannot reliably drop or alter
something it cannot name. The :data:`NAMING_CONVENTION` below gives every
index, unique constraint, check, foreign key and primary key a deterministic
name derived from its table and columns, which is what makes a generated
downgrade actually work.

**Type mapping.** :attr:`Base.type_annotation_map` binds the Python types used
across the schema layer to the right Postgres types once, so ``Mapped[dict]``
is always ``JSONB`` and ``Mapped[datetime]`` is always ``TIMESTAMPTZ``. A naive
``TIMESTAMP`` column anywhere in this product would corrupt source freshness
and "last verified" the first time the server crossed a DST boundary, so the
mapping is timezone-aware everywhere, matching
``app.schemas.common.UtcDatetime``.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Annotated, Any, Final

from sqlalchemy import DateTime, MetaData, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "JsonDict",
    "SoftDeleteMixin",
    "TimestampMixin",
    "UuidPk",
    "metadata",
]

#: Deterministic names for every constraint and index.
#:
#: ``column_0_N_name`` expands to all participating columns, so composite
#: constraints get stable names too. Check constraints use the explicit
#: ``constraint_name`` given at the call site, which means every
#: ``CheckConstraint`` in the models must pass ``name=...``.
NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

#: UUID primary key, defaulted both client-side (``uuid.uuid4``, applied by the
#: ORM at flush) and server-side (``gen_random_uuid()``, built in since PG13) so
#: a row inserted by a migration, by ``psql`` or by the seed script gets an id
#: the same way an ORM insert does. Program rows pass an explicit
#: ``uuid5(PROGRAM_NAMESPACE, slug)`` instead, so catalog ids are reproducible
#: across seeds; see ``app.schemas.common.program_uuid``.
UuidPk = Annotated[
    uuid.UUID,
    mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    ),
]

#: A JSONB object column. Used for the rule DSL, evidence trees and profile
#: snapshots, which are validated by Pydantic on the way in and out and have no
#: business being decomposed into relational columns.
JsonDict = Annotated[dict[str, Any], mapped_column(JSONB)]


class Base(DeclarativeBase):
    """Declarative base for every ORM model in the application."""

    metadata = metadata

    type_annotation_map = {
        uuid.UUID: UUID(as_uuid=True),
        dt.datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
        list[Any]: JSONB,
        str: Text,
        Decimal: Numeric(12, 4),
    }

    def __repr__(self) -> str:
        """Identity-only repr.

        Deliberately prints just the table name and primary key. The default
        SQLAlchemy repr is harmless, but a hand-written one that interpolated
        column values would put narratives and income into every debugger
        session, exception message and log line that touched an ORM object.
        """
        mapper = type(self).__mapper__
        key_values = ", ".join(
            f"{column.key}={getattr(self, str(column.key), None)!r}"
            for column in mapper.primary_key
        )
        return f"<{type(self).__name__} {key_values}>"


class TimestampMixin:
    """``created_at`` / ``updated_at``, maintained by the database.

    Both defaults are server-side (``now()``), so a row inserted by a
    migration, by ``psql`` or by the seed script is timestamped identically to
    one inserted by the ORM. ``onupdate`` is also set client-side so an ORM
    update refreshes ``updated_at`` without a ``RETURNING`` round trip.
    """

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SoftDeleteMixin:
    """``deleted_at``, for tables where a hard delete would break an audit trail.

    The account data-deletion flow hard-deletes user-owned rows - a privacy
    promise has to be real. This mixin is for catalog and audit tables, where a
    retired program must stop appearing in results while remaining resolvable
    from an action plan a user saved last month.

    Queries must filter ``deleted_at IS NULL`` explicitly; there is no global
    filter, because an invisible one is the kind of magic that eventually hides
    a row someone needed.
    """

    deleted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


#: Short indexed identifier, e.g. a slug or an enum value stored as text.
ShortIdentifier = Annotated[str, mapped_column(String(120))]
