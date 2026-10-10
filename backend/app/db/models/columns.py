"""Reusable column annotations shared by the ORM models.

``app.db.base`` fixes the global type mapping (``dict[str, Any]`` -> ``JSONB``,
``datetime`` -> ``TIMESTAMPTZ``, ``str`` -> ``TEXT``). What it cannot express is
the handful of shapes that need a concrete Postgres type rather than a Python
one:

* ``text[]`` columns. ``Mapped[list[str]]`` does not match the ``list[Any]``
  entry in ``type_annotation_map``, so every array column must name
  ``ARRAY(Text)`` explicitly. Arrays are used - rather than JSONB - wherever a
  value is a flat list of short tokens that the database itself may need to
  search (``keywords``, ``needs``, ``missing_fields``), because ``&&`` and
  ``@>`` are indexable and ``array_to_string`` feeds the full-text vector.
* JSON *arrays* of objects (evidence trees, shared documents, model usage).
  Those stay JSONB: they are validated by Pydantic on the way in and out and
  have no business being decomposed.
* The user id. It is the Supabase ``sub`` claim, carried as text so that the
  row-level-security policies can compare it to ``auth.uid()::text`` without a
  cast on the column side (a cast there would make the index unusable).

Every default is both client-side and server-side for the same reason
``TimestampMixin`` is: a row written by a migration, by ``psql`` or by the seed
script must come out of the database looking exactly like one written by the
ORM.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Annotated, Any, Final

from sqlalchemy import ARRAY, BigInteger, Date, Integer, Numeric, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import mapped_column

__all__ = [
    "Cents",
    "ConfidenceNumeric",
    "EmptyJsonList",
    "EmptyJsonObject",
    "EmptyTextArray",
    "EnumText",
    "NullableEnumText",
    "OptionalDate",
    "RequiredDate",
    "SmallCount",
    "TextArray",
    "UserId",
]

#: Server-side literals for the "empty" defaults, so DDL and ORM agree.
_EMPTY_ARRAY: Final = text("'{}'::text[]")
_EMPTY_OBJECT: Final = text("'{}'::jsonb")
_EMPTY_LIST: Final = text("'[]'::jsonb")

#: The authenticated user's id (the Supabase ``sub``). Never a display name,
#: never an email. Indexed on every user-owned table and compared directly to
#: ``auth.uid()::text`` by the RLS policies.
UserId = Annotated[str, mapped_column(Text, nullable=False)]

#: A closed-vocabulary value stored as text, e.g. a ``MatchStatus`` member.
#:
#: Deliberately **not** a Postgres ``ENUM`` type and deliberately without a
#: ``CHECK`` constraint. ``app.schemas.enums`` promises that members can be
#: added freely and never renamed; a native enum or a check would turn every
#: new member into a migration, and the Pydantic boundary already rejects a
#: value outside the vocabulary before it can reach a column.
EnumText = Annotated[str, mapped_column(Text, nullable=False)]
NullableEnumText = Annotated[str | None, mapped_column(Text, nullable=True)]

#: A flat ``text[]`` column that is never null - absent means empty.
EmptyTextArray = Annotated[
    list[str],
    mapped_column(ARRAY(Text), nullable=False, default=list, server_default=_EMPTY_ARRAY),
]

#: A nullable-free ``text[]`` with no default, for columns that must be supplied.
TextArray = Annotated[list[str], mapped_column(ARRAY(Text), nullable=False)]

#: A JSONB object that defaults to ``{}``.
EmptyJsonObject = Annotated[
    dict[str, Any],
    mapped_column(JSONB, nullable=False, default=dict, server_default=_EMPTY_OBJECT),
]

#: A JSONB array that defaults to ``[]``.
EmptyJsonList = Annotated[
    list[Any],
    mapped_column(JSONB, nullable=False, default=list, server_default=_EMPTY_LIST),
]

#: Integer cents. Money is never a float; ``BIGINT`` because annual income in
#: cents overflows ``INTEGER`` at roughly $21M and a validation error there
#: would be a strange way to find out.
Cents = Annotated[int | None, mapped_column(BigInteger, nullable=True)]

#: A small bounded count (household size, age, sort order).
SmallCount = Annotated[int | None, mapped_column(Integer, nullable=True)]

#: A ranking score in ``[0, 1]``. ``NUMERIC`` rather than ``double precision``
#: so a stored score round-trips to the exact same value it was computed as,
#: which is what makes a persisted evaluation reproducible.
ConfidenceNumeric = Annotated[
    Decimal, mapped_column(Numeric(5, 4), nullable=False, server_default=text("0"))
]

RequiredDate = Annotated[dt.date, mapped_column(Date, nullable=False)]
OptionalDate = Annotated[dt.date | None, mapped_column(Date, nullable=True)]
