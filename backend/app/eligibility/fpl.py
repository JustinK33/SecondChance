"""Federal poverty guideline tables, as versioned data with their source.

Why this module exists as data rather than as a formula
------------------------------------------------------
The HHS poverty guidelines are published once a year and a rule set pins the
year it was written against (``IncomeFplCondition.fpl_year``). An evaluation
made in March must still be explainable in December, after the guidelines have
been updated, so the table is append-only versioned data and nothing here reads
the clock.

What is in the table
--------------------
For each year: the 48-contiguous-states-and-DC guideline for a household of
one, and the amount HHS instructs to add for each additional person, plus the
URL the figures were read from. New Jersey uses the 48-state figures - the
separate Alaska and Hawaii schedules are deliberately absent, because this
product only serves New Jersey and a wrong-state table is worse than a missing
one.

Household sizes above eight are **not** an extrapolation. The published tables
list one through eight and then instruct the reader to "add $N for each
additional person", so computing size nine from the increment is the official
method, not a guess. Sizes outside ``1..MAX_HOUSEHOLD_SIZE`` and years not in
the table return ``None``, which the engine must turn into ``UNKNOWN`` with the
missing field recorded.

Rounding
--------
Thresholds round **up** to the whole cent. A threshold that is a cent too
generous can only move a borderline person from ``UNLIKELY_MATCH`` to
``POTENTIAL_MATCH``; rounding the other way could tell someone not to apply for
help they can get, which is the one failure this product must not have.

Every function here is pure: no I/O, no clock, no configuration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Final

from app.schemas.enums import IncomePeriod
from app.schemas.profile import MAX_HOUSEHOLD_SIZE

__all__ = [
    "FPL_YEARS",
    "MAX_HOUSEHOLD_SIZE",
    "MONTHS_PER_YEAR",
    "FplTable",
    "FplYear",
    "available_years",
    "default_fpl_table",
    "fpl_annual_cents",
    "fpl_threshold_cents",
]

MONTHS_PER_YEAR: Final[int] = 12
_CENTS_PER_DOLLAR: Final[int] = 100
_PERCENT: Final[int] = 100

#: The last household size the published tables enumerate explicitly. Larger
#: households are computed with the published per-person increment.
PUBLISHED_TABLE_ROWS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class FplYear:
    """One year of the 48-state poverty guidelines, with its provenance.

    ``base_annual_cents`` is the guideline for a household of one and
    ``increment_annual_cents`` is the published "add this for each additional
    person" figure. Both are integer cents: money is never a float here.
    """

    year: int
    base_annual_cents: int
    increment_annual_cents: int
    source_url: str
    source_title: str
    retrieved_on: date

    def annual_cents(self, household_size: int) -> int:
        """Annual guideline for ``household_size``. Caller validates the size."""
        return self.base_annual_cents + (household_size - 1) * self.increment_annual_cents


def _dollars(amount: int) -> int:
    return amount * _CENTS_PER_DOLLAR


def _exact_pct(pct: float) -> Fraction:
    """``pct`` as the exact decimal the rule author wrote.

    ``Fraction(133.33)`` is the binary double, which is a hair above 133.33 and
    pushes a rounded-up threshold a cent further than intended. Going through
    the shortest-repr string takes the decimal the author meant, so a rule set
    and the table agree to the cent.
    """
    return Fraction(str(pct))


_ASPE_CURRENT: Final[str] = (
    "https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines"
)
_ASPE_PRIOR: Final[str] = (
    "https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines/"
    "prior-hhs-poverty-guidelines-federal-register-references"
)

#: Verified HHS poverty guidelines for the 48 contiguous states and DC.
#:
#: Each entry was read from the URL it records on ``retrieved_on`` and
#: cross-checked against the published four-person figure:
#:
#: * 2026 - $15,960 + $5,680/person; published 4-person value $33,000.
#: * 2025 - $15,650 + $5,500/person; published 4-person value $32,150.
#: * 2024 - $15,060 + $5,380/person; published 4-person value $31,200.
#: * 2023 - $14,580 + $5,140/person; published 4-person value $30,000.
#:
#: Only years confirmed against aspe.hhs.gov appear. Do not add a year from
#: memory: an unverified guideline silently changes real people's results.
FPL_YEARS: Final[Mapping[int, FplYear]] = {
    2023: FplYear(
        year=2023,
        base_annual_cents=_dollars(14_580),
        increment_annual_cents=_dollars(5_140),
        source_url=_ASPE_PRIOR,
        source_title="Prior HHS Poverty Guidelines and Federal Register References",
        retrieved_on=date(2026, 10, 10),
    ),
    2024: FplYear(
        year=2024,
        base_annual_cents=_dollars(15_060),
        increment_annual_cents=_dollars(5_380),
        source_url=_ASPE_PRIOR,
        source_title="Prior HHS Poverty Guidelines and Federal Register References",
        retrieved_on=date(2026, 10, 10),
    ),
    2025: FplYear(
        year=2025,
        base_annual_cents=_dollars(15_650),
        increment_annual_cents=_dollars(5_500),
        source_url=_ASPE_PRIOR,
        source_title="Prior HHS Poverty Guidelines and Federal Register References",
        retrieved_on=date(2026, 10, 10),
    ),
    2026: FplYear(
        year=2026,
        base_annual_cents=_dollars(15_960),
        increment_annual_cents=_dollars(5_680),
        source_url=_ASPE_CURRENT,
        source_title="HHS Poverty Guidelines for 2026",
        retrieved_on=date(2026, 10, 10),
    ),
}


def available_years() -> tuple[int, ...]:
    """Years present in the table, ascending."""
    return tuple(sorted(FPL_YEARS))


def _valid_household_size(household_size: int) -> bool:
    return 1 <= household_size <= MAX_HOUSEHOLD_SIZE


def fpl_annual_cents(year: int, household_size: int) -> int | None:
    """Annual poverty guideline in cents, or ``None`` if it cannot be looked up.

    ``None`` means "we do not know", never "zero" and never a guess: an unknown
    year or an out-of-range household size has no defensible answer, so the
    engine reports ``UNKNOWN`` instead of inventing a threshold.
    """
    entry = FPL_YEARS.get(year)
    if entry is None or not _valid_household_size(household_size):
        return None
    return entry.annual_cents(household_size)


def fpl_threshold_cents(
    year: int,
    household_size: int,
    pct: float,
    period: IncomePeriod = IncomePeriod.ANNUAL,
) -> int | None:
    """``pct`` percent of the poverty guideline, in cents for ``period``.

    ``pct`` is a percentage, so 130.0 means 130%. The result is rounded up to
    the whole cent (see the module docstring), and the monthly figure is
    derived from the annual one by dividing by twelve and rounding up, which is
    how the programs that publish monthly limits do it.
    """
    annual = fpl_annual_cents(year, household_size)
    if annual is None:
        return None
    exact = _exact_pct(pct) * annual / _PERCENT
    if period is IncomePeriod.MONTHLY:
        exact = exact / MONTHS_PER_YEAR
    return math.ceil(exact)


class FplTable:
    """The engine's view of the guideline tables.

    A class rather than bare functions because the rule engine takes its table
    by construction (``RuleEngine(fpl=...)``): a test can then pin a specific
    year's figures, and a future need for a state-specific schedule has an
    obvious place to live. The default instance wraps :data:`FPL_YEARS`.
    """

    __slots__ = ("_years",)

    def __init__(self, years: Mapping[int, FplYear] | None = None) -> None:
        self._years: Mapping[int, FplYear] = dict(FPL_YEARS if years is None else years)

    def has_year(self, year: int) -> bool:
        return year in self._years

    def available_years(self) -> tuple[int, ...]:
        return tuple(sorted(self._years))

    def annual_cents(self, *, household_size: int, year: int) -> int | None:
        """The 100%-of-guideline annual figure, or ``None`` if unavailable."""
        entry = self._years.get(year)
        if entry is None or not _valid_household_size(household_size):
            return None
        return entry.annual_cents(household_size)

    def annual_threshold_cents(
        self, *, household_size: int, pct: float, year: int
    ) -> int | None:
        """``pct``% of the annual guideline, rounded up, or ``None``."""
        annual = self.annual_cents(household_size=household_size, year=year)
        if annual is None:
            return None
        return math.ceil(_exact_pct(pct) * annual / _PERCENT)

    def monthly_threshold_cents(
        self, *, household_size: int, pct: float, year: int
    ) -> int | None:
        """``pct``% of the guideline as a monthly figure, rounded up, or ``None``."""
        annual = self.annual_cents(household_size=household_size, year=year)
        if annual is None:
            return None
        return math.ceil(_exact_pct(pct) * annual / _PERCENT / MONTHS_PER_YEAR)

    def threshold_cents(
        self, *, household_size: int, pct: float, year: int, period: IncomePeriod
    ) -> int | None:
        """Period-selecting wrapper, for callers that carry the period in data."""
        if period is IncomePeriod.MONTHLY:
            return self.monthly_threshold_cents(
                household_size=household_size, pct=pct, year=year
            )
        return self.annual_threshold_cents(household_size=household_size, pct=pct, year=year)

    def source_url(self, year: int) -> str | None:
        """Where this year's figures were read from, for citation."""
        entry = self._years.get(year)
        return None if entry is None else entry.source_url

    def source_title(self, year: int) -> str | None:
        entry = self._years.get(year)
        return None if entry is None else entry.source_title


def default_fpl_table() -> FplTable:
    """The table every production code path should use."""
    return FplTable()
