"""Tests for the federal poverty guideline tables.

Every expected number here was taken from the published HHS table, not from the
code, so a transcription error in ``app.eligibility.fpl`` fails these tests
rather than being confirmed by them:

* 2026 - https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines
* 2023-2025 - the "Prior HHS Poverty Guidelines and Federal Register
  References" page linked from it.

The guidelines decide who this product tells to apply for food and health
coverage, so "the arithmetic is obviously right" is not good enough: the table
rows, the per-person increment, the percentage arithmetic, the monthly division
and the refusal to extrapolate are each checked separately.
"""

from __future__ import annotations

import pytest

from app.eligibility.fpl import (
    FPL_YEARS,
    MONTHS_PER_YEAR,
    FplTable,
    FplYear,
    available_years,
    default_fpl_table,
    fpl_annual_cents,
    fpl_threshold_cents,
)
from app.schemas import IncomePeriod
from app.schemas.profile import MAX_HOUSEHOLD_SIZE

#: The published 2026 table for the 48 contiguous states and DC, in dollars.
PUBLISHED_2026 = {
    1: 15_960,
    2: 21_640,
    3: 27_320,
    4: 33_000,
    5: 38_680,
    6: 44_360,
    7: 50_040,
    8: 55_720,
}

#: The published four-person figure for every year in the table. Checking this
#: pins the base and the increment together: a wrong increment that happens to
#: give the right one-person figure still fails here.
PUBLISHED_FOUR_PERSON = {2023: 30_000, 2024: 31_200, 2025: 32_150, 2026: 33_000}

#: Published one-person figure and per-additional-person increment, in dollars.
PUBLISHED_BASE_AND_INCREMENT = {
    2023: (14_580, 5_140),
    2024: (15_060, 5_380),
    2025: (15_650, 5_500),
    2026: (15_960, 5_680),
}


def dollars(amount: int) -> int:
    return amount * 100


class TestPublishedTables:
    @pytest.mark.parametrize(("household_size", "expected"), sorted(PUBLISHED_2026.items()))
    def test_2026_matches_the_published_table(self, household_size: int, expected: int) -> None:
        assert fpl_annual_cents(2026, household_size) == dollars(expected)

    @pytest.mark.parametrize(("year", "expected"), sorted(PUBLISHED_FOUR_PERSON.items()))
    def test_four_person_figure_matches_every_published_year(
        self, year: int, expected: int
    ) -> None:
        assert fpl_annual_cents(year, 4) == dollars(expected)

    @pytest.mark.parametrize(("year", "figures"), sorted(PUBLISHED_BASE_AND_INCREMENT.items()))
    def test_base_and_increment_match_the_published_figures(
        self, year: int, figures: tuple[int, int]
    ) -> None:
        base, increment = figures
        entry = FPL_YEARS[year]
        assert entry.base_annual_cents == dollars(base)
        assert entry.increment_annual_cents == dollars(increment)

    def test_only_verified_years_are_present(self) -> None:
        assert available_years() == (2023, 2024, 2025, 2026)

    def test_every_year_records_an_hhs_source_url(self) -> None:
        for entry in FPL_YEARS.values():
            assert entry.source_url.startswith("https://aspe.hhs.gov/")
            assert entry.source_title

    @pytest.mark.parametrize("household_size", [9, 10, 15, MAX_HOUSEHOLD_SIZE])
    def test_large_households_use_the_published_increment(self, household_size: int) -> None:
        """Above eight people, "add $5,680 for each additional person" is the
        published method, so it is not an extrapolation."""
        expected = dollars(15_960) + (household_size - 1) * dollars(5_680)
        assert fpl_annual_cents(2026, household_size) == expected


class TestRefusalToGuess:
    @pytest.mark.parametrize("year", [1999, 2019, 2022, 2027, 2100])
    def test_unknown_year_returns_none(self, year: int) -> None:
        assert fpl_annual_cents(year, 2) is None
        assert fpl_threshold_cents(year, 2, 130.0) is None

    @pytest.mark.parametrize("household_size", [-1, 0, MAX_HOUSEHOLD_SIZE + 1, 100])
    def test_out_of_table_household_size_returns_none(self, household_size: int) -> None:
        assert fpl_annual_cents(2026, household_size) is None
        assert fpl_threshold_cents(2026, household_size, 130.0) is None

    def test_table_methods_agree_with_the_module_functions(self) -> None:
        table = default_fpl_table()
        assert table.annual_cents(household_size=99, year=2026) is None
        assert table.annual_threshold_cents(household_size=4, pct=130.0, year=1999) is None
        assert table.monthly_threshold_cents(household_size=4, pct=130.0, year=1999) is None
        assert not table.has_year(2022)
        assert table.source_url(2022) is None
        assert table.source_title(2022) is None


class TestThresholdArithmetic:
    @pytest.mark.parametrize(
        ("household_size", "pct", "expected_dollars"),
        [
            # 185% of the 2026 one-person guideline: 15,960 * 1.85 = 29,526
            (1, 185.0, 29_526),
            # 185% for two: 21,640 * 1.85 = 40,034
            (2, 185.0, 40_034),
            # 185% for three: 27,320 * 1.85 = 50,542
            (3, 185.0, 50_542),
            # 130% for four: 33,000 * 1.30 = 42,900
            (4, 130.0, 42_900),
            # 200% for three: 27,320 * 2 = 54,640
            (3, 200.0, 54_640),
            # 300% for four: 33,000 * 3 = 99,000
            (4, 300.0, 99_000),
            # 400% for one: 15,960 * 4 = 63,840
            (1, 400.0, 63_840),
            # 138% for two, the Medicaid expansion figure: 21,640 * 1.38 = 29,863.20
            (2, 138.0, 29_863.20),
            # 100% is the guideline itself
            (8, 100.0, 55_720),
        ],
    )
    def test_annual_thresholds(
        self, household_size: int, pct: float, expected_dollars: float
    ) -> None:
        expected = round(expected_dollars * 100)
        assert fpl_threshold_cents(2026, household_size, pct) == expected
        assert (
            default_fpl_table().annual_threshold_cents(
                household_size=household_size, pct=pct, year=2026
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("household_size", "pct", "expected_cents"),
        [
            # 29,526.00 / 12 = 2,460.50 exactly
            (1, 185.0, 246_050),
            # 40,034.00 / 12 = 3,336.1666... -> rounds up to 3,336.17
            (2, 185.0, 333_617),
            # 42,900.00 / 12 = 3,575.00 exactly
            (4, 130.0, 357_500),
            # 29,863.20 / 12 = 2,488.60 exactly
            (2, 138.0, 248_860),
        ],
    )
    def test_monthly_thresholds_round_up(
        self, household_size: int, pct: float, expected_cents: int
    ) -> None:
        assert (
            fpl_threshold_cents(2026, household_size, pct, IncomePeriod.MONTHLY) == expected_cents
        )

    def test_rounding_is_always_outward_and_never_loses_more_than_a_cent(self) -> None:
        """A threshold is only ever rounded in the direction that helps the user.

        Being a cent generous can turn a borderline UNLIKELY_MATCH into a
        POTENTIAL_MATCH; rounding the other way could tell someone not to apply
        for help they can get.
        """
        table = default_fpl_table()
        for household_size in range(1, 9):
            for pct in (130.0, 138.0, 185.0, 200.0, 400.0):
                annual = table.annual_threshold_cents(
                    household_size=household_size, pct=pct, year=2026
                )
                monthly = table.monthly_threshold_cents(
                    household_size=household_size, pct=pct, year=2026
                )
                assert annual is not None
                assert monthly is not None
                assert monthly * MONTHS_PER_YEAR >= annual
                assert monthly * MONTHS_PER_YEAR < annual + MONTHS_PER_YEAR

    def test_period_wrapper_dispatches(self) -> None:
        table = default_fpl_table()
        for period in IncomePeriod:
            direct = (
                table.monthly_threshold_cents(household_size=3, pct=185.0, year=2026)
                if period is IncomePeriod.MONTHLY
                else table.annual_threshold_cents(household_size=3, pct=185.0, year=2026)
            )
            assert (
                table.threshold_cents(household_size=3, pct=185.0, year=2026, period=period)
                == direct
            )

    def test_thresholds_increase_with_household_size_and_percentage(self) -> None:
        previous = 0
        for household_size in range(1, MAX_HOUSEHOLD_SIZE + 1):
            value = fpl_threshold_cents(2026, household_size, 185.0)
            assert value is not None
            assert value > previous
            previous = value

        by_pct = [fpl_threshold_cents(2026, 4, pct) for pct in (100.0, 130.0, 185.0, 400.0)]
        assert by_pct == sorted(by_pct)  # type: ignore[type-var]

    def test_fractional_percentages_are_exact_not_floating_point_noise(self) -> None:
        """133.33% of a 2026 four-person guideline is 43,998.90, not 43,998.8999..."""
        assert fpl_threshold_cents(2026, 4, 133.33) == 4_399_890


class TestPurityAndInjection:
    def test_repeated_calls_are_identical(self) -> None:
        first = [fpl_threshold_cents(2026, size, 185.0) for size in range(1, 9)]
        second = [fpl_threshold_cents(2026, size, 185.0) for size in range(1, 9)]
        assert first == second

    def test_a_custom_table_does_not_affect_the_default(self) -> None:
        from datetime import date

        custom = FplTable(
            {
                2030: FplYear(
                    year=2030,
                    base_annual_cents=dollars(20_000),
                    increment_annual_cents=dollars(7_000),
                    source_url="https://aspe.hhs.gov/example",
                    source_title="Illustrative test table",
                    retrieved_on=date(2026, 10, 10),
                )
            }
        )
        assert custom.has_year(2030)
        assert custom.annual_cents(household_size=2, year=2030) == dollars(27_000)
        assert not custom.has_year(2026)
        assert default_fpl_table().has_year(2026)
        assert fpl_annual_cents(2030, 2) is None

    def test_year_entry_is_immutable(self) -> None:
        entry = FPL_YEARS[2026]
        with pytest.raises((AttributeError, TypeError)):
            entry.base_annual_cents = 1  # type: ignore[misc]
