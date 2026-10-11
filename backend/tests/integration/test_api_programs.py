"""``/api/v1/programs`` - the public catalog, and the filters it has to get right.

Two properties here are product rules rather than plumbing.

**A county filter must not hide a state-wide program.** Somebody in Middlesex
County filtering for their county has to still see SNAP, which has no counties
of its own because it covers the whole state. Getting this backwards would
quietly remove the largest programs from the results of the people who need
them.

**No eligibility is assessed.** A status is a statement about one person's
situation, and these routes have no person. So the catalog returns the record,
its sources and the requirements as author-written prose, and nothing that
reads like a decision.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import (
    AssistanceProgram,
    MatchStatus,
    ProgramCategory,
    ProgramDetail,
    ProgramPage,
    VerificationStatus,
)
from tests.conftest import make_program, make_ruleset
from tests.security.harness import ApiHarness, rolled_back_api, seed_catalog

__all__ = ["rolled_back_api"]

#: The seeded catalog, keyed by slug, as the fixture below hands it over.
type Catalog = dict[str, AssistanceProgram]

pytestmark = pytest.mark.db


@pytest.fixture
async def catalog(session: AsyncSession) -> Catalog:
    """Four fictional programs chosen to exercise every filter."""
    food = make_program("example-grocery-help", category=ProgramCategory.FOOD)
    statewide = make_program(
        "example-statewide-food",
        category=ProgramCategory.FOOD,
        counties=(),
        keywords=("groceries", "statewide"),
    )
    utilities = make_program(
        "example-utility-help",
        category=ProgramCategory.UTILITIES,
        keywords=("electric", "utility"),
    )
    unverified = make_program(
        "example-unverified-help",
        category=ProgramCategory.HOUSING,
        verification_status=VerificationStatus.UNVERIFIED,
        active_rule_version=None,
    )
    programs = [food, statewide, utilities, unverified]
    await seed_catalog(session, programs, [make_ruleset(food.slug)])
    return {program.slug: program for program in programs}


class TestListing:
    async def test_the_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs")
        assert response.status_code == 200, response.text
        page = ProgramPage.model_validate(response.json())
        assert page.total == 4
        assert {item.slug for item in page.items} == set(catalog)
        for item in page.items:
            assert item.administering_org, "a program must name who administers it"
            assert item.official_url.startswith("https://")

    async def test_it_needs_no_authentication(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        assert (await rolled_back_api.client.get("/api/v1/programs")).status_code == 200

    async def test_no_status_language_appears_anywhere(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs")
        text = response.text.lower()
        for status in MatchStatus:
            assert status.value.lower() not in text
        assert "eligible" not in text

    async def test_filtering_by_category(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs?category=FOOD")
        page = ProgramPage.model_validate(response.json())
        assert {item.slug for item in page.items} == {
            "example-grocery-help",
            "example-statewide-food",
        }

    async def test_the_category_filter_is_repeatable(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get(
            "/api/v1/programs?category=FOOD&category=UTILITIES"
        )
        page = ProgramPage.model_validate(response.json())
        assert len(page.items) == 3

    async def test_an_unknown_category_is_a_validation_error(
        self, rolled_back_api: ApiHarness
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs?category=CRYPTO")
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert error["detail"]["fields"] == ["category.0"]
        assert "CRYPTO" not in response.text, "the rejected value must not be echoed"

    async def test_filtering_by_state(self, rolled_back_api: ApiHarness, catalog: Catalog) -> None:
        assert (
            ProgramPage.model_validate(
                (await rolled_back_api.client.get("/api/v1/programs?state=NJ")).json()
            ).total
            == 4
        )
        assert (
            ProgramPage.model_validate(
                (await rolled_back_api.client.get("/api/v1/programs?state=NY")).json()
            ).total
            == 0
        )

    async def test_a_county_filter_does_not_hide_a_statewide_program(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs?county=Middlesex")
        slugs = {item["slug"] for item in response.json()["items"]}
        assert "example-grocery-help" in slugs, "the county-scoped program should match"
        assert "example-statewide-food" in slugs, (
            "a program with no counties covers every county, so it must still appear"
        )

    async def test_full_text_search(self, rolled_back_api: ApiHarness, catalog: Catalog) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs?q=electric")
        page = ProgramPage.model_validate(response.json())
        assert {item.slug for item in page.items} == {"example-utility-help"}

    async def test_filtering_by_verification_status(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get(
            "/api/v1/programs?verification_status=UNVERIFIED"
        )
        page = ProgramPage.model_validate(response.json())
        assert {item.slug for item in page.items} == {"example-unverified-help"}
        assert page.items[0].last_verified_at is None

    @pytest.mark.parametrize(
        "query",
        ["limit=0", "limit=201", "offset=-1", "state=NEWJERSEY", "q=a"],
    )
    async def test_a_bad_window_or_filter_is_refused(
        self, rolled_back_api: ApiHarness, query: str
    ) -> None:
        response = await rolled_back_api.client.get(f"/api/v1/programs?{query}")
        assert response.status_code == 422, f"{query} -> {response.status_code}"
        assert response.json()["error"]["code"] == "validation_error"

    async def test_the_window_is_applied_and_does_not_overlap(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        first = ProgramPage.model_validate(
            (await rolled_back_api.client.get("/api/v1/programs?limit=2&offset=0")).json()
        )
        second = ProgramPage.model_validate(
            (await rolled_back_api.client.get("/api/v1/programs?limit=2&offset=2")).json()
        )
        assert first.total == second.total == 4
        assert len(first.items) == len(second.items) == 2
        assert {item.id for item in first.items}.isdisjoint({item.id for item in second.items})


class TestDetail:
    async def test_the_shape_matches_the_declared_model(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        program = catalog["example-grocery-help"]
        response = await rolled_back_api.client.get(f"/api/v1/programs/{program.id}")
        assert response.status_code == 200, response.text
        detail = ProgramDetail.model_validate(response.json())
        assert detail.program.id == program.id
        assert detail.sources, "a program detail must carry the sources behind it"
        assert all(source.retrieved_at for source in detail.sources)
        assert detail.requirements, "requirements are rendered from the approved rule set"
        assert detail.rule_version == "2026.1"

    async def test_a_slug_resolves_to_the_same_record(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        """A shareable link should read ``/programs/nj-snap``, not a UUID."""
        program = catalog["example-grocery-help"]
        by_slug = await rolled_back_api.client.get(f"/api/v1/programs/{program.slug}")
        by_id = await rolled_back_api.client.get(f"/api/v1/programs/{program.id}")
        assert by_slug.status_code == by_id.status_code == 200
        assert by_slug.json() == by_id.json()

    async def test_requirements_are_author_written_prose_with_a_source(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        response = await rolled_back_api.client.get("/api/v1/programs/example-grocery-help")
        detail = ProgramDetail.model_validate(response.json())
        for bullet in detail.requirements:
            assert bullet.text
            assert "eligible" not in bullet.text.lower()

    async def test_a_program_with_no_approved_rules_renders_no_requirements(
        self, rolled_back_api: ApiHarness, catalog: Catalog
    ) -> None:
        """Rather than inventing a requirement list it does not have."""
        response = await rolled_back_api.client.get("/api/v1/programs/example-unverified-help")
        detail = ProgramDetail.model_validate(response.json())
        assert detail.requirements == []
        assert detail.rule_version is None
        assert detail.limitations, "the gap must be shown, not hidden"

    @pytest.mark.parametrize("key", ["no-such-program", "00000000-0000-0000-0000-000000000000"])
    async def test_an_unknown_program_is_not_found(
        self, rolled_back_api: ApiHarness, catalog: Catalog, key: str
    ) -> None:
        response = await rolled_back_api.client.get(f"/api/v1/programs/{key}")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert error["request_id"]

    async def test_a_random_uuid_is_not_found_rather_than_an_error(
        self, rolled_back_api: ApiHarness
    ) -> None:
        response = await rolled_back_api.client.get(f"/api/v1/programs/{uuid.uuid4()}")
        assert response.status_code == 404
