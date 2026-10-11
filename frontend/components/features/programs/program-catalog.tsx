"use client";

import { ArrowRight, ExternalLink, Search } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { LastVerified } from "@/components/ui/last-verified";
import { LoadingCards } from "@/components/ui/loading-state";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { humanizeEnum, pluralize } from "@/lib/format";
import { usePrograms } from "@/lib/hooks";
import type { ListProgramsQuery, Program, ProgramCategory } from "@/lib/types";

import { UnverifiedNotice } from "../results/unverified-notice";

/**
 * The browsable catalog, for someone who just wants to look around.
 *
 * Deliberately not gated behind the intake flow. Plenty of people are not ready
 * to describe their situation to a computer, and a plain searchable list of real
 * programs with their official links is useful on its own. This is also the only
 * screen that works with no identity at all, because `GET /programs` is the one
 * unauthenticated data route.
 *
 * Filters are server-side - the backend owns the geography semantics, including
 * the rule that a state-wide program still matches a county query, which is what
 * stops a county filter from hiding SNAP. The one client-side concession is a
 * debounce, so typing a word does not fire a request per keystroke.
 */
const PAGE_SIZE = 20;
const SEARCH_DEBOUNCE_MS = 300;
const ALL = "ALL";

/** Every `ProgramCategory`, in the order they are most asked about. */
const CATEGORIES: readonly ProgramCategory[] = [
  "FOOD",
  "HEALTHCARE",
  "UTILITIES",
  "HOUSING",
  "CHILDCARE",
  "CASH_TAX_CREDITS",
  "UNIVERSITY_EMERGENCY_AID",
  "COMMUNITY_SERVICES",
] as const;

/** New Jersey's 21 counties. Public geography, not program data. */
const NJ_COUNTIES: readonly string[] = [
  "Atlantic",
  "Bergen",
  "Burlington",
  "Camden",
  "Cape May",
  "Cumberland",
  "Essex",
  "Gloucester",
  "Hudson",
  "Hunterdon",
  "Mercer",
  "Middlesex",
  "Monmouth",
  "Morris",
  "Ocean",
  "Passaic",
  "Salem",
  "Somerset",
  "Sussex",
  "Union",
  "Warren",
] as const;

export function ProgramCatalog(): ReactNode {
  const [searchText, setSearchText] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<string>(ALL);
  const [county, setCounty] = useState<string>(ALL);
  const [limit, setLimit] = useState<number>(PAGE_SIZE);

  // A new filter means a new first page. Keeping a grown `limit` would quietly
  // request eighty rows of something the user has not seen yet, so every filter
  // change resets it - done in the handlers rather than in an effect, because an
  // effect that only corrects state is a render the user pays for.
  useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(searchText.trim());
      setLimit(PAGE_SIZE);
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchText]);

  const listQuery = useMemo<ListProgramsQuery>(
    () => ({
      ...(query.length === 0 ? {} : { q: query }),
      ...(category === ALL ? {} : { category: [category as ProgramCategory] }),
      ...(county === ALL ? {} : { county }),
      limit,
    }),
    [query, category, county, limit],
  );

  const { programs, total, isLoading, isRefreshing, error, refetch, isEmpty } =
    usePrograms(listQuery);
  const hasFilters = query.length > 0 || category !== ALL || county !== ALL;

  function chooseCategory(next: string): void {
    setCategory(next);
    setLimit(PAGE_SIZE);
  }

  function chooseCounty(next: string): void {
    setCounty(next);
    setLimit(PAGE_SIZE);
  }

  function clearFilters(): void {
    setSearchText("");
    setQuery("");
    setCategory(ALL);
    setCounty(ALL);
    setLimit(PAGE_SIZE);
  }

  return (
    <div className="space-y-6">
      <section aria-labelledby="catalog-filters" className="space-y-4">
        <h2 id="catalog-filters" className="sr-only">
          Search and filter
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="program-search">Search programs</Label>
            <div className="relative">
              <Search
                aria-hidden="true"
                className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2"
              />
              <Input
                id="program-search"
                type="search"
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
                placeholder="rent, food, electricity..."
                className="touch-row pl-9"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="program-category">Kind of help</Label>
            <Select value={category} onValueChange={chooseCategory}>
              <SelectTrigger id="program-category" className="touch-row w-full">
                <SelectValue placeholder="Any kind of help" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>Any kind of help</SelectItem>
                {CATEGORIES.map((value) => (
                  <SelectItem key={value} value={value}>
                    {humanizeEnum(value)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="program-county">County</Label>
            <Select value={county} onValueChange={chooseCounty}>
              <SelectTrigger id="program-county" className="touch-row w-full">
                <SelectValue placeholder="Anywhere in New Jersey" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>Anywhere in New Jersey</SelectItem>
                {NJ_COUNTIES.map((name) => (
                  <SelectItem key={name} value={name}>
                    {name} County
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <p role="status" aria-live="polite" className="text-muted-foreground text-sm">
            {isLoading
              ? "Loading programs..."
              : total === 0
                ? "No programs match those filters."
                : `Showing ${programs.length} of ${pluralize(total, "program")}.`}
          </p>
          {hasFilters ? (
            <Button variant="ghost" size="lg" className="touch-row" onClick={clearFilters}>
              Clear filters
            </Button>
          ) : null}
        </div>
        <p className="text-muted-foreground text-xs">
          Filtering by county keeps programs that cover the whole state, so nothing state-wide is
          hidden.
        </p>
      </section>

      {error !== null && programs.length === 0 ? (
        <ErrorState error={error} title="We could not load the programs" onRetry={refetch} />
      ) : null}

      {isLoading && programs.length === 0 ? <LoadingCards count={4} /> : null}

      {isEmpty ? (
        <EmptyState
          title="Nothing matches those filters"
          description="Try a different kind of help, or clear the filters to see everything we have checked."
          action={
            <Button variant="outline" size="lg" className="touch-row" onClick={clearFilters}>
              Clear filters
            </Button>
          }
        />
      ) : null}

      {programs.length > 0 ? (
        <ol className="space-y-4" aria-busy={isRefreshing}>
          {programs.map((program) => (
            <li key={program.id}>
              <ProgramListCard program={program} />
            </li>
          ))}
        </ol>
      ) : null}

      {programs.length < total ? (
        <Button
          variant="outline"
          size="lg"
          className="touch-row"
          onClick={() => setLimit((current) => current + PAGE_SIZE)}
          disabled={isRefreshing}
        >
          {isRefreshing
            ? "Loading..."
            : `Show ${Math.min(PAGE_SIZE, total - programs.length)} more`}
        </Button>
      ) : null}
    </div>
  );
}

/** One catalog row. No match status here: nothing has been evaluated. */
function ProgramListCard({ program }: { program: Program }): ReactNode {
  return (
    <Card className="gap-3">
      <CardHeader className="gap-1">
        <h3 className="text-base leading-snug font-semibold">{program.official_name}</h3>
        <p className="text-muted-foreground text-sm">
          Run by {program.administering_org} · {humanizeEnum(program.category)} ·{" "}
          {program.geo_scope_label}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        <UnverifiedNotice
          status={program.verification_status}
          programName={program.official_name}
          officialUrl={program.official_url}
        />
        <p className="reading-width text-sm">{program.plain_language_summary}</p>
        <LastVerified
          status={program.verification_status}
          lastVerifiedAt={program.last_verified_at}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button asChild variant="outline" size="lg" className="touch-row">
            <Link href={ROUTES.program(program.id)}>
              Details and requirements
              <ArrowRight aria-hidden="true" className="size-4" />
            </Link>
          </Button>
          <Button asChild variant="ghost" size="lg" className="touch-row">
            <a href={program.official_url} target="_blank" rel="noopener noreferrer">
              Official page
              <ExternalLink aria-hidden="true" className="size-4" />
              <span className="sr-only">
                for {program.official_name} (opens in a new tab, leaving SecondChance)
              </span>
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
