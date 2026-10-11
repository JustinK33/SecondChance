/**
 * Display formatting. Pure functions, no state, safe on the server.
 *
 * Money arrives as integer cents and as a *range*, because a profile never
 * carries a single income figure. Rendering a range as one number would be a
 * precision the data does not have.
 */

import type { IncomePeriod, SourceFreshness } from "./types";

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

/** Integer cents to `$1,234`. */
export function formatCents(cents: number): string {
  return USD.format(Math.round(cents / 100));
}

/**
 * An income range, as the user should read it.
 *
 * `null` for both bounds means the income was never stated, which is not zero.
 */
export function formatIncomeRange(
  minCents: number | null | undefined,
  maxCents: number | null | undefined,
  period: IncomePeriod | null | undefined,
): string {
  const suffix = period === "MONTHLY" ? " a month" : period === "ANNUAL" ? " a year" : "";
  const hasMin = typeof minCents === "number";
  const hasMax = typeof maxCents === "number";
  if (!hasMin && !hasMax) return "Not provided";
  if (hasMin && hasMax) {
    if (minCents === maxCents) return `${formatCents(minCents)}${suffix}`;
    return `${formatCents(minCents)} to ${formatCents(maxCents)}${suffix}`;
  }
  if (hasMin) return `${formatCents(minCents)} or more${suffix}`;
  return `Up to ${formatCents(maxCents as number)}${suffix}`;
}

const LONG_DATE = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

/** An ISO timestamp as `January 5, 2026`. Returns `null` for an unusable value. */
export function formatDate(iso: string | null | undefined): string | null {
  if (typeof iso !== "string" || iso.length === 0) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return LONG_DATE.format(date);
}

/** Relative age in words: `3 days ago`. Returns `null` for an unusable value. */
export function formatRelativeAge(iso: string | null | undefined, now?: Date): string | null {
  if (typeof iso !== "string" || iso.length === 0) return null;
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return null;
  const days = Math.floor(((now ?? new Date()).getTime() - then.getTime()) / 86_400_000);
  if (days < 0) return "just now";
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return months === 1 ? "last month" : `${months} months ago`;
  const years = Math.floor(days / 365);
  return years === 1 ? "last year" : `${years} years ago`;
}

/** How much confidence to place in a source's age. */
export function freshnessLabel(freshness: SourceFreshness): string {
  switch (freshness) {
    case "FRESH":
      return "Recently checked";
    case "AGING":
      return "Checked a while ago";
    case "STALE":
      return "Needs re-checking";
  }
}

/** Title-case an enum member for display: `UNDERGRAD_FULL_TIME` -> `Undergrad full time`. */
export function humanizeEnum(value: string): string {
  const words = value.toLowerCase().replace(/_/g, " ").trim();
  return words.length === 0 ? value : words.charAt(0).toUpperCase() + words.slice(1);
}

/** `1 program` / `3 programs`, without the caller writing the branch. */
export function pluralize(count: number, singular: string, plural?: string): string {
  return `${count} ${count === 1 ? singular : (plural ?? `${singular}s`)}`;
}

/**
 * Official-sounding display names for the program categories.
 *
 * `humanizeEnum` is fine for debug text but produces things like "Cash tax
 * credits", which reads as a database value rather than a category an agency
 * would print. These are the names users see, so they are written out by hand.
 * The plain-language explanation of a category belongs in the body copy, not
 * here - the label's job is to look like it came from an official list.
 */
const PROGRAM_CATEGORY_LABELS: Record<string, string> = {
  FOOD: "Food assistance",
  HEALTHCARE: "Health coverage and care",
  UTILITIES: "Utility assistance",
  HOUSING: "Housing assistance",
  CHILDCARE: "Child care assistance",
  CASH_TAX_CREDITS: "Cash and tax credits",
  UNIVERSITY_EMERGENCY_AID: "University emergency aid",
  COMMUNITY_SERVICES: "Community services",
  TRANSPORTATION: "Transportation",
  EDUCATION_COSTS: "Education expenses",
};

/** Display name for a program category, falling back to a readable form. */
export function programCategoryLabel(category: string): string {
  return PROGRAM_CATEGORY_LABELS[category] ?? humanizeEnum(category);
}
