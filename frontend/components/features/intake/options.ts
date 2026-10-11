/**
 * The plain-language wording for every structured choice on the intake screens.
 *
 * Two reasons this is one file rather than inline literals:
 *
 * 1. The enum members are the wire values and must be spelled exactly as the
 *    backend spells them. One table is one place to get that wrong.
 * 2. The same wording has to appear on onboarding and again on the "your
 *    information" screen. Divergent copy for the same field reads like a
 *    different question and gets a different answer.
 *
 * Income is deliberately a band rather than a figure. Every rule the engine can
 * actually check compares against a threshold, so a band is enough to evaluate,
 * and asking for a band rather than a number is both less intrusive and more
 * honest about the precision we need.
 */

import type {
  EmploymentStatus,
  HousingSituation,
  IncomePeriod,
  NeedCategory,
  StudentStatus,
} from "@/lib/types";

export interface Choice<T extends string> {
  readonly value: T;
  readonly label: string;
  /** One short line under the label, where it earns its space. */
  readonly help?: string;
}

/** States we hold programs for, plus an honest "somewhere else". */
export const STATE_CHOICES: readonly Choice<string>[] = [
  { value: "NJ", label: "New Jersey" },
  { value: "NY", label: "New York" },
  { value: "PA", label: "Pennsylvania" },
  { value: "DE", label: "Delaware" },
  { value: "CT", label: "Connecticut" },
] as const;

/** New Jersey's counties, as the state prints them. */
export const NJ_COUNTIES: readonly string[] = [
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

/**
 * Student status. `UNKNOWN` is not offered as a choice: leaving the field unset
 * already means "not stated", and offering both would be two names for one
 * thing.
 */
export const STUDENT_CHOICES: readonly Choice<StudentStatus>[] = [
  { value: "NOT_A_STUDENT", label: "Not a student" },
  { value: "UNDERGRAD_FULL_TIME", label: "Undergraduate, full time" },
  { value: "UNDERGRAD_PART_TIME", label: "Undergraduate, part time" },
  { value: "GRADUATE", label: "Graduate or professional student" },
  { value: "OTHER", label: "Student, but none of these fit" },
] as const;

export const EMPLOYMENT_CHOICES: readonly Choice<EmploymentStatus>[] = [
  { value: "EMPLOYED_FULL_TIME", label: "Working full time" },
  { value: "EMPLOYED_PART_TIME", label: "Working part time" },
  { value: "SELF_EMPLOYED", label: "Self-employed" },
  { value: "SEASONAL_OR_GIG", label: "Seasonal, gig or on-and-off work" },
  { value: "UNEMPLOYED", label: "Not working right now" },
  { value: "UNABLE_TO_WORK", label: "Unable to work" },
  { value: "RETIRED", label: "Retired" },
  { value: "NOT_IN_LABOR_FORCE", label: "Not looking for work right now" },
  { value: "OTHER", label: "None of these fit" },
] as const;

export const HOUSING_CHOICES: readonly Choice<HousingSituation>[] = [
  { value: "RENTING", label: "Renting" },
  { value: "OWN_HOME", label: "I own my home" },
  { value: "LIVING_WITH_FAMILY", label: "Staying with family or friends" },
  { value: "UNIVERSITY_HOUSING", label: "University housing" },
  { value: "TEMPORARY_OR_SHELTER", label: "Temporary housing or a shelter" },
  { value: "UNHOUSED", label: "I do not have housing right now" },
  { value: "OTHER", label: "None of these fit" },
] as const;

/**
 * What someone needs help with, in their framing rather than a program's.
 * `UNKNOWN` is never offered: an empty selection already says it.
 */
export const NEED_CHOICES: readonly Choice<NeedCategory>[] = [
  { value: "FOOD", label: "Food assistance", help: "Groceries, meals, a food pantry" },
  {
    value: "HEALTHCARE",
    label: "Health coverage and care",
    help: "Doctor visits, prescriptions, insurance",
  },
  { value: "UTILITIES", label: "Utility assistance", help: "Electric, gas, water, heating" },
  { value: "HOUSING", label: "Housing assistance", help: "Rent help, deposits, eviction" },
  { value: "CHILDCARE", label: "Child care assistance", help: "Day care, after-school care" },
  { value: "TRANSPORTATION", label: "Transportation", help: "Bus, train, car repairs" },
  {
    value: "EDUCATION_COSTS",
    label: "Education expenses",
    help: "Tuition, books, fees, supplies",
  },
  { value: "INCOME_SUPPORT", label: "Income support", help: "Cash assistance, benefits" },
  { value: "INTERNET_PHONE", label: "Internet and phone service", help: "Service, a device" },
  { value: "LEGAL_HELP", label: "Legal assistance", help: "Free or low-cost legal advice" },
  {
    value: "MENTAL_HEALTH",
    label: "Mental health services",
    help: "Counselling, someone to talk to",
  },
  {
    value: "EMERGENCY_EXPENSE",
    label: "Emergency expenses",
    help: "A one-time cost you cannot cover",
  },
  { value: "OTHER", label: "Other", help: "You can describe it above" },
] as const;

/** A selectable income band, held in integer cents like the profile is. */
export interface IncomeBand {
  /** Stable id for the radio value. Not sent to the backend. */
  readonly id: string;
  readonly label: string;
  readonly minCents: number | null;
  readonly maxCents: number | null;
}

/** Monthly bands, roughly tracking where New Jersey program thresholds sit. */
export const MONTHLY_INCOME_BANDS: readonly IncomeBand[] = [
  { id: "m0", label: "No income coming in right now", minCents: 0, maxCents: 0 },
  { id: "m1", label: "Under $1,000 a month", minCents: 0, maxCents: 100_000 },
  { id: "m2", label: "$1,000 to $2,000 a month", minCents: 100_000, maxCents: 200_000 },
  { id: "m3", label: "$2,000 to $3,000 a month", minCents: 200_000, maxCents: 300_000 },
  { id: "m4", label: "$3,000 to $4,500 a month", minCents: 300_000, maxCents: 450_000 },
  { id: "m5", label: "$4,500 to $6,500 a month", minCents: 450_000, maxCents: 650_000 },
  { id: "m6", label: "More than $6,500 a month", minCents: 650_000, maxCents: null },
] as const;

/** Yearly bands, for someone who knows the figure from a tax return. */
export const ANNUAL_INCOME_BANDS: readonly IncomeBand[] = [
  { id: "a0", label: "No income coming in right now", minCents: 0, maxCents: 0 },
  { id: "a1", label: "Under $15,000 a year", minCents: 0, maxCents: 1_500_000 },
  { id: "a2", label: "$15,000 to $25,000 a year", minCents: 1_500_000, maxCents: 2_500_000 },
  { id: "a3", label: "$25,000 to $40,000 a year", minCents: 2_500_000, maxCents: 4_000_000 },
  { id: "a4", label: "$40,000 to $60,000 a year", minCents: 4_000_000, maxCents: 6_000_000 },
  { id: "a5", label: "$60,000 to $85,000 a year", minCents: 6_000_000, maxCents: 8_500_000 },
  { id: "a6", label: "More than $85,000 a year", minCents: 8_500_000, maxCents: null },
] as const;

export function incomeBandsFor(period: IncomePeriod): readonly IncomeBand[] {
  return period === "ANNUAL" ? ANNUAL_INCOME_BANDS : MONTHLY_INCOME_BANDS;
}

/**
 * The band whose bounds match a stored profile, or `null` when nothing matches.
 *
 * Used to pre-select the radio on the "your information" screen. An amount the
 * model extracted from a narrative will often not line up with a band exactly;
 * returning `null` is correct in that case, because inventing a selection would
 * silently round someone's income.
 */
export function findIncomeBand(
  period: IncomePeriod | null | undefined,
  minCents: number | null | undefined,
  maxCents: number | null | undefined,
): IncomeBand | null {
  if (period === null || period === undefined) return null;
  const bands = incomeBandsFor(period);
  const min = minCents ?? null;
  const max = maxCents ?? null;
  return bands.find((band) => band.minCents === min && band.maxCents === max) ?? null;
}

/** The example that shows someone what "in your own words" is allowed to mean. */
export const NARRATIVE_EXAMPLE = `I lost my part-time job last month and I'm a full-time student at Rutgers in New Brunswick. I live with my mom and my younger brother in Middlesex County. Rent is due in two weeks and we are running low on groceries. My electric bill has a shut-off notice on it.`;

/** Narrative length limits, matching `IntakeRequest.narrative`. */
export const NARRATIVE_MIN_LENGTH = 10;
export const NARRATIVE_MAX_LENGTH = 4000;
