/**
 * The structured half of intake, as one value with one conversion in each
 * direction.
 *
 * The onboarding screen and the "your information" screen collect the same
 * fields and must agree about what each control means, so the shape and the
 * mapping to `ProfileDraft` live here rather than in either screen. The screens
 * then only deal with `StructuredAnswers`, which is the form's vocabulary, and
 * never hand-build a wire object.
 *
 * `null` means "not stated" everywhere in this module, and "not stated" never
 * disqualifies anyone - it is what produces a "need a bit more info" status
 * rather than a rejection.
 */

import type {
  IncomePeriod,
  NeedCategory,
  ProfileDraft,
  StudentStatus,
  UserProfile,
} from "@/lib/types";

import { findIncomeBand, incomeBandsFor } from "./options";

export interface StructuredAnswers {
  /** Two-letter state code, as `StateCode` expects. */
  readonly state: string | null;
  /** County name without the word "County". */
  readonly county: string | null;
  readonly householdSize: number | null;
  readonly studentStatus: StudentStatus | null;
  /** Always set, because a band is meaningless without it. */
  readonly incomePeriod: IncomePeriod;
  /** Id of the chosen band from `options`, or `null` for "not stated". */
  readonly incomeBandId: string | null;
  readonly needs: readonly NeedCategory[];
}

/**
 * The starting state. New Jersey is pre-selected because every program in the
 * catalog is a New Jersey or Rutgers program; it is a plain select and can be
 * changed.
 */
export const EMPTY_ANSWERS: StructuredAnswers = {
  state: "NJ",
  county: null,
  householdSize: null,
  studentStatus: null,
  incomePeriod: "MONTHLY",
  incomeBandId: null,
  needs: [],
};

export const HOUSEHOLD_SIZE_MIN = 1;
export const HOUSEHOLD_SIZE_MAX = 20;

/**
 * The wire object, or `null` when the user filled nothing in.
 *
 * Unset fields are omitted rather than sent as `null`. Both are accepted by the
 * backend, but omitting keeps the request an honest statement of what the user
 * actually said.
 */
export function toProfileDraft(answers: StructuredAnswers): ProfileDraft | null {
  const draft: ProfileDraft = {};

  if (answers.state !== null && answers.state.length > 0) draft.state = answers.state;
  if (answers.county !== null && answers.county.length >= 2) draft.county = answers.county;
  if (answers.householdSize !== null) draft.household_size = answers.householdSize;
  if (answers.studentStatus !== null) draft.student_status = answers.studentStatus;
  if (answers.needs.length > 0) draft.needs = [...answers.needs];

  if (answers.incomeBandId !== null) {
    const band = incomeBandsFor(answers.incomePeriod).find(
      (candidate) => candidate.id === answers.incomeBandId,
    );
    if (band !== undefined) {
      draft.income_period = answers.incomePeriod;
      draft.income_min_cents = band.minCents;
      // Omitted rather than null for the open-ended top band: "more than
      // $6,500" has no upper bound, and `null` would read as one of zero.
      if (band.maxCents !== null) draft.income_max_cents = band.maxCents;
    }
  }

  return Object.keys(draft).length === 0 ? null : draft;
}

/**
 * Pre-fill the form from a profile the backend already holds.
 *
 * An income that does not line up with a band leaves the radio unselected, which
 * is deliberate: silently rounding someone's income to the nearest band would
 * change what they told us.
 */
export function fromProfile(profile: UserProfile | null): StructuredAnswers {
  if (profile === null) return EMPTY_ANSWERS;
  const period: IncomePeriod = profile.income_period ?? "MONTHLY";
  const band = findIncomeBand(
    profile.income_period,
    profile.income_min_cents,
    profile.income_max_cents,
  );
  return {
    state: profile.state ?? null,
    county: profile.county ?? null,
    householdSize: profile.household_size ?? null,
    studentStatus: profile.student_status ?? null,
    incomePeriod: period,
    incomeBandId: band?.id ?? null,
    needs: profile.needs ?? [],
  };
}
