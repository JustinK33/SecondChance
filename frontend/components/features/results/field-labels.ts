/**
 * Profile field names, said the way a person would say them.
 *
 * The rule engine reports what it could not check as machine field names -
 * `income_min_cents`, `flags.has_children`. Putting those on screen would be
 * asking someone to debug our data model, so every name the backend can emit
 * has a phrase here. The set of names is closed (`PROFILE_FIELD_NAMES` in
 * `app/schemas/profile.py`), so this map can be complete rather than hopeful.
 *
 * Anything unrecognised falls back to a de-underscored version of the name
 * rather than being dropped: a missing requirement the user cannot see is worse
 * than one phrased awkwardly.
 */

import { humanizeEnum } from "@/lib/format";

/** Phrase for a single field, written to sit after "we still need". */
const FIELD_PHRASES: Record<string, string> = {
  state: "which state you live in",
  county: "which New Jersey county you live in",
  student_status: "whether you are a student, and full or part time",
  household_size: "how many people you shop and cook for",
  household_composition: "who else is in your household",
  income_min_cents: "roughly what your household takes in",
  income_max_cents: "roughly what your household takes in",
  income_period: "whether that income figure is per month or per year",
  income: "roughly what your household takes in",
  employment_status: "your work situation at the moment",
  housing_situation: "your housing situation",
  needs: "the kind of help you are looking for",
  age: "your age",
};

/** Phrase for a `flags.<name>` field. Keys are `ProfileFlag` values. */
const FLAG_PHRASES: Record<string, string> = {
  receives_snap: "whether you already get SNAP (food stamps)",
  receives_wic: "whether you already get WIC",
  receives_medicaid: "whether you already have Medicaid or NJ FamilyCare",
  receives_tanf: "whether you already get TANF or WorkFirst NJ",
  receives_ssi: "whether you already get SSI",
  receives_ssdi: "whether you already get SSDI",
  receives_unemployment: "whether you are getting unemployment",
  receives_free_school_meals: "whether a child in your home gets free school meals",
  is_pregnant: "whether anyone in your household is pregnant",
  has_children: "whether there are children in your household",
  is_veteran: "whether you are a veteran",
  is_disabled: "whether you have a disability",
  is_senior: "whether you are 60 or older",
  is_caregiver: "whether you care for someone else",
  is_rutgers_affiliated: "whether you are a Rutgers student or employee",
  has_utility_shutoff_notice: "whether you have had a shut-off notice",
  has_eviction_notice: "whether you have had an eviction notice",
  filed_taxes_last_year: "whether you filed taxes last year",
};

/** The plain-language phrase for one profile field name. */
export function describeMissingField(field: string): string {
  if (field.startsWith("flags.")) {
    const flag = field.slice("flags.".length);
    return FLAG_PHRASES[flag] ?? humanizeEnum(flag).toLowerCase();
  }
  return FIELD_PHRASES[field] ?? humanizeEnum(field).toLowerCase();
}

/**
 * Phrases for a list of field names, de-duplicated.
 *
 * The minimum and maximum of an income range are two fields that read as one
 * question, so the same phrase must not appear twice.
 */
export function describeMissingFields(fields: readonly string[]): string[] {
  const seen = new Set<string>();
  const phrases: string[] = [];
  for (const field of fields) {
    const phrase = describeMissingField(field);
    if (seen.has(phrase)) continue;
    seen.add(phrase);
    phrases.push(phrase);
  }
  return phrases;
}
