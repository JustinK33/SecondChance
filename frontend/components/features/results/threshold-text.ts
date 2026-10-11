/**
 * The engine's `threshold_applied` string, turned into something a person reads.
 *
 * The eligibility engine records what a condition was measured against so the
 * result can be audited. Most of those strings are already written for people -
 * `income of at most $54,943 a year`, `age of at least 60` - but the set
 * conditions are not: they come out as `housing_situation is one of
 * LIVING_WITH_FAMILY, OTHER, RENTING` and the boolean ones as
 * `is_pregnant is true`.
 *
 * Putting those on screen breaks a product rule - no raw enum values and no
 * internal field names in the interface - and it also simply does not help
 * anyone. So each clause is rewritten here from the same wording tables the
 * intake screens use, which is what stops the two halves of the product from
 * describing the same field differently.
 *
 * When a clause cannot be rewritten, the whole line is dropped rather than
 * shown half-translated. The row above it already carries the author-written
 * description of the requirement and its outcome, so nothing a user can act on
 * is lost - only the audit detail, which belongs in the API response and not in
 * someone's results.
 */

import {
  EMPLOYMENT_CHOICES,
  HOUSING_CHOICES,
  NEED_CHOICES,
  STATE_CHOICES,
  STUDENT_CHOICES,
} from "../intake/options";
import type { Choice } from "../intake/options";

/** Anything that still looks like a field name or an enum member. */
const MACHINE_TOKEN = /[a-z0-9]+_[a-z0-9]+|\b[A-Z][A-Z0-9]{1,}\b/;

/** `field is one of A, B` and `field is one of A`. */
const SET_CLAUSE = /^([a-z_.]+) is one of (.+)$/;

/** `field is true` / `field is false`. */
const BOOLEAN_CLAUSE = /^([a-z_.]+) is (true|false)$/;

/**
 * How a set condition on each field reads as a whole clause.
 *
 * A single noun plus "being" does not fit every field - "being a student being
 * not a student" is what that produces - so each field owns its own sentence.
 */
const FIELD_CLAUSES: Record<string, (values: string) => string> = {
  student_status: (values) => `being ${values}`,
  housing_situation: (values) => `your housing situation being ${values}`,
  employment_status: (values) => `your work situation being ${values}`,
  needs: (values) => `looking for help with ${values}`,
};

/**
 * What a `true` (and a `false`) on each flag actually requires, as a noun
 * phrase that can follow "checked against".
 *
 * Written out rather than derived from the flag name: "is_pregnant is true"
 * de-underscored reads "is pregnant is true", which is not English.
 */
const FLAG_REQUIREMENTS: Record<string, { readonly yes: string; readonly no: string }> = {
  receives_snap: {
    yes: "already getting SNAP",
    no: "not already getting SNAP",
  },
  receives_wic: { yes: "already getting WIC", no: "not already getting WIC" },
  receives_medicaid: {
    yes: "already having Medicaid or NJ FamilyCare",
    no: "not already having Medicaid or NJ FamilyCare",
  },
  receives_tanf: {
    yes: "already getting TANF or WorkFirst NJ",
    no: "not already getting TANF or WorkFirst NJ",
  },
  receives_ssi: { yes: "already getting SSI", no: "not already getting SSI" },
  receives_ssdi: { yes: "already getting SSDI", no: "not already getting SSDI" },
  receives_unemployment: {
    yes: "getting unemployment",
    no: "not getting unemployment",
  },
  receives_free_school_meals: {
    yes: "a child in your home getting free school meals",
    no: "no child in your home getting free school meals",
  },
  is_pregnant: {
    yes: "someone in your household being pregnant",
    no: "nobody in your household being pregnant",
  },
  has_children: {
    yes: "a child under 18 in your household",
    no: "no children under 18 in your household",
  },
  is_veteran: { yes: "being a veteran", no: "not being a veteran" },
  is_disabled: { yes: "having a disability", no: "not having a disability" },
  is_senior: { yes: "being 60 or older", no: "being under 60" },
  is_caregiver: {
    yes: "caring for someone else",
    no: "not caring for someone else",
  },
  is_rutgers_affiliated: {
    yes: "being a Rutgers student or employee",
    no: "not being a Rutgers student or employee",
  },
  has_utility_shutoff_notice: {
    yes: "having had a shut-off notice",
    no: "not having had a shut-off notice",
  },
  has_eviction_notice: {
    yes: "having had an eviction notice",
    no: "not having had an eviction notice",
  },
  filed_taxes_last_year: {
    yes: "having filed taxes last year",
    no: "not having filed taxes last year",
  },
};

/** The intake wording table for each field that carries a closed set of values. */
const CHOICES_BY_FIELD: Record<string, readonly Choice<string>[]> = {
  state: STATE_CHOICES,
  student_status: STUDENT_CHOICES,
  housing_situation: HOUSING_CHOICES,
  employment_status: EMPLOYMENT_CHOICES,
  needs: NEED_CHOICES,
};

/**
 * A choice label as it reads mid-sentence.
 *
 * The tables are written for a label above a control, so they start with a
 * capital. Lowercasing the first letter is right for all of them except the
 * handful phrased in the first person ("I own my home"), where the capital is
 * the pronoun.
 */
function midSentence(label: string): string {
  if (label.startsWith("I ") || label.startsWith("I'")) return label;
  return label.charAt(0).toLowerCase() + label.slice(1);
}

/** `a, b and c`, which is how a requirement set reads out loud. */
function joinOr(parts: readonly string[]): string {
  if (parts.length === 0) return "";
  if (parts.length === 1) return parts[0] as string;
  return `${parts.slice(0, -1).join(", ")} or ${parts[parts.length - 1] as string}`;
}

/**
 * Wording that only makes sense inside a requirement.
 *
 * The choice tables are written as options someone picks from, which is not
 * the same sentence. "None of these fit" has to become "something else", and
 * the first-person options ("I do not have housing right now") have to become
 * third-person, because here they describe a rule rather than the reader.
 */
const GENERIC_OVERRIDES: Record<string, string> = {
  OTHER: "something else",
};

const FIELD_VALUE_OVERRIDES: Record<string, Record<string, string>> = {
  housing_situation: {
    OWN_HOME: "owning your home",
    UNHOUSED: "not having housing",
  },
};

/** The display wording for one value of one field, or `null` if unknown to us. */
function valueLabel(field: string, value: string): string | null {
  if (field === "county") return `${value} County`;
  const override = FIELD_VALUE_OVERRIDES[field]?.[value] ?? GENERIC_OVERRIDES[value];
  if (override !== undefined) return override;
  const choices = CHOICES_BY_FIELD[field];
  if (choices === undefined) return null;
  const match = choices.find((choice) => choice.value === value);
  if (match === undefined) return null;
  // A place name is a proper noun and keeps its capital wherever it appears.
  return field === "state" ? match.label : midSentence(match.label);
}

/** One `;`-separated clause, rewritten, or `null` when it cannot be. */
function humaniseClause(clause: string): string | null {
  const trimmed = clause.trim();
  if (trimmed.length === 0) return null;

  const boolean = BOOLEAN_CLAUSE.exec(trimmed);
  if (boolean !== null) {
    const field = (boolean[1] as string).replace(/^flags\./, "");
    const requirement = FLAG_REQUIREMENTS[field];
    if (requirement === undefined) return null;
    return boolean[2] === "true" ? requirement.yes : requirement.no;
  }

  const set = SET_CLAUSE.exec(trimmed);
  if (set !== null) {
    const field = (set[1] as string).replace(/^flags\./, "");
    const values = (set[2] as string)
      .split(",")
      .map((value) => value.trim())
      .filter((value) => value.length > 0);
    const labels = values.map((value) => valueLabel(field, value));
    if (labels.some((label) => label === null)) return null;
    const list = joinOr(labels as string[]);
    // Where you live reads as a place, not as a field with a value.
    if (field === "state" || field === "county") return `living in ${list}`;
    const clause = FIELD_CLAUSES[field];
    return clause === undefined ? null : clause(list);
  }

  // Already written for a person - "income of at most $54,943 a year". Accept
  // it only if nothing machine-shaped survived in it.
  return MACHINE_TOKEN.test(trimmed) ? null : trimmed;
}

/**
 * `threshold_applied` as a sentence fragment to follow "Checked against", or
 * `null` when no part of it can be stated in plain language.
 */
export function describeThreshold(threshold: string | null | undefined): string | null {
  if (typeof threshold !== "string") return null;
  const clauses = threshold.split(";");
  const humanised: string[] = [];
  for (const clause of clauses) {
    const text = humaniseClause(clause);
    // One unreadable clause makes the whole line misleading, because the
    // requirement was the conjunction of all of them.
    if (text === null) return null;
    humanised.push(text);
  }
  if (humanised.length === 0) return null;
  return humanised.join(" and ");
}
