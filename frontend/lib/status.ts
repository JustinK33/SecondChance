/**
 * The words we use for a match status, in one place.
 *
 * The rule engine decides the status; this file decides only how it is worded.
 * Two product rules are enforced here rather than trusted to each component:
 *
 * 1. We never say "eligible" or "ineligible". A deterministic engine run against
 *    a partial profile cannot make an eligibility decision, and implying it can
 *    would be a false promise to someone deciding whether to spend an afternoon
 *    on an application.
 * 2. The wording is non-judgmental. "Probably not a fit" describes the match;
 *    "you do not qualify" would describe the person.
 */

import type { MatchStatus, Trilean, VerificationStatus } from "./types";

export interface StatusPresentation {
  /** The short label on a badge. Always rendered as text, never colour alone. */
  readonly label: string;
  /** One sentence explaining what the label means. */
  readonly description: string;
  /** What the user can usefully do next. */
  readonly nextStep: string;
  /** Design-token name, used by `StatusBadge`. Never the only carrier of meaning. */
  readonly tone: "positive" | "neutral" | "muted";
  /** Sort weight, lowest first, so the most actionable matches lead a list. */
  readonly order: number;
}

const PRESENTATION: Record<MatchStatus, StatusPresentation> = {
  POTENTIAL_MATCH: {
    label: "Worth exploring",
    description:
      "Based on what you told us, you appear to meet the requirements we were able to check.",
    nextStep: "Read what it covers, then apply through the official link.",
    tone: "positive",
    order: 0,
  },
  MORE_INFO_NEEDED: {
    label: "Need a bit more info",
    description:
      "We could not check every requirement with the information we have. Nothing here rules you out.",
    nextStep: "Answer a couple of short questions and we will check again.",
    tone: "neutral",
    order: 1,
  },
  UNLIKELY_MATCH: {
    label: "Probably not a fit",
    description:
      "One of the requirements we could check does not appear to match your situation right now.",
    nextStep: "Situations change. It is worth revisiting if yours does.",
    tone: "muted",
    order: 2,
  },
};

export function statusPresentation(status: MatchStatus): StatusPresentation {
  return PRESENTATION[status];
}

export function statusLabel(status: MatchStatus): string {
  return PRESENTATION[status].label;
}

/** Sort comparator putting the most actionable statuses first. */
export function compareByStatus(a: MatchStatus, b: MatchStatus): number {
  return PRESENTATION[a].order - PRESENTATION[b].order;
}

/** Plain-language wording for a single requirement's outcome. */
export function trileanLabel(outcome: Trilean): string {
  switch (outcome) {
    case "TRUE":
      return "Looks met";
    case "FALSE":
      return "Does not look met";
    case "UNKNOWN":
      return "Not enough information";
  }
}

export interface VerificationPresentation {
  readonly label: string;
  readonly description: string;
  /** True when this program's details may be used in eligibility wording. */
  readonly supportsClaims: boolean;
}

/**
 * How much we are willing to say about a program's own data.
 *
 * `UNVERIFIED` and `NEEDS_REVIEW` programs are shown - hiding a real program
 * helps nobody - but they are labelled, and they are excluded from eligibility
 * claims by the backend.
 */
export function verificationPresentation(status: VerificationStatus): VerificationPresentation {
  switch (status) {
    case "VERIFIED":
      return {
        label: "Checked against the official source",
        description: "We read this program's details from its official page and recorded when.",
        supportsClaims: true,
      };
    case "NEEDS_REVIEW":
      return {
        label: "Being re-checked",
        description:
          "This program's details are being re-checked, so we are not using them to assess a match.",
        supportsClaims: false,
      };
    case "UNVERIFIED":
      return {
        label: "Not yet verified",
        description:
          "We have not been able to confirm these details against an official source, so treat them as a starting point and check the official page.",
        supportsClaims: false,
      };
  }
}
