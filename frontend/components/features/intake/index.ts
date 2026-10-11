/**
 * The intake half's public surface: landing, sign-in, onboarding, clarification.
 *
 * Screens import from here. The one export that is not a screen component is
 * `ClarificationFlow`, which the run screen should also render while a run sits
 * at `AWAITING_INPUT` so there is only one implementation of that interaction.
 */

export { LandingHero } from "./landing-hero";
export { HowItWorks } from "./how-it-works";
export { PreliminaryNotice } from "./preliminary-notice";
export { WhatWeCollect } from "./what-we-collect";
export { SignInForm } from "./sign-in-form";
export { OnboardingForm } from "./onboarding-form";
export { NarrativeField } from "./narrative-field";
export { StructuredHelpers } from "./structured-helpers";
export { IncomeRangeField } from "./income-range-field";
export { NeedsPicker } from "./needs-picker";
export { UnderstoodSummary } from "./understood-summary";
export { ClarificationFlow } from "./clarification-flow";
export { ClarificationQuestionCard } from "./clarification-question-card";

export {
  EMPTY_ANSWERS,
  HOUSEHOLD_SIZE_MAX,
  HOUSEHOLD_SIZE_MIN,
  fromProfile,
  toProfileDraft,
} from "./structured-answers";
export type { StructuredAnswers } from "./structured-answers";

export {
  ANNUAL_INCOME_BANDS,
  EMPLOYMENT_CHOICES,
  HOUSING_CHOICES,
  MONTHLY_INCOME_BANDS,
  NARRATIVE_EXAMPLE,
  NARRATIVE_MAX_LENGTH,
  NARRATIVE_MIN_LENGTH,
  NEED_CHOICES,
  NJ_COUNTIES,
  STATE_CHOICES,
  STUDENT_CHOICES,
  findIncomeBand,
  incomeBandsFor,
} from "./options";
export type { Choice, IncomeBand } from "./options";
