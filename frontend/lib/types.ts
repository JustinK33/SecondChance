/**
 * Stable, hand-written names for the backend contract.
 *
 * `lib/api-types.gen.ts` is regenerated from the backend's OpenAPI document by
 * `npm run types`, so its shape is not ours to depend on directly. Screens and
 * components import from this file instead; if the generator's output moves, the
 * repair happens here once rather than in thirty component files.
 *
 * Rule: nothing outside `lib/` imports `api-types.gen`.
 */

import type { components, operations } from "./api-types.gen";

type Schemas = components["schemas"];

// ---------------------------------------------------------------------------
// enums - the member values are the wire values
// ---------------------------------------------------------------------------

/** The only three things we are allowed to say about a program. Never "eligible". */
export type MatchStatus = Schemas["MatchStatus"];
export type Trilean = Schemas["Trilean"];
export type WorkflowStatus = Schemas["WorkflowStatus"];
export type WorkflowPhase = Schemas["WorkflowPhase"];
export type ProgramCategory = Schemas["ProgramCategory"];
export type NeedCategory = Schemas["NeedCategory"];
export type StudentStatus = Schemas["StudentStatus"];
export type EmploymentStatus = Schemas["EmploymentStatus"];
export type HousingSituation = Schemas["HousingSituation"];
export type IncomePeriod = Schemas["IncomePeriod"];
export type VerificationStatus = Schemas["VerificationStatus"];
export type SourceFreshness = Schemas["SourceFreshness"];
export type ActionItemKind = Schemas["ActionItemKind"];
export type ClarificationInputKind = Schemas["ClarificationInputKind"];
export type ConfidenceLevel = Schemas["ConfidenceLevel"];
export type ProfileFlag = Schemas["ProfileFlag"];
export type ConditionKind = Schemas["ConditionKind"];
export type ErrorCode = Schemas["ErrorCode"];

/** Every `MatchStatus` member, in the order we want to show them. */
export const MATCH_STATUSES: readonly MatchStatus[] = [
  "POTENTIAL_MATCH",
  "MORE_INFO_NEEDED",
  "UNLIKELY_MATCH",
] as const;

/** Statuses after which polling must stop because nothing more will change. */
export const TERMINAL_WORKFLOW_STATUSES: readonly WorkflowStatus[] = [
  "COMPLETED",
  "FAILED",
  "CANCELLED",
] as const;

export function isTerminalWorkflowStatus(status: WorkflowStatus): boolean {
  return TERMINAL_WORKFLOW_STATUSES.includes(status);
}

/** A run parked on a clarification question. Not terminal: answering it resumes the run. */
export function isAwaitingInput(status: WorkflowStatus): boolean {
  return status === "AWAITING_INPUT";
}

// ---------------------------------------------------------------------------
// profile and intake
// ---------------------------------------------------------------------------

export type UserProfile = Schemas["UserProfile"];
export type ProfileDraft = Schemas["ProfileDraft"];
export type HouseholdComposition = Schemas["HouseholdComposition"];
export type IntakeRequest = Schemas["IntakeRequest"];
export type IntakeResponse = Schemas["IntakeResponse"];
export type ClarificationQuestion = Schemas["ClarificationQuestion"];
export type ClarificationOption = Schemas["ClarificationOption"];
export type ClarificationAnswer = Schemas["ClarificationAnswer"];
export type RangeValue = Schemas["RangeValue"];
/** The union a clarification answer may carry. `null` means "answered with nothing". */
export type ClarificationAnswerValue = ClarificationAnswer["value"];

// ---------------------------------------------------------------------------
// programs
// ---------------------------------------------------------------------------

/** The list-shaped program record. What a card renders. */
export type Program = Schemas["ProgramSummary"];
/** Alias kept for readers who expect the backend's own spelling. */
export type ProgramSummary = Schemas["ProgramSummary"];
/** A program plus its sources, rendered requirements and rule version. */
export type ProgramDetail = Schemas["ProgramDetail"];
export type AssistanceProgram = Schemas["AssistanceProgram"];
export type ProgramSource = Schemas["ProgramSource"];
export type GeoScope = Schemas["GeoScope"];
export type RequirementBullet = Schemas["RequirementBullet"];
export type ProgramPage = Schemas["Page_ProgramSummary_"];

// ---------------------------------------------------------------------------
// matching and eligibility
// ---------------------------------------------------------------------------

export type ProgramMatch = Schemas["ProgramMatch"];
export type MatchExplanation = Schemas["MatchExplanation"];
export type EligibilityEvaluation = Schemas["EligibilityEvaluation"];
export type ConditionEvidence = Schemas["ConditionEvidence"];

// ---------------------------------------------------------------------------
// workflow runs
// ---------------------------------------------------------------------------

export type WorkflowRun = Schemas["WorkflowRunDetail"];
/** Alias kept for readers who expect the backend's own spelling. */
export type WorkflowRunDetail = Schemas["WorkflowRunDetail"];
export type WorkflowRunSummary = Schemas["WorkflowRunSummary"];
export type WorkflowCreateRequest = Schemas["WorkflowCreateRequest"];
export type WorkflowCreateResponse = Schemas["WorkflowCreateResponse"];
export type WorkflowResumeRequest = Schemas["WorkflowResumeRequest"];
export type MatchRequest = Schemas["MatchRequest"];
export type UserFacingError = Schemas["UserFacingError"];
export type WorkflowRunPage = Schemas["Page_WorkflowRunSummary_"];

// ---------------------------------------------------------------------------
// action plan
// ---------------------------------------------------------------------------

export type ActionPlan = Schemas["ActionPlan"];
export type ActionItem = Schemas["ActionItem"];
export type ActionItemPatch = Schemas["ActionItemPatch"];
export type SharedDocument = Schemas["SharedDocument"];
export type ActionPlanPage = Schemas["Page_ActionPlan_"];

// ---------------------------------------------------------------------------
// assistant
// ---------------------------------------------------------------------------

export type ChatRequest = Schemas["ChatRequest"];
export type ChatResponse = Schemas["ChatResponse"];
export type ChatMessage = Schemas["ChatMessage"];
export type Citation = Schemas["Citation"];
export type FactStatement = Schemas["FactStatement"];

// ---------------------------------------------------------------------------
// system
// ---------------------------------------------------------------------------

export type HealthResponse = Schemas["HealthResponse"];
export type ReadinessResponse = Schemas["ReadinessResponse"];
export type DependencyStatus = Schemas["DependencyStatus"];
export type AccountDeletionReceipt = Schemas["AccountDeletionReceipt"];

// ---------------------------------------------------------------------------
// error envelope
// ---------------------------------------------------------------------------

/** The inner object of every non-2xx body. `message` is safe to render verbatim. */
export type ApiErrorDetail = Schemas["ErrorDetail"];
export type ApiErrorResponse = Schemas["ErrorResponse"];

// ---------------------------------------------------------------------------
// query parameters, lifted off the generated operations so they cannot drift
// ---------------------------------------------------------------------------

export type ListProgramsQuery = NonNullable<
  operations["list_programs_api_v1_programs_get"]["parameters"]["query"]
>;
export type PaginationQuery = {
  limit?: number;
  offset?: number;
};

/** One page of anything, matching the backend's `Page[T]`. */
export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}
