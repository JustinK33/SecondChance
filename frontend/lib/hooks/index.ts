/**
 * The data layer's public surface.
 *
 * Screens and feature components import from `@/lib/hooks` and nothing else in
 * `lib/`. Keeping one entry point is what makes "no component fetches" an
 * enforceable rule rather than a convention.
 */

export { useQuery, useAction } from "./use-async";
export type { QueryResult, QueryState, ActionResult } from "./use-async";

export { usePrograms, useProgram } from "./use-programs";
export type { ProgramsResult, ProgramResult } from "./use-programs";

export { useIntake } from "./use-intake";
export type { IntakeResult } from "./use-intake";

export { useStartWorkflow, useWorkflow, useWorkflows, useReevaluate } from "./use-workflow";
export type {
  StartWorkflowResult,
  WorkflowResult,
  WorkflowsResult,
  ReevaluateResult,
} from "./use-workflow";

export { usePlan, usePlans } from "./use-plan";
export type { PlanResult, PlansResult } from "./use-plan";

export { useActionItems } from "./use-action-items";
export type { ActionItemsResult } from "./use-action-items";

export { useAssistant } from "./use-assistant";
export type { AssistantResult, AssistantTurn, AssistantOptions } from "./use-assistant";

export { useProfile } from "./use-profile";
export type { ProfileResult } from "./use-profile";

export { useAccountDeletion } from "./use-account-deletion";
export type { AccountDeletionResult } from "./use-account-deletion";

export { useDraft } from "./use-draft";
export type { DraftResult } from "./use-draft";
