"use client";

/**
 * What the system currently understands about the user, and correcting it.
 *
 * There is no profile endpoint. The profile the user should see is the one on
 * their most recent run, which is what `WorkflowRunDetail.profile` carries, and
 * a correction is `POST /api/v1/matches` - a new run over the edited fields.
 * Modelling it any other way would invent an API that does not exist.
 */

import { useCallback, useMemo } from "react";

import { getWorkflow, listWorkflows } from "../api-client";
import type { ApiError } from "../errors";
import type { ProfileDraft, UserProfile, WorkflowRunSummary } from "../types";
import { useQuery } from "./use-async";
import { useReevaluate } from "./use-workflow";

export interface ProfileResult {
  readonly profile: UserProfile | null;
  /** The run the profile was read from, which is what a correction is applied to. */
  readonly sourceRunId: string | null;
  /** Fields the backend recorded as never stated. Never treated as disqualifying. */
  readonly unknownFields: string[];
  readonly isLoading: boolean;
  readonly error: ApiError | null;
  refetch(): void;
  /**
   * Submit corrected fields. Resolves with the id of the new run to follow, or
   * `null` if it failed.
   */
  saveChanges(changes: ProfileDraft): Promise<string | null>;
  readonly isSaving: boolean;
  readonly saveError: ApiError | null;
}

/**
 * The profile from a given run, or from the latest run when `runId` is omitted.
 */
export function useProfile(runId?: string | null): ProfileResult {
  const explicitId = typeof runId === "string" && runId.length > 0 ? runId : null;

  const result = useQuery(
    async (signal) => {
      if (explicitId !== null) {
        const run = await getWorkflow(explicitId, { signal });
        return {
          runId: run.id,
          profile: run.profile ?? null,
          unknownFields: run.profile?.unknown_fields ?? [],
        };
      }
      const page = await listWorkflows({ limit: 1, offset: 0 }, { signal });
      const latest: WorkflowRunSummary | undefined = (page.items ?? [])[0];
      if (latest === undefined) {
        return { runId: null, profile: null, unknownFields: [] as string[] };
      }
      const run = await getWorkflow(latest.id, { signal });
      return {
        runId: run.id,
        profile: run.profile ?? null,
        unknownFields: run.profile?.unknown_fields ?? [],
      };
    },
    [explicitId],
  );

  const reevaluation = useReevaluate();
  const sourceRunId = result.data?.runId ?? null;

  const saveChanges = useCallback(
    async (changes: ProfileDraft): Promise<string | null> => {
      if (sourceRunId === null) return null;
      const response = await reevaluation.reevaluate(sourceRunId, changes);
      return response?.run_id ?? null;
    },
    [sourceRunId, reevaluation],
  );

  return useMemo<ProfileResult>(
    () => ({
      profile: result.data?.profile ?? null,
      sourceRunId,
      unknownFields: result.data?.unknownFields ?? [],
      isLoading: result.isLoading,
      error: result.error,
      refetch: result.refetch,
      saveChanges,
      isSaving: reevaluation.isPending,
      saveError: reevaluation.error,
    }),
    [result, sourceRunId, saveChanges, reevaluation.isPending, reevaluation.error],
  );
}
