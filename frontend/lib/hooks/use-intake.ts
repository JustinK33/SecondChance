"use client";

/**
 * Profile extraction without matching.
 *
 * Used by the onboarding screen to show "here is what we understood" before
 * committing to a full run. The narrative is never logged.
 */

import { useCallback } from "react";

import { createIntake } from "../api-client";
import type { IntakeResponse, ProfileDraft } from "../types";
import { useAction } from "./use-async";
import type { ApiError } from "../errors";

export interface IntakeResult {
  /** Send the narrative. Resolves with the response, or `null` on failure. */
  submit(narrative: string, prefilled?: ProfileDraft | null): Promise<IntakeResponse | null>;
  readonly data: IntakeResponse | null;
  readonly isPending: boolean;
  readonly error: ApiError | null;
  reset(): void;
}

export function useIntake(): IntakeResult {
  const action = useAction((narrative: string, prefilled?: ProfileDraft | null) =>
    createIntake({
      narrative,
      ...(prefilled === undefined || prefilled === null ? {} : { prefilled }),
    }),
  );

  const submit = useCallback(
    (narrative: string, prefilled?: ProfileDraft | null) => action.run(narrative, prefilled),
    [action],
  );

  return {
    submit,
    data: action.data,
    isPending: action.isPending,
    error: action.error,
    reset: action.reset,
  };
}
