"use client";

/**
 * Starting a run, and following it to completion.
 *
 * The backend answers `POST /workflows` with 202 and a durable `run_id`; no HTTP
 * request is ever held open for a model-backed workflow. So the client polls.
 *
 * Polling policy, and why:
 *
 * - First poll after ~1.2s. Fast enough that a cheap run feels immediate.
 * - Gentle geometric backoff to a 6s ceiling. A run that takes a minute should
 *   not cost sixty requests, and the progress message changes slowly anyway.
 * - Stop on a terminal status, and stop on `AWAITING_INPUT`: nothing will change
 *   until the user answers, so continuing to poll is pure waste.
 * - Pause while the tab is hidden, and poll immediately when it comes back. A
 *   backgrounded tab should not keep a connection warm for nothing.
 * - A failed poll does not surface an error until it has failed repeatedly. One
 *   dropped request during a long run is noise, not news.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  cancelWorkflow,
  createMatches,
  createWorkflow,
  listWorkflows,
  newIdempotencyKey,
  pollWorkflow,
  resumeWorkflow,
} from "../api-client";
import { ApiError, toApiError } from "../errors";
import type {
  ClarificationAnswer,
  ClarificationQuestion,
  MatchRequest,
  ProfileDraft,
  ProgramMatch,
  UserFacingError,
  WorkflowCreateResponse,
  WorkflowPhase,
  WorkflowRun,
  WorkflowRunSummary,
  WorkflowStatus,
} from "../types";
import { isTerminalWorkflowStatus } from "../types";
import { useAction, useQuery } from "./use-async";
import type { QueryResult } from "./use-async";

const FIRST_INTERVAL_MS = 1_200;
const MAX_INTERVAL_MS = 6_000;
const BACKOFF_FACTOR = 1.35;

/** Consecutive poll failures tolerated before the error reaches the screen. */
const POLL_FAILURE_GRACE = 3;

// ---------------------------------------------------------------------------
// starting a run
// ---------------------------------------------------------------------------

export interface StartWorkflowResult {
  /**
   * Start a run from a narrative.
   *
   * The idempotency key is minted once per hook instance and reused, so a
   * resubmission after a network failure returns the original run with
   * `deduplicated: true` rather than creating a second one. Call `reset` to
   * deliberately start a different run.
   */
  start(narrative: string, prefilled?: ProfileDraft | null): Promise<WorkflowCreateResponse | null>;
  readonly response: WorkflowCreateResponse | null;
  readonly runId: string | null;
  readonly isPending: boolean;
  readonly error: ApiError | null;
  reset(): void;
}

export function useStartWorkflow(): StartWorkflowResult {
  const keyRef = useRef<string | null>(null);
  const action = useAction((narrative: string, prefilled?: ProfileDraft | null) => {
    keyRef.current ??= newIdempotencyKey();
    return createWorkflow(
      {
        narrative,
        ...(prefilled === undefined || prefilled === null ? {} : { prefilled }),
      },
      keyRef.current,
    );
  });

  const reset = useCallback((): void => {
    keyRef.current = null;
    action.reset();
  }, [action]);

  return {
    start: (narrative, prefilled) => action.run(narrative, prefilled),
    response: action.data,
    runId: action.data?.run_id ?? null,
    isPending: action.isPending,
    error: action.error,
    reset,
  };
}

// ---------------------------------------------------------------------------
// following a run
// ---------------------------------------------------------------------------

export interface WorkflowResult {
  readonly run: WorkflowRun | null;
  readonly status: WorkflowStatus | null;
  readonly phase: WorkflowPhase | null;
  /** The sentence to put on screen. Written by the backend, safe to render verbatim. */
  readonly progressMessage: string;
  /** 0..1 for a determinate progress bar. */
  readonly progressFraction: number;
  readonly matches: ProgramMatch[];
  readonly questions: ClarificationQuestion[];
  readonly limitations: string[];
  /** The run's own failure, distinct from a transport failure reaching it. */
  readonly runError: UserFacingError | null;
  readonly isLoading: boolean;
  readonly isPolling: boolean;
  readonly isTerminal: boolean;
  readonly isAwaitingInput: boolean;
  readonly isComplete: boolean;
  /** A transport or HTTP failure, after the grace period. */
  readonly error: ApiError | null;
  /** Poll once, immediately. */
  refetch(): void;
  /** Answer the clarification questions and let the run continue. */
  submitAnswers(answers: ClarificationAnswer[]): Promise<WorkflowRun | null>;
  readonly isSubmittingAnswers: boolean;
  readonly submitError: ApiError | null;
  /** Ask the backend to stop the run. */
  cancel(): Promise<WorkflowRun | null>;
  readonly isCancelling: boolean;
}

/**
 * Follow one run.
 *
 * Pass `null` before a run exists; the hook stays idle rather than erroring.
 */
export function useWorkflow(runId: string | null | undefined): WorkflowResult {
  const id = typeof runId === "string" && runId.length > 0 ? runId : null;

  const [run, setRun] = useState<WorkflowRun | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(id !== null);
  const [isPolling, setIsPolling] = useState<boolean>(false);

  const failureCount = useRef(0);
  const intervalRef = useRef(FIRST_INTERVAL_MS);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  // The poll loop reads this to decide whether to schedule another tick, so it
  // must be a ref: a stale closure over state would poll a finished run forever.
  const stopRef = useRef(false);

  const clearTimer = useCallback((): void => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const tick = useCallback(
    async function tick(targetId: string): Promise<void> {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      try {
        const next = await pollWorkflow(targetId, { signal: controller.signal });
        if (!mounted.current) return;
        failureCount.current = 0;
        setRun(next);
        setError(null);
        setIsLoading(false);

        const settled = isTerminalWorkflowStatus(next.status) || next.status === "AWAITING_INPUT";
        if (settled) {
          stopRef.current = true;
          setIsPolling(false);
          clearTimer();
          return;
        }
        intervalRef.current = Math.min(
          Math.round(intervalRef.current * BACKOFF_FACTOR),
          MAX_INTERVAL_MS,
        );
      } catch (caught) {
        const apiError = toApiError(caught);
        if (!mounted.current || apiError.isAborted) return;
        failureCount.current += 1;
        setIsLoading(false);
        // A non-retryable answer from the server is a decision: a deleted run,
        // or a run that is not ours. Stop rather than hammering it.
        if (!apiError.isRetryable) {
          stopRef.current = true;
          setIsPolling(false);
          setError(apiError);
          clearTimer();
          return;
        }
        if (failureCount.current >= POLL_FAILURE_GRACE) setError(apiError);
        intervalRef.current = Math.min(
          Math.round(intervalRef.current * BACKOFF_FACTOR),
          MAX_INTERVAL_MS,
        );
      }

      if (!mounted.current || stopRef.current) return;
      clearTimer();
      if (typeof document !== "undefined" && document.visibilityState === "hidden") {
        // Hidden tab: the visibility listener restarts us on return.
        return;
      }
      timerRef.current = setTimeout(() => void tick(targetId), intervalRef.current);
    },
    [clearTimer],
  );

  // Mount / unmount bookkeeping, separate from the run-id effect so an id change
  // does not flip `mounted` off and on.
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    clearTimer();
    abortRef.current?.abort();
    failureCount.current = 0;
    intervalRef.current = FIRST_INTERVAL_MS;
    stopRef.current = false;

    if (id === null) {
      setRun(null);
      setError(null);
      setIsLoading(false);
      setIsPolling(false);
      return;
    }

    setRun(null);
    setError(null);
    setIsLoading(true);
    setIsPolling(true);
    void tick(id);

    return () => {
      clearTimer();
      abortRef.current?.abort();
    };
  }, [id, tick, clearTimer]);

  // Resume promptly when the tab comes back, rather than waiting out a timer
  // that browsers throttle while hidden.
  useEffect(() => {
    if (id === null || typeof document === "undefined") return;
    const onVisible = (): void => {
      if (document.visibilityState !== "visible") return;
      if (stopRef.current || timerRef.current !== null) return;
      intervalRef.current = FIRST_INTERVAL_MS;
      void tick(id);
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [id, tick]);

  const refetch = useCallback((): void => {
    if (id === null) return;
    clearTimer();
    stopRef.current = false;
    intervalRef.current = FIRST_INTERVAL_MS;
    setIsPolling(true);
    void tick(id);
  }, [id, tick, clearTimer]);

  // ------------------------------------------------------------------ resume
  const resumeAction = useAction((answers: ClarificationAnswer[]) =>
    resumeWorkflow(id as string, { answers }),
  );
  const submitAnswers = useCallback(
    async (answers: ClarificationAnswer[]): Promise<WorkflowRun | null> => {
      if (id === null) return null;
      const next = await resumeAction.run(answers);
      if (next === null) return null;
      setRun(next);
      // The run is moving again, so restart the loop from the fast interval.
      if (!isTerminalWorkflowStatus(next.status) && next.status !== "AWAITING_INPUT") {
        stopRef.current = false;
        intervalRef.current = FIRST_INTERVAL_MS;
        setIsPolling(true);
        clearTimer();
        timerRef.current = setTimeout(() => void tick(id), FIRST_INTERVAL_MS);
      }
      return next;
    },
    [id, resumeAction, tick, clearTimer],
  );

  // ------------------------------------------------------------------ cancel
  const cancelAction = useAction(() => cancelWorkflow(id as string));
  const cancel = useCallback(async (): Promise<WorkflowRun | null> => {
    if (id === null) return null;
    const next = await cancelAction.run();
    if (next !== null) {
      setRun(next);
      stopRef.current = true;
      setIsPolling(false);
      clearTimer();
    }
    return next;
  }, [id, cancelAction, clearTimer]);

  return useMemo<WorkflowResult>(() => {
    const status = run?.status ?? null;
    return {
      run,
      status,
      phase: run?.phase ?? null,
      progressMessage: run?.progress_message ?? "",
      progressFraction: run?.progress_fraction ?? 0,
      matches: run?.matches ?? [],
      questions: run?.clarification_questions ?? [],
      limitations: run?.limitations ?? [],
      runError: run?.error ?? null,
      isLoading,
      isPolling,
      isTerminal: status !== null && isTerminalWorkflowStatus(status),
      isAwaitingInput: status === "AWAITING_INPUT",
      isComplete: status === "COMPLETED",
      error,
      refetch,
      submitAnswers,
      isSubmittingAnswers: resumeAction.isPending,
      submitError: resumeAction.error,
      cancel,
      isCancelling: cancelAction.isPending,
    };
  }, [
    run,
    isLoading,
    isPolling,
    error,
    refetch,
    submitAnswers,
    resumeAction.isPending,
    resumeAction.error,
    cancel,
    cancelAction.isPending,
  ]);
}

// ---------------------------------------------------------------------------
// run history
// ---------------------------------------------------------------------------

export interface WorkflowsResult extends QueryResult<{
  items: WorkflowRunSummary[];
  total: number;
}> {
  readonly runs: WorkflowRunSummary[];
  readonly total: number;
  readonly isEmpty: boolean;
}

/** The signed-in user's previous runs, newest first (the backend orders them). */
export function useWorkflows(query?: { limit?: number; offset?: number }): WorkflowsResult {
  const key = JSON.stringify(query ?? {});
  const result = useQuery(
    (signal) =>
      listWorkflows(query, { signal }).then((page) => ({
        items: page.items ?? [],
        total: page.total,
      })),
    [key],
  );
  return useMemo(
    () => ({
      ...result,
      runs: result.data?.items ?? [],
      total: result.data?.total ?? 0,
      isEmpty: result.data !== null && result.data.items.length === 0,
    }),
    [result],
  );
}

// ---------------------------------------------------------------------------
// re-matching after a profile edit
// ---------------------------------------------------------------------------

export interface ReevaluateResult {
  /** Re-run matching with corrected profile fields. Returns the new run to poll. */
  reevaluate(runId: string, changes: ProfileDraft): Promise<WorkflowCreateResponse | null>;
  readonly response: WorkflowCreateResponse | null;
  readonly runId: string | null;
  readonly isPending: boolean;
  readonly error: ApiError | null;
  reset(): void;
}

export function useReevaluate(): ReevaluateResult {
  const action = useAction((runId: string, changes: ProfileDraft) => {
    const body: MatchRequest = { run_id: runId, changes };
    return createMatches(body, newIdempotencyKey());
  });
  return {
    reevaluate: (runId, changes) => action.run(runId, changes),
    response: action.data,
    runId: action.data?.run_id ?? null,
    isPending: action.isPending,
    error: action.error,
    reset: action.reset,
  };
}
