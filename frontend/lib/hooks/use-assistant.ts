"use client";

/**
 * The grounded question-and-answer assistant.
 *
 * The transcript is local to the component. The backend caps history at eight
 * messages, so we send only the most recent window and keep the rest for display
 * - truncating the request instead of letting a 422 reach the user.
 *
 * A failed question is never retried automatically: a retry is a second model
 * call for one question.
 */

import { useCallback, useMemo, useRef, useState } from "react";

import { sendChat } from "../api-client";
import { ApiError, toApiError } from "../errors";
import type { ChatMessage, ChatResponse } from "../types";

/** The contract's `ChatRequest.history` maximum. */
const MAX_HISTORY = 8;

export interface AssistantTurn {
  /** Stable key for rendering. Not persisted anywhere. */
  readonly id: string;
  readonly role: "user" | "assistant";
  readonly text: string;
  /** Present on an assistant turn: the full grounded response with its citations. */
  readonly response: ChatResponse | null;
}

export interface AssistantOptions {
  /** Scope retrieval to one program's sources. */
  readonly programId?: string | null;
  /** Scope the answer to a run. Ownership is checked server-side. */
  readonly runId?: string | null;
}

export interface AssistantResult {
  readonly turns: AssistantTurn[];
  /** Ask a question. Resolves with the grounded response, or `null` on failure. */
  ask(message: string): Promise<ChatResponse | null>;
  readonly isPending: boolean;
  readonly error: ApiError | null;
  readonly lastResponse: ChatResponse | null;
  /** Empty the transcript. */
  reset(): void;
  clearError(): void;
}

function newTurnId(): string {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function useAssistant(options?: AssistantOptions): AssistantResult {
  const [turns, setTurns] = useState<AssistantTurn[]>([]);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const mounted = useRef(true);

  const programId = options?.programId ?? null;
  const runId = options?.runId ?? null;

  const ask = useCallback(
    async (message: string): Promise<ChatResponse | null> => {
      const text = message.trim();
      if (text.length === 0) return null;

      // Build the wire history from what is already on screen, before adding
      // this question, and keep only the window the backend accepts.
      const history: ChatMessage[] = turns
        .slice(-MAX_HISTORY)
        .map((turn) => ({ role: turn.role, content: turn.text }));

      setTurns((current) => [...current, { id: newTurnId(), role: "user", text, response: null }]);
      setIsPending(true);
      setError(null);

      try {
        const response = await sendChat({
          message: text,
          ...(programId === null ? {} : { program_id: programId }),
          ...(runId === null ? {} : { run_id: runId }),
          ...(history.length === 0 ? {} : { history }),
        });
        if (mounted.current) {
          setTurns((current) => [
            ...current,
            { id: newTurnId(), role: "assistant", text: response.answer, response },
          ]);
          setIsPending(false);
        }
        return response;
      } catch (caught) {
        const apiError = toApiError(caught);
        if (mounted.current) {
          setIsPending(false);
          if (!apiError.isAborted) setError(apiError);
        }
        return null;
      }
    },
    [turns, programId, runId],
  );

  const reset = useCallback((): void => {
    setTurns([]);
    setError(null);
    setIsPending(false);
  }, []);

  const clearError = useCallback((): void => setError(null), []);

  return useMemo<AssistantResult>(() => {
    const lastAssistant = [...turns].reverse().find((turn) => turn.role === "assistant");
    return {
      turns,
      ask,
      isPending,
      error,
      lastResponse: lastAssistant?.response ?? null,
      reset,
      clearError,
    };
  }, [turns, ask, isPending, error, reset, clearError]);
}
