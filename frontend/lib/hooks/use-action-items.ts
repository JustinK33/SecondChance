"use client";

/**
 * Ticking things off the action plan.
 *
 * Completion is optimistic. Someone working through a checklist on a phone on a
 * bus deserves the checkbox to move the instant they tap it, and the request
 * that follows is not worth a spinner. If the request fails the item snaps back
 * and the caller is handed an error to show, so the UI never quietly disagrees
 * with the server.
 *
 * The hook owns a local copy of the item list, seeded from the plan and re-seeded
 * whenever the plan identity changes. It is the single source of truth for the
 * screen while it is mounted.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { patchActionItem } from "../api-client";
import { ApiError, toApiError } from "../errors";
import type { ActionItem, ActionPlan, SharedDocument } from "../types";

export interface ActionItemsResult {
  readonly items: ActionItem[];
  readonly sharedDocuments: SharedDocument[];
  readonly completedCount: number;
  readonly totalCount: number;
  /** 0..1. Defined as 1 for an empty plan, matching `ActionPlan.progress_fraction`. */
  readonly progressFraction: number;
  /** Flip one item. Resolves `true` when the server agreed. */
  setComplete(itemId: string, isComplete: boolean): Promise<boolean>;
  /** Convenience wrapper over {@link setComplete} for a checkbox. */
  toggle(itemId: string): Promise<boolean>;
  /** True while the given item has a request in flight. */
  isSaving(itemId: string): boolean;
  /** True while any item has a request in flight. */
  readonly isSavingAny: boolean;
  /** The most recent failure, already rolled back. */
  readonly error: ApiError | null;
  clearError(): void;
}

export function useActionItems(plan: ActionPlan | null | undefined): ActionItemsResult {
  const [items, setItems] = useState<ActionItem[]>(plan?.items ?? []);
  const [saving, setSaving] = useState<ReadonlySet<string>>(new Set());
  const [error, setError] = useState<ApiError | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // Re-seed on a genuinely different plan, or a different version of one. Using
  // the id and version rather than the array identity avoids throwing away
  // optimistic state every time the parent re-renders.
  //
  // This is React's documented "adjust state when a prop changes" pattern: the
  // comparison happens during render and the re-render is discarded immediately,
  // which is cheaper and flicker-free compared with doing it in an effect.
  const planKey = plan === null || plan === undefined ? null : `${plan.id}:${plan.version}`;
  const [seededKey, setSeededKey] = useState<string | null>(planKey);
  if (planKey !== seededKey) {
    setSeededKey(planKey);
    setItems(plan?.items ?? []);
    setSaving(new Set());
    setError(null);
  }

  const setComplete = useCallback(async (itemId: string, isComplete: boolean): Promise<boolean> => {
    let previous: ActionItem | null = null;
    setItems((current) =>
      current.map((item) => {
        if (item.id !== itemId) return item;
        previous = item;
        return {
          ...item,
          is_complete: isComplete,
          completed_at: isComplete ? new Date().toISOString() : null,
        };
      }),
    );
    if (previous === null) {
      // Nothing to toggle. Treat as a no-op rather than inventing a failure.
      return false;
    }

    setSaving((current) => new Set(current).add(itemId));
    setError(null);
    try {
      const confirmed = await patchActionItem(itemId, { is_complete: isComplete });
      if (mounted.current) {
        // Adopt the server's row: it owns `completed_at` and we only guessed it.
        setItems((current) => current.map((item) => (item.id === itemId ? confirmed : item)));
      }
      return true;
    } catch (caught) {
      const apiError = toApiError(caught);
      if (mounted.current) {
        const restore = previous as ActionItem;
        setItems((current) => current.map((item) => (item.id === itemId ? restore : item)));
        if (!apiError.isAborted) setError(apiError);
      }
      return false;
    } finally {
      if (mounted.current) {
        setSaving((current) => {
          const next = new Set(current);
          next.delete(itemId);
          return next;
        });
      }
    }
  }, []);

  const toggle = useCallback(
    (itemId: string): Promise<boolean> => {
      const item = items.find((candidate) => candidate.id === itemId);
      if (item === undefined) return Promise.resolve(false);
      return setComplete(itemId, !(item.is_complete ?? false));
    },
    [items, setComplete],
  );

  const isSaving = useCallback((itemId: string): boolean => saving.has(itemId), [saving]);
  const clearError = useCallback((): void => setError(null), []);

  return useMemo<ActionItemsResult>(() => {
    const completedCount = items.filter((item) => item.is_complete === true).length;
    return {
      items,
      sharedDocuments: plan?.shared_documents ?? [],
      completedCount,
      totalCount: items.length,
      progressFraction: items.length === 0 ? 1 : completedCount / items.length,
      setComplete,
      toggle,
      isSaving,
      isSavingAny: saving.size > 0,
      error,
      clearError,
    };
  }, [items, plan, setComplete, toggle, isSaving, saving, error, clearError]);
}
