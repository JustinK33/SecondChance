"use client";

/**
 * The two shapes every other hook in this directory is built from.
 *
 * `useQuery` is for reading something and showing it; `useAction` is for doing
 * something when the user asks. Both of them exist so that no screen has to
 * hand-roll the "is it loading, did it fail, is this response stale" triangle,
 * and so that an unmounted component never calls `setState`.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, toApiError } from "../errors";

export interface QueryState<T> {
  readonly data: T | null;
  readonly isLoading: boolean;
  /** True while a background refetch is in flight and `data` is still the old value. */
  readonly isRefreshing: boolean;
  readonly error: ApiError | null;
}

export interface QueryResult<T> extends QueryState<T> {
  /** Fetch again, keeping the current `data` visible while it runs. */
  refetch(): void;
}

/**
 * Run `fetcher` whenever `deps` change, with abort on unmount and on re-run.
 *
 * `enabled: false` is the "no id yet" case and must not be modelled as an error.
 */
export function useQuery<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  options?: { readonly enabled?: boolean },
): QueryResult<T> {
  const enabled = options?.enabled ?? true;
  const [state, setState] = useState<QueryState<T>>({
    data: null,
    isLoading: enabled,
    isRefreshing: false,
    error: null,
  });
  const [nonce, setNonce] = useState(0);

  // The fetcher is usually an inline closure, so it is a new function on every
  // render and cannot go in the dependency array without refetching forever.
  // Mirroring it into a ref in its own effect - declared before the effect that
  // reads it, so it is updated first - keeps the latest closure available
  // without making it a dependency.
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  }, [fetcher]);

  useEffect(() => {
    // Disabled is not a state to store: it is derived below from `enabled`, so
    // there is nothing to reset here.
    if (!enabled) return;
    const controller = new AbortController();
    let active = true;

    // Entering the loading state is the whole point of starting a fetch, so the
    // one extra render this costs is the correct behaviour rather than a
    // cascade to avoid. Deriving it instead would mean hashing an arbitrary
    // dependency array during render, which is worse on every axis.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState((previous) => ({
      data: previous.data,
      isLoading: previous.data === null,
      isRefreshing: previous.data !== null,
      error: null,
    }));

    void fetcherRef
      .current(controller.signal)
      .then((data) => {
        if (!active) return;
        setState({ data, isLoading: false, isRefreshing: false, error: null });
      })
      .catch((caught: unknown) => {
        if (!active) return;
        const error = toApiError(caught);
        // An abort is our own teardown, not something to show the user.
        if (error.isAborted) return;
        setState((previous) => ({
          data: previous.data,
          isLoading: false,
          isRefreshing: false,
          error,
        }));
      });

    return () => {
      active = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, nonce, ...deps]);

  const refetch = useCallback((): void => setNonce((value) => value + 1), []);

  // While disabled - typically "no id yet" - report idle rather than loading,
  // and hide any data left over from a previous id.
  if (!enabled) {
    return { data: null, isLoading: false, isRefreshing: false, error: null, refetch };
  }

  return { ...state, refetch };
}

export interface ActionResult<TArgs extends readonly unknown[], TResult> {
  /** Invoke the action. Resolves with the result, or `null` when it failed. */
  run(...args: TArgs): Promise<TResult | null>;
  readonly data: TResult | null;
  readonly isPending: boolean;
  readonly error: ApiError | null;
  /** Clear `data` and `error` without running anything. */
  reset(): void;
}

/**
 * A user-initiated call.
 *
 * Rejections are captured into `error` rather than propagated, because an
 * unhandled rejection from an event handler is a blank screen. `run` still
 * resolves with `null` so a caller can branch on failure.
 */
export function useAction<TArgs extends readonly unknown[], TResult>(
  action: (...args: TArgs) => Promise<TResult>,
): ActionResult<TArgs, TResult> {
  const [data, setData] = useState<TResult | null>(null);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const mounted = useRef(true);
  const actionRef = useRef(action);

  // Same reasoning as `useQuery`: the action is an inline closure, and `run`
  // must stay referentially stable so a caller can put it in a dependency array.
  useEffect(() => {
    actionRef.current = action;
  }, [action]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async (...args: TArgs): Promise<TResult | null> => {
    setIsPending(true);
    setError(null);
    try {
      const result = await actionRef.current(...args);
      if (mounted.current) {
        setData(result);
        setIsPending(false);
      }
      return result;
    } catch (caught) {
      const apiError = toApiError(caught);
      if (mounted.current) {
        setIsPending(false);
        if (!apiError.isAborted) setError(apiError);
      }
      return null;
    }
  }, []);

  const reset = useCallback((): void => {
    setData(null);
    setError(null);
    setIsPending(false);
  }, []);

  return { run, data, isPending, error, reset };
}
