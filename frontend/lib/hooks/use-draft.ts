"use client";

/**
 * Keeping typed answers on the device so a failure never loses them.
 *
 * Someone describing a hard month in their own words may type several hundred
 * words. Losing that to a dropped connection, an accidental back gesture or a
 * 502 is the single most damaging thing this interface could do, so every piece
 * of onboarding input is mirrored into `localStorage` as it is typed.
 *
 * Properties this needs to have, and does:
 *
 * - Hydration-safe. The first client render matches the server render, and the
 *   stored value is adopted in an effect, so React never warns and the user
 *   never sees a flash of someone else's text.
 * - Debounced. Writing on every keystroke is wasteful and janky on a phone.
 * - Fail-soft. Private browsing with storage disabled degrades to in-memory
 *   state rather than throwing on every keystroke.
 * - Not a database. `clear()` is called once the draft has been submitted,
 *   because keeping a narrative about someone's finances on their device
 *   indefinitely is not ours to decide.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const PREFIX = "secondchance.draft.";
const WRITE_DEBOUNCE_MS = 400;

export interface DraftResult<T> {
  /** The current draft. Equals `initial` until hydration completes. */
  readonly value: T;
  /** Replace the draft. */
  setValue(next: T): void;
  /** Merge a partial update into an object draft. */
  patch(partial: Partial<T>): void;
  /** Delete the stored draft and return to `initial`. Call after a successful submit. */
  clear(): void;
  /** True once the stored value has been read. Gate a "restored your answers" notice on it. */
  readonly isHydrated: boolean;
  /** True when a stored draft was found and adopted. */
  readonly wasRestored: boolean;
}

function readStored<T>(key: string): T | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    if (raw === null) return undefined;
    return JSON.parse(raw) as T;
  } catch {
    // Corrupt or unreadable. Treat as absent rather than breaking the screen.
    return undefined;
  }
}

/**
 * A `localStorage`-backed draft.
 *
 * `key` is namespaced for this app; pass something stable and non-personal such
 * as `"onboarding"`.
 */
export function useDraft<T>(key: string, initial: T): DraftResult<T> {
  const [value, setState] = useState<T>(initial);
  const [isHydrated, setIsHydrated] = useState(false);
  const [wasRestored, setWasRestored] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const initialRef = useRef(initial);

  // Adopt the stored draft after mount, never during render.
  //
  // This genuinely has to be a post-mount state update. Reading `localStorage`
  // in the initialiser would make the first client render disagree with the
  // server render, and the draft is subsequently *owned* by component state
  // because the user edits it, so `useSyncExternalStore` is not a substitute.
  // The one cascading render this costs happens once per mount, before paint.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const stored = readStored<T>(key);
    if (stored !== undefined) {
      setState(stored);
      setWasRestored(true);
    }
    setIsHydrated(true);
  }, [key]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Debounced write. Skipped until hydration so we cannot overwrite a stored
  // draft with the initial value on first paint.
  useEffect(() => {
    if (!isHydrated || typeof window === "undefined") return;
    if (timerRef.current !== null) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      try {
        window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
      } catch {
        // Storage full or disabled. In-memory state still works.
      }
    }, WRITE_DEBOUNCE_MS);
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [key, value, isHydrated]);

  const setValue = useCallback((next: T): void => setState(next), []);

  const patch = useCallback((partial: Partial<T>): void => {
    setState((current) => ({ ...(current as object), ...(partial as object) }) as T);
  }, []);

  const clear = useCallback((): void => {
    if (timerRef.current !== null) clearTimeout(timerRef.current);
    setState(initialRef.current);
    setWasRestored(false);
    if (typeof window === "undefined") return;
    try {
      window.localStorage.removeItem(PREFIX + key);
    } catch {
      // Nothing to remove.
    }
  }, [key]);

  return useMemo(
    () => ({ value, setValue, patch, clear, isHydrated, wasRestored }),
    [value, setValue, patch, clear, isHydrated, wasRestored],
  );
}
