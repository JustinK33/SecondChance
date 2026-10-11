"use client";

/**
 * Screen-reader announcements for things that change without a navigation.
 *
 * A polite `aria-live` region is the only way a sighted-user-only progress
 * animation becomes perceivable to someone using a screen reader, and this app's
 * central interaction - a run that takes tens of seconds - is exactly that case.
 *
 * The region itself lives in `components/layout/StatusAnnouncer`. This module is
 * only the channel, so a hook can announce without importing a component.
 */

import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type Politeness = "polite" | "assertive";

export interface AnnouncerValue {
  /** The current polite message, or the empty string. */
  readonly politeMessage: string;
  /** The current assertive message, or the empty string. */
  readonly assertiveMessage: string;
  /** Replace the live message. Repeating the same text is a no-op, as it would not be read again. */
  announce(message: string, politeness?: Politeness): void;
  /** Empty both regions. */
  clear(): void;
}

const AnnouncerContext = createContext<AnnouncerValue | null>(null);

export function AnnouncerProvider({ children }: { children: ReactNode }): ReactNode {
  const [politeMessage, setPolite] = useState("");
  const [assertiveMessage, setAssertive] = useState("");

  const announce = useCallback((message: string, politeness: Politeness = "polite"): void => {
    const text = message.trim();
    if (politeness === "assertive") setAssertive(text);
    else setPolite(text);
  }, []);

  const clear = useCallback((): void => {
    setPolite("");
    setAssertive("");
  }, []);

  const value = useMemo<AnnouncerValue>(
    () => ({ politeMessage, assertiveMessage, announce, clear }),
    [politeMessage, assertiveMessage, announce, clear],
  );

  return <AnnouncerContext.Provider value={value}>{children}</AnnouncerContext.Provider>;
}

/**
 * The announcer, or a no-op when no provider is mounted.
 *
 * A missing announcer must never break a screen - the consequence is a missed
 * announcement, not a crash - so this degrades instead of throwing.
 */
export function useAnnouncer(): AnnouncerValue {
  const value = useContext(AnnouncerContext);
  return value ?? NOOP;
}

const NOOP: AnnouncerValue = {
  politeMessage: "",
  assertiveMessage: "",
  announce: () => undefined,
  clear: () => undefined,
};
