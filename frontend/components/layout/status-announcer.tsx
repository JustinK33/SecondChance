"use client";

/**
 * The application's `aria-live` regions.
 *
 * Rendered once, high in the tree, and fed by `useAnnouncer()` from anywhere.
 * Two regions rather than one because politeness is not a property of the
 * listener: run progress should wait for a pause, a failure should not.
 *
 * The regions are always present in the DOM. Mounting a live region at the same
 * time as its text is a well-known way to have nothing announced at all.
 */

import { useAnnouncer } from "@/lib/announcer";

export function StatusAnnouncer() {
  const { politeMessage, assertiveMessage } = useAnnouncer();
  return (
    <>
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {politeMessage}
      </div>
      <div className="sr-only" role="alert" aria-live="assertive" aria-atomic="true">
        {assertiveMessage}
      </div>
    </>
  );
}
