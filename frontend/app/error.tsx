"use client";

/**
 * The route-level error boundary.
 *
 * Deliberately non-technical. The person reading this did not cause it, cannot
 * fix it, and does not need to know what a hydration mismatch is. They need to
 * know their answers are safe and what to press.
 *
 * `error.digest` is Next's own correlation id for the server-side log entry. It
 * identifies an occurrence, not a person, so it is safe to show and it is the
 * only way a report becomes actionable.
 */

import Link from "next/link";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";

export default function RouteError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Next already logs this server-side. Re-logging it in the browser would
    // risk putting a message containing user input into the console.
  }, [error]);

  return (
    <div className="page-container py-16">
      <div className="reading-column space-y-5">
        <h1 className="text-2xl font-semibold">This page could not load</h1>
        <p>
          Something went wrong on our side, not yours. Anything you had already typed is saved on
          this device, so you should not need to retype it.
        </p>
        <div className="flex flex-wrap gap-3">
          <Button onClick={reset} className="touch-row">
            Try this page again
          </Button>
          <Button asChild variant="outline" className="touch-row">
            <Link href="/">Go to the start</Link>
          </Button>
        </div>
        {error.digest === undefined ? null : (
          <p className="text-muted-foreground text-xs">
            If you tell us about this, quote reference{" "}
            <code className="bg-muted rounded px-1 py-0.5 font-mono select-all">
              {error.digest}
            </code>
          </p>
        )}
      </div>
    </div>
  );
}
