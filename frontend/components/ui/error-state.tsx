"use client";

import { RotateCcw, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/errors";
import { cn } from "@/lib/utils";

/**
 * Something failed, said without jargon.
 *
 * Rules this encodes:
 *
 * - The backend's `message` is plain language and safe to render verbatim, so it
 *   is preferred over anything we would write here.
 * - A retry button appears only when a retry could plausibly work. Offering one
 *   for a 404 just wastes a tap.
 * - The request id is shown, small and selectable, because it is the one thing
 *   that makes a support conversation possible. It identifies a request, not a
 *   person.
 * - No stack trace, no status code, no exception class.
 */
export function ErrorState({
  error,
  title = "Something went wrong",
  onRetry,
  className,
}: {
  /** An `ApiError` where possible; any thrown value is handled. */
  error?: unknown;
  title?: string;
  onRetry?: () => void;
  className?: string;
}): ReactNode {
  const apiError = error instanceof ApiError ? error : null;
  const message = apiError?.message ?? "We could not finish that. Please try again in a moment.";
  const hint = apiError?.hint ?? null;
  const requestId = apiError?.requestId ?? null;
  const showRetry = onRetry !== undefined && (apiError === null || apiError.isRetryable);

  return (
    <div
      role="alert"
      className={cn(
        "bg-card flex flex-col items-start gap-3 rounded-lg border p-4 sm:p-6",
        className,
      )}
    >
      <div className="flex items-center gap-2">
        <TriangleAlert aria-hidden="true" className="text-destructive size-5 shrink-0" />
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      <p className="reading-width text-sm">{message}</p>
      {hint !== null ? <p className="text-muted-foreground text-sm">{hint}</p> : null}
      {showRetry ? (
        <Button onClick={onRetry} variant="outline" className="touch-row">
          <RotateCcw aria-hidden="true" className="size-4" />
          Try again
        </Button>
      ) : null}
      {requestId !== null ? (
        <p className="text-muted-foreground text-xs">
          If you contact us about this, quote reference{" "}
          <code className="bg-muted rounded px-1 py-0.5 font-mono select-all">{requestId}</code>
        </p>
      ) : null}
    </div>
  );
}

/** A compact inline variant for a failure next to one control. */
export function InlineError({
  error,
  className,
}: {
  error?: unknown;
  className?: string;
}): ReactNode {
  const apiError = error instanceof ApiError ? error : null;
  if (error === undefined || error === null) return null;
  return (
    <p role="alert" className={cn("text-destructive flex items-start gap-1.5 text-sm", className)}>
      <TriangleAlert aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      <span>{apiError?.message ?? "That did not work. Please try again."}</span>
    </p>
  );
}
