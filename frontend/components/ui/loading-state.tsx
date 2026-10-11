import type { ReactNode } from "react";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

/**
 * Waiting, said out loud.
 *
 * A spinner with no text tells a screen reader nothing and tells an anxious
 * person nothing either, so the message is required and is announced through
 * `role="status"`. Where the eventual shape is known, prefer the skeleton
 * variants below: a layout that does not jump is worth more than an animation.
 */
export function LoadingState({
  message = "Loading...",
  className,
}: {
  message?: string;
  className?: string;
}): ReactNode {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn("flex flex-col items-center justify-center gap-3 px-4 py-12", className)}
    >
      <span
        aria-hidden="true"
        className="border-muted border-t-primary size-6 animate-spin rounded-full border-2 motion-reduce:animate-none"
      />
      <p className="text-muted-foreground text-sm">{message}</p>
    </div>
  );
}

/** Placeholder rows that match the shape of a card list. */
export function LoadingCards({
  count = 3,
  className,
}: {
  count?: number;
  className?: string;
}): ReactNode {
  return (
    <div className={cn("space-y-4", className)} aria-hidden="true">
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="space-y-3 rounded-lg border p-4">
          <Skeleton className="h-5 w-2/3" />
          <Skeleton className="h-4 w-1/3" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
        </div>
      ))}
    </div>
  );
}

/** Placeholder lines for a block of prose. */
export function LoadingText({
  lines = 3,
  className,
}: {
  lines?: number;
  className?: string;
}): ReactNode {
  return (
    <div className={cn("space-y-2", className)} aria-hidden="true">
      {Array.from({ length: lines }, (_, index) => (
        <Skeleton key={index} className={index === lines - 1 ? "h-4 w-2/3" : "h-4 w-full"} />
      ))}
    </div>
  );
}
