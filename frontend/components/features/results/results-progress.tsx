"use client";

import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * The waiting experience for a run that is still working.
 *
 * Three rules shaped this:
 *
 * 1. The sentence on screen is the backend's `progress_message`, rendered
 *    verbatim. It is written for a person. A node name, a phase enum or the word
 *    "LangGraph" must never reach this screen, which is why this component takes
 *    a message rather than a phase and has no mapping table of its own.
 * 2. A determinate bar, because "something is happening" is not reassuring for
 *    forty seconds. `progress_fraction` is the backend's own estimate.
 * 3. Skeleton cards in the shape of the results that are coming, so the page
 *    does not jump when they arrive and so the wait looks like progress rather
 *    than a stall.
 *
 * The message is also announced politely, once per change, by the caller.
 */
export function ResultsProgress({
  message,
  fraction,
  onCancel,
  isCancelling = false,
}: {
  message: string;
  /** 0..1, from `WorkflowRunDetail.progress_fraction`. */
  fraction: number;
  onCancel?: () => void;
  isCancelling?: boolean;
}): ReactNode {
  const percent = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  const text = message.trim().length > 0 ? message : "Working through your situation.";

  return (
    <div className="space-y-6">
      <div className="space-y-3">
        <p className="text-base font-medium">{text}</p>
        <Progress
          value={percent}
          aria-label="How far through your results we are"
          className="h-2"
        />
        <p className="text-muted-foreground text-sm">
          {percent}% of the way there. This usually takes under a minute. You can stay on this page
          - it updates on its own.
        </p>
      </div>

      <ol className="space-y-4" aria-hidden="true">
        {[0, 1, 2].map((index) => (
          <li key={index} className="space-y-3 rounded-lg border p-4">
            <div className="flex items-start justify-between gap-3">
              <Skeleton className="h-5 w-1/2" />
              <Skeleton className="h-6 w-28 rounded-full" />
            </div>
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-2/3" />
          </li>
        ))}
      </ol>

      {onCancel !== undefined ? (
        <Button
          variant="ghost"
          size="lg"
          className="touch-row"
          onClick={onCancel}
          disabled={isCancelling}
        >
          {isCancelling ? "Stopping..." : "Stop and come back later"}
        </Button>
      ) : null}
    </div>
  );
}
