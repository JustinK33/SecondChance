import type { ReactNode } from "react";

import { Progress } from "@/components/ui/progress";
import { pluralize } from "@/lib/format";

/**
 * How far through the plan the person is.
 *
 * Deliberately encouraging and deliberately not congratulatory about things
 * that have not happened: finishing the checklist means the applications have
 * been sent, not that any of them has been decided. The wording never implies
 * an outcome.
 */
export function PlanProgress({
  completed,
  total,
}: {
  completed: number;
  total: number;
}): ReactNode {
  const percent = total === 0 ? 100 : Math.round((completed / total) * 100);
  const remaining = Math.max(0, total - completed);

  return (
    <div className="space-y-2">
      <Progress value={percent} aria-label="How much of your plan is done" className="h-2" />
      <p className="text-sm font-medium">
        {total === 0
          ? "No steps on this plan yet."
          : remaining === 0
            ? `All ${pluralize(total, "step")} done. Nice work.`
            : `${completed} of ${total} done - ${pluralize(remaining, "step")} to go.`}
      </p>
    </div>
  );
}
