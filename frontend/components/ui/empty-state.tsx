import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * Nothing here, explained.
 *
 * An empty list is usually the result of a filter, not of a problem, so the copy
 * is informative rather than apologetic and the component always offers
 * somewhere to go next.
 */
export function EmptyState({
  title,
  description,
  action,
  icon,
  className,
}: {
  title: string;
  description?: string;
  /** A button or link. Keep it to one. */
  action?: ReactNode;
  icon?: ReactNode;
  className?: string;
}): ReactNode {
  return (
    <div
      className={cn(
        "border-border flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed px-6 py-12 text-center",
        className,
      )}
    >
      {icon !== undefined ? (
        <span aria-hidden="true" className="text-muted-foreground">
          {icon}
        </span>
      ) : null}
      <h2 className="text-base font-semibold">{title}</h2>
      {description !== undefined ? (
        <p className="text-muted-foreground reading-width text-sm">{description}</p>
      ) : null}
      {action !== undefined ? <div className="pt-1">{action}</div> : null}
    </div>
  );
}
