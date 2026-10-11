import { Lock } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * A short, specific promise about the information on the current screen.
 *
 * Placed next to the input it describes rather than buried in a policy page,
 * because the moment someone hesitates over typing their income is the moment
 * the promise is worth anything.
 *
 * The default text states the two things that are structurally true of this
 * system: we never ask for identifiers the programs do not need, and personal
 * content is never written to a log.
 */
export function PrivacyNote({
  children,
  className,
}: {
  children?: ReactNode;
  className?: string;
}): ReactNode {
  return (
    <div
      className={cn(
        "bg-notice text-notice-foreground border-notice-border flex items-start gap-2 rounded-md border p-3 text-sm",
        className,
      )}
    >
      <Lock aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      <div className="reading-width space-y-1">
        {children ?? (
          <>
            <p>
              You can describe your situation in whatever detail you are comfortable with. Anything
              you leave out simply means we ask instead of assuming.
            </p>
            <p>
              We never ask for your Social Security number, bank details or passwords, and what you
              write is never written into our logs.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

/** The one-line version, for under a single field. */
export function PrivacyHint({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}): ReactNode {
  return (
    <p className={cn("text-muted-foreground flex items-start gap-1.5 text-xs", className)}>
      <Lock aria-hidden="true" className="mt-0.5 size-3 shrink-0" />
      <span>{children}</span>
    </p>
  );
}
