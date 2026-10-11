import { CircleHelp, CircleMinus, CircleCheck } from "lucide-react";
import type { ReactNode } from "react";

import { statusPresentation } from "@/lib/status";
import type { MatchStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * The three match statuses, worded for a person rather than for a rule engine.
 *
 * Three invariants are enforced here so no screen can break them:
 *
 * 1. The label is always rendered as text. Colour and icon are redundant
 *    encodings, so the badge survives greyscale, colour blindness and a screen
 *    reader.
 * 2. The words are never "eligible" or "ineligible". The engine checks the
 *    requirements it can check against a partial profile; that is not a
 *    decision, and the labels say so.
 * 3. `UNLIKELY_MATCH` is quiet, not alarming. It uses the neutral grey token
 *    rather than the destructive one, because "probably not a fit" is
 *    information about a program, not an error by the person.
 */
export function StatusBadge({
  status,
  size = "default",
  className,
}: {
  status: MatchStatus;
  size?: "default" | "sm";
  className?: string;
}): ReactNode {
  const presentation = statusPresentation(status);
  const Icon =
    presentation.tone === "positive"
      ? CircleCheck
      : presentation.tone === "neutral"
        ? CircleHelp
        : CircleMinus;

  return (
    <span
      className={cn(
        "inline-flex w-fit shrink-0 items-center gap-1.5 rounded-full border font-medium",
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-sm",
        presentation.tone === "positive" &&
          "bg-status-positive text-status-positive-foreground border-status-positive-border",
        presentation.tone === "neutral" &&
          "bg-status-neutral text-status-neutral-foreground border-status-neutral-border",
        presentation.tone === "muted" &&
          "bg-status-quiet text-status-quiet-foreground border-status-quiet-border",
        className,
      )}
    >
      <Icon aria-hidden="true" className={size === "sm" ? "size-3.5" : "size-4"} />
      {presentation.label}
    </span>
  );
}

/** The badge plus its one-sentence explanation, for the top of a result card. */
export function StatusSummary({
  status,
  className,
}: {
  status: MatchStatus;
  className?: string;
}): ReactNode {
  const presentation = statusPresentation(status);
  return (
    <div className={cn("space-y-2", className)}>
      <StatusBadge status={status} />
      <p className="text-muted-foreground text-sm">{presentation.description}</p>
    </div>
  );
}
