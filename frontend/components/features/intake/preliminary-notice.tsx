import { Info } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * "This is not a decision."
 *
 * The single most important sentence on the product. A screener that reads like
 * an approval will send someone to an office with the wrong expectation, and
 * that costs them a day they could not spare. It is stated in the user's own
 * terms - what they will get, and who actually decides - rather than as legal
 * boilerplate, and it is never collapsed behind a toggle.
 */
export function PreliminaryNotice({
  className,
  heading = "What this is and is not",
}: {
  className?: string;
  heading?: string;
}): ReactNode {
  return (
    <section
      aria-labelledby="preliminary-notice-heading"
      className={cn(
        "bg-notice text-notice-foreground border-notice-border flex items-start gap-3 rounded-lg border p-4",
        className,
      )}
    >
      <Info aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
      <div className="space-y-2 text-sm">
        <h2 id="preliminary-notice-heading" className="text-base font-semibold">
          {heading}
        </h2>
        <p>
          What you get here is a starting point, not an answer. We can tell you which programs look
          worth your time based on what you told us. We cannot tell you whether you will get them.
        </p>
        <p>
          Only the office or organisation running a program can decide that, and they will ask for
          proof we never see. We are an independent student project, not a government agency.
        </p>
      </div>
    </section>
  );
}
