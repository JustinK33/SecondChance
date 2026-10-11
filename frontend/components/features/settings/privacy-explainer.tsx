import { Lock } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * What we hold, who can see it, and how long it stays - in plain sentences.
 *
 * A privacy policy nobody reads is not consent. This is the short version,
 * placed on the screen where someone is already looking at their own data, and
 * every line is a statement about this system rather than a general assurance.
 */

const POINTS: readonly { heading: string; body: string }[] = [
  {
    heading: "What we keep",
    body: "What you wrote about your situation, the answers you gave us, the programs we matched and the plan we produced. Nothing else.",
  },
  {
    heading: "What we never ask for",
    body: "Your Social Security number, your bank or card details, a password, or an immigration document number. There is nowhere on this site to enter them, and no program we list needs them from us.",
  },
  {
    heading: "What we never write down",
    body: "Your words and your numbers never appear in our logs. We record that a search happened and how long it took, not what was in it.",
  },
  {
    heading: "Who can see it",
    body: "Only you. We do not sell it, we do not share it with any agency or university office, and applying for a program is always something you do yourself on the official site.",
  },
  {
    heading: "How long it stays",
    body: "Until you delete it. You can remove everything at any time, and we will show you exactly what was removed.",
  },
];

export function PrivacyExplainer({ className }: { className?: string }): ReactNode {
  return (
    <section
      aria-labelledby="privacy-explainer-heading"
      className={cn(
        "bg-notice text-notice-foreground border-notice-border rounded-lg border p-4",
        className,
      )}
    >
      <h2
        id="privacy-explainer-heading"
        className="flex items-center gap-2 text-base font-semibold"
      >
        <Lock aria-hidden="true" className="size-4 shrink-0" />
        How we handle your information
      </h2>
      <dl className="reading-width mt-3 space-y-3 text-sm">
        {POINTS.map((point) => (
          <div key={point.heading}>
            <dt className="font-medium">{point.heading}</dt>
            <dd>{point.body}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
