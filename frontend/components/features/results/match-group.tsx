import type { ReactNode } from "react";

import { statusPresentation } from "@/lib/status";
import { pluralize } from "@/lib/format";
import type { MatchStatus, ProgramMatch } from "@/lib/types";

import { MatchCard } from "./match-card";

/**
 * One status's worth of results, under its own heading.
 *
 * Grouping rather than one long sorted list, because the three statuses mean
 * genuinely different things and the first group is the only one most people
 * need. The heading repeats the status wording and the count, so the structure
 * is readable by heading navigation alone.
 */
export function MatchGroup({
  status,
  matches,
  runId,
}: {
  status: MatchStatus;
  matches: readonly ProgramMatch[];
  runId?: string | null;
}): ReactNode {
  if (matches.length === 0) return null;
  const presentation = statusPresentation(status);
  const headingId = `match-group-${status.toLowerCase()}`;

  return (
    <section aria-labelledby={headingId} className="space-y-4">
      <div className="space-y-1">
        <h2 id={headingId} className="text-xl font-semibold">
          {presentation.label}
        </h2>
        <p className="text-muted-foreground reading-width text-sm">
          {pluralize(matches.length, "program")}. {presentation.description} {presentation.nextStep}
        </p>
      </div>
      <ol className="space-y-4">
        {matches.map((match) => (
          <li key={match.program.id}>
            <MatchCard match={match} runId={runId} />
          </li>
        ))}
      </ol>
    </section>
  );
}
