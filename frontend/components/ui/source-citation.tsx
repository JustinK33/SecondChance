import { ExternalLink } from "lucide-react";
import type { ReactNode } from "react";

import { formatRelativeAge, freshnessLabel } from "@/lib/format";
import type { Citation, ProgramSource } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Where a statement came from.
 *
 * The product rule is that nothing is asserted without a source, so this
 * component is not optional decoration - it is the evidence. It shows the
 * publisher, the excerpt the claim rests on, when we read it, and a link to the
 * original. The excerpt is quoted rather than paraphrased, because a paraphrase
 * is a new claim.
 *
 * External links get `rel="noopener noreferrer"` and an explicit "opens in a new
 * tab" announcement, so nobody is navigated away without warning.
 */
export function SourceCitation({
  source,
  className,
}: {
  source: ProgramSource;
  className?: string;
}): ReactNode {
  const age = formatRelativeAge(source.retrieved_at);
  return (
    <div
      className={cn("bg-notice/40 border-notice-border space-y-2 rounded-md border p-3", className)}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="link touch-row inline-flex items-center gap-1 text-sm font-medium"
        >
          {source.title}
          <ExternalLink aria-hidden="true" className="size-3.5" />
          <span className="sr-only">(opens in a new tab)</span>
        </a>
        <span className="text-muted-foreground text-xs">{source.publisher}</span>
      </div>
      <blockquote className="text-muted-foreground border-notice-border border-l-2 pl-3 text-sm italic">
        &ldquo;{source.excerpt}&rdquo;
      </blockquote>
      <p className="text-muted-foreground text-xs">
        {freshnessLabel(source.freshness)}
        {age === null ? "" : ` · read ${age}`}
      </p>
    </div>
  );
}

/** The assistant's lighter-weight citation, which carries no freshness field. */
export function CitationLink({
  citation,
  className,
}: {
  citation: Citation;
  className?: string;
}): ReactNode {
  return (
    <div className={cn("space-y-1", className)}>
      <a
        href={citation.url}
        target="_blank"
        rel="noopener noreferrer"
        className="link touch-row inline-flex items-center gap-1 text-sm font-medium"
      >
        {citation.title}
        <ExternalLink aria-hidden="true" className="size-3.5" />
        <span className="sr-only">(opens in a new tab)</span>
      </a>
      <blockquote className="text-muted-foreground border-border border-l-2 pl-3 text-sm italic">
        &ldquo;{citation.excerpt}&rdquo;
      </blockquote>
    </div>
  );
}

/** A list of sources under a heading, for a program detail panel. */
export function SourceList({
  sources,
  heading = "Where this comes from",
  className,
}: {
  sources: readonly ProgramSource[];
  heading?: string;
  className?: string;
}): ReactNode {
  if (sources.length === 0) {
    return (
      <p className={cn("text-muted-foreground text-sm", className)}>
        We do not have a verified source on file for this program yet, so treat the details above as
        a starting point and check the official page.
      </p>
    );
  }
  return (
    <section className={cn("space-y-3", className)}>
      <h3 className="text-sm font-semibold">{heading}</h3>
      <ul className="space-y-3">
        {sources.map((source) => (
          <li key={source.id}>
            <SourceCitation source={source} />
          </li>
        ))}
      </ul>
    </section>
  );
}
