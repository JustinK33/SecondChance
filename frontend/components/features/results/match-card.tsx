import { ArrowRight, ExternalLink } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { LastVerified } from "@/components/ui/last-verified";
import { StatusBadge } from "@/components/ui/status-badge";
import { humanizeEnum } from "@/lib/format";
import { statusPresentation } from "@/lib/status";
import type { ProgramMatch } from "@/lib/types";

import { EvidenceList, evidenceWithOutcome } from "./evidence-list";
import { describeMissingFields } from "./field-labels";
import { UnverifiedNotice } from "./unverified-notice";

/**
 * One program, as it appears in a list of results.
 *
 * What the card has to carry, and why each piece is not optional:
 *
 * - The status, in the engine's words, never ours. The badge text is the only
 *   claim this card makes about the person's situation.
 * - The plain-language summary, so someone can tell in one line whether this is
 *   even the sort of help they were looking for.
 * - Why it may fit, drawn from evidence the engine actually resolved. When the
 *   model wrote a reason list we use it, because it reads better; when it did
 *   not, we fall back to the engine's own descriptions rather than showing
 *   nothing. Both are grounded in the same evaluation.
 * - What is still missing, so "need a bit more info" is actionable instead of
 *   mysterious.
 * - Who administers it and when we last checked it, because this project is not
 *   the provider and must not look like one.
 */
export function MatchCard({
  match,
  runId,
  headingLevel = 3,
}: {
  match: ProgramMatch;
  /** Threaded into the detail link so the detail page can show this evaluation. */
  runId?: string | null;
  headingLevel?: 2 | 3;
}): ReactNode {
  const program = match.program;
  const explanation = match.explanation ?? null;
  const evaluation = match.evaluation;
  const presentation = statusPresentation(match.status);
  const Heading = headingLevel === 2 ? "h2" : "h3";

  const positiveLeaves = evidenceWithOutcome(evaluation.evidence, "TRUE");
  const unknownLeaves = evidenceWithOutcome(evaluation.evidence, "UNKNOWN");
  const negativeLeaves = evidenceWithOutcome(evaluation.evidence, "FALSE");

  const modelReasons = explanation?.why_it_may_fit ?? [];
  const modelGaps = explanation?.what_we_still_need ?? [];
  const missingPhrases = describeMissingFields(evaluation.missing_fields ?? []);
  const caveats = explanation?.caveats ?? evaluation.limitations ?? [];

  const detailHref =
    typeof runId === "string" && runId.length > 0
      ? `${ROUTES.program(program.id)}?run=${encodeURIComponent(runId)}`
      : ROUTES.program(program.id);

  return (
    <Card className="gap-4">
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="space-y-1">
            <Heading className="text-base leading-snug font-semibold">
              {program.official_name}
            </Heading>
            <p className="text-muted-foreground text-sm">
              Run by {program.administering_org} · {humanizeEnum(program.category)} ·{" "}
              {program.geo_scope_label}
            </p>
          </div>
          <StatusBadge status={match.status} />
        </div>
        <p className="text-muted-foreground text-sm">{presentation.description}</p>
      </CardHeader>

      <CardContent className="space-y-4">
        <UnverifiedNotice
          status={program.verification_status}
          programName={program.official_name}
          officialUrl={program.official_url}
        />

        <p className="reading-width text-sm">{program.plain_language_summary}</p>

        {explanation !== null ? (
          <p className="reading-width text-sm font-medium">{explanation.headline}</p>
        ) : null}

        {modelReasons.length > 0 ? (
          <section className="space-y-2">
            <h4 className="text-sm font-semibold">Why this may fit</h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {modelReasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </section>
        ) : (
          <EvidenceList
            heading="Why this may fit"
            description="The requirements we were able to check against what you told us."
            leaves={positiveLeaves}
          />
        )}

        {negativeLeaves.length > 0 ? (
          <EvidenceList heading="What did not look met" leaves={negativeLeaves} />
        ) : null}

        {modelGaps.length > 0 ? (
          <section className="space-y-2">
            <h4 className="text-sm font-semibold">What we still need to know</h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {modelGaps.map((gap) => (
                <li key={gap}>{gap}</li>
              ))}
            </ul>
          </section>
        ) : missingPhrases.length > 0 ? (
          <section className="space-y-2">
            <h4 className="text-sm font-semibold">What we still need to know</h4>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {missingPhrases.map((phrase) => (
                <li key={phrase} className="first-letter:uppercase">
                  {phrase}
                </li>
              ))}
            </ul>
          </section>
        ) : unknownLeaves.length > 0 ? (
          <EvidenceList heading="What we could not check" leaves={unknownLeaves} />
        ) : null}

        {caveats.length > 0 ? (
          <section className="space-y-2">
            <h4 className="text-sm font-semibold">Worth knowing</h4>
            <ul className="text-muted-foreground list-disc space-y-1 pl-5 text-sm">
              {caveats.map((caveat) => (
                <li key={caveat}>{caveat}</li>
              ))}
            </ul>
          </section>
        ) : null}

        <LastVerified
          status={program.verification_status}
          lastVerifiedAt={program.last_verified_at}
        />

        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Button asChild variant="outline" size="lg" className="touch-row">
            <Link href={detailHref}>
              What this is and how to apply
              <ArrowRight aria-hidden="true" className="size-4" />
            </Link>
          </Button>
          {typeof program.application_url === "string" && program.application_url.length > 0 ? (
            <Button asChild variant="ghost" size="lg" className="touch-row">
              <a href={program.application_url} target="_blank" rel="noopener noreferrer">
                Apply on the official site
                <ExternalLink aria-hidden="true" className="size-4" />
                <span className="sr-only">
                  (opens {program.administering_org} in a new tab, leaving SecondChance)
                </span>
              </a>
            </Button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}
