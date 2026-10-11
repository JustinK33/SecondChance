"use client";

import { CircleHelp, ExternalLink, MessageCircleQuestion, Phone } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { LastVerified } from "@/components/ui/last-verified";
import { LoadingCards } from "@/components/ui/loading-state";
import { SourceList } from "@/components/ui/source-citation";
import { StatusBadge } from "@/components/ui/status-badge";
import { formatDate, programCategoryLabel } from "@/lib/format";
import { useProgram, useWorkflow } from "@/lib/hooks";
import { statusPresentation } from "@/lib/status";

import { EvidenceList, evidenceWithOutcome } from "../results/evidence-list";
import { describeMissingFields } from "../results/field-labels";
import { UnverifiedNotice } from "../results/unverified-notice";

/**
 * Everything on file about one program.
 *
 * Two audiences on one page, and the page has to serve both without blurring
 * them:
 *
 * - A browser who arrived from the catalog. They get the official name, who
 *   runs it, what it is in plain language, the requirements as readable bullets,
 *   and the sources those bullets came from with the date each was read.
 * - Someone who arrived from their own results, identified by `runId`. They
 *   additionally get their own evaluation: the status the engine decided, the
 *   evidence behind it and what is still missing.
 *
 * The requirement bullets are rendered from author-written `description` text
 * only. Nothing on this page is generated prose about what a program requires,
 * because a fabricated requirement is the most damaging thing this product
 * could produce.
 *
 * Both outbound links say they leave SecondChance. This project is not the
 * provider and never pretends to be the application.
 */
export function ProgramDetailView({
  programId,
  runId = null,
}: {
  programId: string;
  /** The run whose evaluation to show, when the visitor came from their results. */
  runId?: string | null;
}): ReactNode {
  const { program: detail, isLoading, error, refetch } = useProgram(programId);
  const workflow = useWorkflow(runId);
  const match = workflow.matches.find((candidate) => candidate.program.id === programId) ?? null;

  if (isLoading && detail === null) {
    return <LoadingCards count={3} />;
  }

  if (detail === null) {
    return (
      <div className="space-y-6">
        <ErrorState error={error} title="We could not open that program" onRetry={refetch} />
        <Button asChild variant="outline" size="lg" className="touch-row">
          <Link href={ROUTES.programs}>Back to all programs</Link>
        </Button>
      </div>
    );
  }

  const program = detail.program;
  const requirements = detail.requirements ?? [];
  const sources = detail.sources ?? [];
  const limitations = detail.limitations ?? [];
  const whoItHelps = program.who_it_helps ?? [];
  const phone =
    typeof program.phone === "string" && program.phone.length > 0 ? program.phone : null;
  const applicationUrl =
    typeof program.application_url === "string" && program.application_url.length > 0
      ? program.application_url
      : null;
  const effectiveFrom = formatDate(detail.rule_effective_from);

  const positiveLeaves =
    match === null ? [] : evidenceWithOutcome(match.evaluation.evidence, "TRUE");
  const unknownLeaves =
    match === null ? [] : evidenceWithOutcome(match.evaluation.evidence, "UNKNOWN");
  const negativeLeaves =
    match === null ? [] : evidenceWithOutcome(match.evaluation.evidence, "FALSE");
  const missingPhrases =
    match === null ? [] : describeMissingFields(match.evaluation.missing_fields ?? []);

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold sm:text-3xl">{program.official_name}</h1>
        <p className="text-muted-foreground">
          Run by {program.administering_org} · {programCategoryLabel(program.category)} ·{" "}
          {program.geo_scope.label}
        </p>
        <LastVerified
          status={program.verification_status}
          lastVerifiedAt={program.last_verified_at}
        />
      </header>

      <UnverifiedNotice
        status={program.verification_status}
        programName={program.official_name}
        officialUrl={program.official_url}
      />

      <section aria-labelledby="program-summary" className="space-y-3">
        <h2 id="program-summary" className="text-xl font-semibold">
          What this is
        </h2>
        <p className="reading-width text-sm leading-relaxed">{program.plain_language_summary}</p>
        <p className="text-muted-foreground reading-width text-sm leading-relaxed">
          {program.description}
        </p>
        {whoItHelps.length > 0 ? (
          <>
            <h3 className="text-sm font-semibold">Who it is for</h3>
            <ul className="list-disc space-y-1 pl-5 text-sm">
              {whoItHelps.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </>
        ) : null}
      </section>

      <section aria-labelledby="program-links" className="space-y-3">
        <h2 id="program-links" className="text-xl font-semibold">
          Applying
        </h2>
        <p className="text-muted-foreground reading-width text-sm">
          Applications are handled by {program.administering_org}, not by SecondChance. These links
          open their site in a new tab.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          {applicationUrl !== null ? (
            <Button asChild size="lg" className="touch-row">
              <a href={applicationUrl} target="_blank" rel="noopener noreferrer">
                Start the official application
                <ExternalLink aria-hidden="true" className="size-4" />
                <span className="sr-only">(opens in a new tab, leaving SecondChance)</span>
              </a>
            </Button>
          ) : null}
          <Button asChild variant="outline" size="lg" className="touch-row">
            <a href={program.official_url} target="_blank" rel="noopener noreferrer">
              Official information page
              <ExternalLink aria-hidden="true" className="size-4" />
              <span className="sr-only">(opens in a new tab, leaving SecondChance)</span>
            </a>
          </Button>
          {phone !== null ? (
            <Button asChild variant="ghost" size="lg" className="touch-row">
              <a href={`tel:${phone.replace(/[^0-9+]/g, "")}`}>
                <Phone aria-hidden="true" className="size-4" />
                Call {phone}
              </a>
            </Button>
          ) : null}
        </div>
      </section>

      {match !== null ? (
        <Card aria-labelledby="your-result">
          <CardHeader className="gap-2">
            <CardTitle className="text-xl">
              <h2 id="your-result">What we found for you</h2>
            </CardTitle>
            <StatusBadge status={match.status} />
            <p className="text-muted-foreground text-sm">
              {statusPresentation(match.status).description}
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            {match.explanation !== null && match.explanation !== undefined ? (
              <p className="reading-width text-sm font-medium">{match.explanation.headline}</p>
            ) : null}
            <EvidenceList
              headingLevel={3}
              heading="Why this may fit"
              description="The requirements we could check against what you told us."
              leaves={positiveLeaves}
            />
            <EvidenceList
              headingLevel={3}
              heading="What we could not check"
              leaves={unknownLeaves}
            />
            <EvidenceList
              headingLevel={3}
              heading="What did not look met"
              leaves={negativeLeaves}
            />
            {missingPhrases.length > 0 ? (
              <section className="space-y-2">
                <h3 className="text-sm font-semibold">What we still need to know</h3>
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {missingPhrases.map((phrase) => (
                    <li key={phrase} className="first-letter:uppercase">
                      {phrase}
                    </li>
                  ))}
                </ul>
                <Button asChild variant="outline" size="lg" className="touch-row">
                  <Link href={ROUTES.run(runId as string)}>Answer those questions</Link>
                </Button>
              </section>
            ) : null}
            {(match.evaluation.limitations ?? []).length > 0 ? (
              <section className="space-y-2">
                <h3 className="text-sm font-semibold">Limits of this assessment</h3>
                <ul className="text-muted-foreground list-disc space-y-1 pl-5 text-sm">
                  {(match.evaluation.limitations ?? []).map((limitation) => (
                    <li key={limitation}>{limitation}</li>
                  ))}
                </ul>
              </section>
            ) : null}
            <p className="text-muted-foreground text-xs">
              Checked with rule set {match.evaluation.rule_version}. SecondChance does not decide
              any application - {program.administering_org} does.
            </p>
          </CardContent>
        </Card>
      ) : null}

      <section aria-labelledby="program-requirements" className="space-y-3">
        <h2 id="program-requirements" className="text-xl font-semibold">
          What the program asks for
        </h2>
        {requirements.length === 0 ? (
          <p className="text-muted-foreground reading-width text-sm">
            We do not have the requirements for this program written down in a form we can check
            yet. The official page above is the authority on them.
          </p>
        ) : (
          <ul className="space-y-2">
            {requirements.map((requirement) => (
              <li key={requirement.text} className="flex items-start gap-2 text-sm">
                {requirement.is_evaluable === false ? (
                  <CircleHelp aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
                ) : (
                  <span
                    aria-hidden="true"
                    className="bg-foreground mt-2 size-1.5 shrink-0 rounded-full"
                  />
                )}
                <span>
                  {requirement.text}
                  {requirement.is_evaluable === false ? (
                    <span className="text-muted-foreground">
                      {" "}
                      - we cannot check this one for you, so confirm it with the program.
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        )}
        {detail.rule_version !== null && detail.rule_version !== undefined ? (
          <p className="text-muted-foreground text-xs">
            From rule set {detail.rule_version}
            {effectiveFrom === null ? "" : `, in effect since ${effectiveFrom}`}.
          </p>
        ) : null}
      </section>

      {limitations.length > 0 ? (
        <section aria-labelledby="program-limitations" className="space-y-2">
          <h2 id="program-limitations" className="text-base font-semibold">
            What this record does not cover
          </h2>
          <ul className="text-muted-foreground reading-width list-disc space-y-1 pl-5 text-sm">
            {limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <section aria-labelledby="program-sources" className="space-y-3">
        <h2 id="program-sources" className="text-xl font-semibold">
          Where this comes from
        </h2>
        <SourceList sources={sources} heading="Sources we read" />
      </section>

      <div className="flex flex-wrap gap-2 border-t pt-6">
        <Button asChild variant="outline" size="lg" className="touch-row">
          <Link href={`${ROUTES.assistant}?program=${encodeURIComponent(programId)}`}>
            <MessageCircleQuestion aria-hidden="true" className="size-4" />
            Ask a question about this program
          </Link>
        </Button>
        <Button asChild variant="ghost" size="lg" className="touch-row">
          <Link href={ROUTES.programs}>Back to all programs</Link>
        </Button>
      </div>
    </div>
  );
}
