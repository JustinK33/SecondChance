"use client";

import { ArrowRight, ListChecks, MessageCircleQuestion, RotateCcw } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingCards } from "@/components/ui/loading-state";
import { useAnnouncer } from "@/lib/announcer";
import { pluralize } from "@/lib/format";
import { useReevaluate, useWorkflow } from "@/lib/hooks";
import { compareByStatus } from "@/lib/status";
import { MATCH_STATUSES } from "@/lib/types";
import type { MatchStatus, ProgramMatch } from "@/lib/types";

import { MatchGroup } from "./match-group";
import { ResultsProgress } from "./results-progress";

/**
 * The results screen.
 *
 * It has to be honest about four different situations and never pretend one is
 * another:
 *
 * - *still working* - the backend's own progress sentence plus skeletons.
 * - *waiting on the person* - the run is parked on a question, so this screen
 *   sends them back to answer it rather than showing an empty result.
 * - *finished* - programs grouped by status, "worth exploring" first.
 * - *failed* - the plain-language message the backend wrote, and a retry that
 *   re-runs matching against the profile already on file. Retrying never asks
 *   anyone to retype their situation.
 *
 * Polling, backoff and the stop conditions all live in `useWorkflow`. This
 * component only renders what the hook reports.
 */
export function ResultsView({ runId }: { runId: string }): ReactNode {
  const router = useRouter();
  const workflow = useWorkflow(runId);
  const reevaluation = useReevaluate();
  const { announce } = useAnnouncer();

  // Announce the progress sentence as it changes, and the completion once. A
  // sighted user sees the bar move; this is the same information for everyone
  // else. `useAnnouncer` ignores a repeat of the same text.
  const announcedRef = useRef<string>("");
  useEffect(() => {
    if (workflow.isComplete) {
      const text = `Your results are ready. ${workflow.matches.length} programs to look at.`;
      if (announcedRef.current !== text) {
        announcedRef.current = text;
        announce(text);
      }
      return;
    }
    if (workflow.isAwaitingInput) {
      const text = "We need a couple more answers before we can finish.";
      if (announcedRef.current !== text) {
        announcedRef.current = text;
        announce(text);
      }
      return;
    }
    const message = workflow.progressMessage.trim();
    if (message.length > 0 && announcedRef.current !== message) {
      announcedRef.current = message;
      announce(message);
    }
  }, [
    announce,
    workflow.isComplete,
    workflow.isAwaitingInput,
    workflow.matches.length,
    workflow.progressMessage,
  ]);

  async function retryMatching(): Promise<void> {
    // An empty change set means "same answers, run it again": the profile is
    // already on file, so nothing the user typed is lost.
    const response = await reevaluation.reevaluate(runId, {});
    if (response !== null) router.push(ROUTES.results(response.run_id));
  }

  // ---------------------------------------------------------------- loading
  if (workflow.isLoading && workflow.run === null) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">Getting your results</h1>
        <p className="text-muted-foreground">Picking up where your answers left off.</p>
        <LoadingCards count={3} />
      </div>
    );
  }

  // ------------------------------------------------------- transport failure
  if (workflow.run === null && workflow.error !== null) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">We could not load your results</h1>
        <ErrorState
          error={workflow.error}
          title="We could not load your results"
          onRetry={workflow.refetch}
        />
        <Button asChild variant="outline" size="lg" className="touch-row">
          <Link href={ROUTES.start}>Start again</Link>
        </Button>
      </div>
    );
  }

  // ------------------------------------------------------------ run failure
  const runError = workflow.runError;
  if (runError !== null || workflow.status === "FAILED") {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">That run did not finish</h1>
        <Alert>
          <AlertTitle>
            {runError?.message ?? "Something went wrong part of the way through."}
          </AlertTitle>
          <AlertDescription>
            <p>
              {runError?.retry_hint ??
                "Your answers are saved, so trying again does not mean retyping anything."}
            </p>
          </AlertDescription>
        </Alert>
        <div className="flex flex-wrap gap-2">
          <Button
            size="lg"
            className="touch-row"
            onClick={() => void retryMatching()}
            disabled={reevaluation.isPending}
          >
            <RotateCcw aria-hidden="true" className="size-4" />
            {reevaluation.isPending ? "Starting again..." : "Try again with the same answers"}
          </Button>
          <Button asChild variant="outline" size="lg" className="touch-row">
            <Link href={ROUTES.programs}>Browse programs instead</Link>
          </Button>
        </div>
        {reevaluation.error !== null ? <ErrorState error={reevaluation.error} /> : null}
      </div>
    );
  }

  // ------------------------------------------------------- awaiting an answer
  if (workflow.isAwaitingInput) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">A couple of questions first</h1>
        <p className="text-muted-foreground reading-width">
          We could not check some requirements with what we have so far. Answering a short question
          or two means we can check them instead of guessing.
        </p>
        <Button asChild size="lg" className="touch-row">
          <Link href={ROUTES.run(runId)}>
            Answer the questions
            <ArrowRight aria-hidden="true" className="size-4" />
          </Link>
        </Button>
      </div>
    );
  }

  // -------------------------------------------------------------- cancelled
  if (workflow.status === "CANCELLED") {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">This run was stopped</h1>
        <p className="text-muted-foreground reading-width">
          Nothing was lost. You can run the same answers again whenever you are ready.
        </p>
        <Button
          size="lg"
          className="touch-row"
          onClick={() => void retryMatching()}
          disabled={reevaluation.isPending}
        >
          <RotateCcw aria-hidden="true" className="size-4" />
          Run it again
        </Button>
        {reevaluation.error !== null ? <ErrorState error={reevaluation.error} /> : null}
      </div>
    );
  }

  // ----------------------------------------------------------- still working
  if (!workflow.isComplete) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">Working on your results</h1>
        <ResultsProgress
          message={workflow.progressMessage}
          fraction={workflow.progressFraction}
          onCancel={() => void workflow.cancel()}
          isCancelling={workflow.isCancelling}
        />
        {workflow.error !== null ? (
          <ErrorState
            error={workflow.error}
            title="We are having trouble checking on your run"
            onRetry={workflow.refetch}
          />
        ) : null}
      </div>
    );
  }

  // ---------------------------------------------------------------- finished
  const grouped = groupByStatus(workflow.matches);
  const planId = workflow.run?.plan?.id ?? null;
  const total = workflow.matches.length;

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold sm:text-3xl">What may be worth exploring</h1>
        <p className="text-muted-foreground reading-width">
          {total === 0
            ? "We could not match any of the programs we have on file to what you told us."
            : `We checked ${total} program${total === 1 ? "" : "s"} against what you told us. Nothing here is a decision by the program itself - the official application is.`}
        </p>
      </div>

      {planId !== null || total > 0 ? (
        <div className="flex flex-wrap gap-2">
          {planId !== null ? (
            <Button asChild size="lg" className="touch-row">
              <Link href={ROUTES.plan(planId)}>
                <ListChecks aria-hidden="true" className="size-4" />
                See my action plan
              </Link>
            </Button>
          ) : null}
          <Button asChild variant="outline" size="lg" className="touch-row">
            <Link href={ROUTES.assistant}>
              <MessageCircleQuestion aria-hidden="true" className="size-4" />
              Ask a question about these
            </Link>
          </Button>
        </div>
      ) : null}

      {total === 0 ? (
        <EmptyState
          title="Nothing matched this time"
          description="That usually means we need more detail, or that the programs we have on file do not cover this situation yet. Browsing the full list is often worth it."
          action={
            <Button asChild variant="outline" size="lg" className="touch-row">
              <Link href={ROUTES.programs}>Browse every program</Link>
            </Button>
          }
        />
      ) : (
        MATCH_STATUSES.map((status) => (
          <MatchGroup key={status} status={status} matches={grouped[status]} runId={runId} />
        ))
      )}

      {workflow.limitations.length > 0 ? <Limitations notes={workflow.limitations} /> : null}

      <p className="text-muted-foreground reading-width text-xs">
        SecondChance is an independent project. It is not a government agency and cannot apply on
        your behalf or decide any application.
      </p>
    </div>
  );
}

/**
 * The caveats behind the check, after the results rather than in front of them.
 *
 * These are real and they have to stay on the page: each one is a requirement
 * the engine could not encode, and hiding them would make the list look more
 * certain than it is. But a run against twenty-five programs produces dozens of
 * them, and rendering that wall above the matches buried the actual answer
 * under two screens of fine print. So: a closed disclosure, with the count on
 * the summary so nobody has to guess whether it is worth opening.
 *
 * `<details>` rather than a scripted accordion, because the open state has no
 * business being React state and a native disclosure is keyboard-operable and
 * announced correctly without any help from us.
 */
function Limitations({ notes }: { notes: readonly string[] }): ReactNode {
  return (
    <details className="bg-card rounded-lg border">
      <summary className="touch-row flex cursor-pointer items-center px-4 py-3 text-base font-semibold">
        What this check does not cover ({pluralize(notes.length, "note")})
      </summary>
      <div className="space-y-2 px-4 pb-4">
        <p className="text-muted-foreground reading-width text-sm">
          Each of these is a requirement we could not check for you. They are the reason nothing
          here is a decision.
        </p>
        <ul className="text-muted-foreground reading-width list-disc space-y-1 pl-5 text-sm">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </div>
    </details>
  );
}

/** Matches bucketed by status, each bucket ordered by the engine's score. */
function groupByStatus(matches: readonly ProgramMatch[]): Record<MatchStatus, ProgramMatch[]> {
  const grouped: Record<MatchStatus, ProgramMatch[]> = {
    POTENTIAL_MATCH: [],
    MORE_INFO_NEEDED: [],
    UNLIKELY_MATCH: [],
  };
  for (const match of [...matches].sort(
    (a, b) => compareByStatus(a.status, b.status) || (b.score ?? 0) - (a.score ?? 0),
  )) {
    grouped[match.status].push(match);
  }
  return grouped;
}
