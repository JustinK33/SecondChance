"use client";

/**
 * The clarification step: the questions the run actually asked, one at a time.
 *
 * Decisions worth knowing about:
 *
 * - One question per screen. Several at once reads as a form, and a form is the
 *   thing people give up on. One at a time with visible progress is slower to
 *   look at and faster to finish.
 * - The questions are whatever the run returned. None are hard-coded, none are
 *   reordered, and the control is the one the question declared.
 * - Every question can be answered "I am not sure", including a required one.
 *   That records the field as unknown, which leaves the affected programs at
 *   "need a bit more info". Nothing is ever ruled out for a missing answer, so
 *   there is no reason to make anyone guess.
 * - A question that asks for an identifier we have promised never to collect is
 *   dropped here and submitted as skipped. That should be impossible upstream;
 *   this is the backstop that makes the promise structural rather than a policy
 *   someone has to remember.
 *
 * This component is also what the run screen should render while a run sits at
 * `AWAITING_INPUT`, so the two places do not drift.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useId, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { useAnnouncer } from "@/lib/announcer";
import { useWorkflow } from "@/lib/hooks";
import type { ClarificationAnswer, ClarificationQuestion } from "@/lib/types";

import { ClarificationQuestionCard } from "./clarification-question-card";

/**
 * Identifiers no program needs from us, and which this product has promised
 * never to ask for. Matched against the question text and the field name.
 */
const FORBIDDEN_PATTERN =
  /social security|\bssn\b|\bsin\b|bank account|routing number|account number|card number|\bcvv\b|password|\bpin\b|driver'?s licen|passport number|alien number|\ba-?number\b/i;

function isForbidden(question: ClarificationQuestion): boolean {
  return FORBIDDEN_PATTERN.test(question.question) || FORBIDDEN_PATTERN.test(question.field);
}

/** True when the answer carries something the backend can use. */
function hasValue(answer: ClarificationAnswer | null): boolean {
  if (answer === null) return false;
  if (answer.skipped === true) return true;
  const value = answer.value;
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") {
    const range = value as { min?: number | null; max?: number | null };
    return typeof range.min === "number" || typeof range.max === "number";
  }
  return true;
}

export function ClarificationFlow({ runId }: { runId: string }): ReactNode {
  const router = useRouter();
  const { announce } = useAnnouncer();
  const idPrefix = useId();

  const run = useWorkflow(runId);
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, ClarificationAnswer>>({});
  const [error, setError] = useState<string | null>(null);

  const askable = useMemo(
    () => run.questions.filter((question) => !isForbidden(question)),
    [run.questions],
  );
  const refused = useMemo(
    () => run.questions.filter((question) => isForbidden(question)),
    [run.questions],
  );

  const total = askable.length;
  const current: ClarificationQuestion | undefined = askable[index];

  const submit = useCallback(
    async (collected: Record<string, ClarificationAnswer>): Promise<void> => {
      // Anything we refused to ask is submitted as skipped, so the run stops
      // waiting on it and the field stays honestly unknown.
      const payload = [
        ...askable.map(
          (question) =>
            collected[question.id] ?? {
              question_id: question.id,
              field: question.field,
              value: null,
              skipped: true,
            },
        ),
        ...refused.map((question) => ({
          question_id: question.id,
          field: question.field,
          value: null,
          skipped: true,
        })),
      ];
      announce("Sending your answers.");
      const next = await run.submitAnswers(payload);
      if (next === null) return;
      announce("Thanks. Carrying on with the search.");
      router.push(ROUTES.results(runId));
    },
    [askable, refused, announce, run, router, runId],
  );

  if (run.isLoading) {
    return <QuestionSkeleton />;
  }

  if (run.error !== null) {
    return (
      <ErrorState
        error={run.error}
        title="We could not load your questions"
        onRetry={run.refetch}
      />
    );
  }

  if (!run.isAwaitingInput || total === 0) {
    return (
      <div className="space-y-4">
        <p>
          There are no questions waiting for you right now. Your search is either still running or
          already finished.
        </p>
        <Button asChild className="touch-row">
          <Link href={ROUTES.results(runId)}>Go to your results</Link>
        </Button>
      </div>
    );
  }

  if (current === undefined) {
    // Index ran past the end, which only happens if the run's questions changed
    // under us. Start again from the first one rather than showing a blank card.
    return (
      <div className="space-y-4">
        <p>Those questions have changed. Let us start again from the first one.</p>
        <Button onClick={() => setIndex(0)} className="touch-row">
          Show the questions
        </Button>
      </div>
    );
  }

  const answer = answers[current.id] ?? null;
  const isLast = index === total - 1;

  const onContinue = (): void => {
    if (!hasValue(answer)) {
      setError("Choose an answer, or pick “I am not sure” to leave it blank.");
      announce("Choose an answer, or pick I am not sure.", "assertive");
      return;
    }
    setError(null);
    const collected = { ...answers, [current.id]: answer as ClarificationAnswer };
    if (isLast) {
      void submit(collected);
      return;
    }
    setIndex(index + 1);
    announce(`Question ${index + 2} of ${total}.`);
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-muted-foreground text-sm" aria-live="polite">
          Question {index + 1} of {total}
          {current.is_required === true ? "" : " - optional"}
        </p>
        <Progress
          value={Math.round(((index + 1) / total) * 100)}
          aria-label={`Question ${index + 1} of ${total}`}
          className="h-2"
        />
      </div>

      <ClarificationQuestionCard
        question={current}
        answer={answer}
        onChange={(next) => {
          setAnswers((previous) => ({ ...previous, [current.id]: next }));
          if (error !== null) setError(null);
        }}
        error={error}
        idPrefix={idPrefix}
        disabled={run.isSubmittingAnswers}
      />

      {run.submitError !== null ? (
        <ErrorState
          error={run.submitError}
          title="We could not send your answers"
          onRetry={() => void submit(answers)}
        />
      ) : null}

      <div className="flex flex-col gap-3 border-t pt-6 sm:flex-row">
        <Button
          size="lg"
          onClick={onContinue}
          disabled={run.isSubmittingAnswers}
          className="touch-row w-full px-6 text-base sm:w-auto"
        >
          {run.isSubmittingAnswers
            ? "Sending..."
            : isLast
              ? "Done - carry on with the search"
              : "Next question"}
        </Button>
        {index > 0 ? (
          <Button
            size="lg"
            variant="outline"
            disabled={run.isSubmittingAnswers}
            onClick={() => {
              setError(null);
              setIndex(index - 1);
              announce(`Question ${index} of ${total}.`);
            }}
            className="touch-row w-full px-6 text-base sm:w-auto"
          >
            Back
          </Button>
        ) : null}
      </div>

      <p aria-live="polite" className="text-muted-foreground text-sm empty:hidden">
        {run.isSubmittingAnswers ? "Sending your answers and carrying on with the search." : ""}
      </p>

      <p className="text-muted-foreground text-sm">
        We only ask questions whose answers could change your results. We never ask for a Social
        Security number, bank details or a password.
      </p>
    </div>
  );
}

/**
 * The shape of the question that is coming, while the run is being read.
 *
 * A bare spinner tells someone nothing and the page jumps when the real card
 * arrives. This is the same layout - progress, question, "why we are asking",
 * two choices - so the only thing that changes on load is the text.
 */
function QuestionSkeleton(): ReactNode {
  return (
    <div className="space-y-6">
      <p role="status" className="sr-only">
        Loading your questions.
      </p>
      <div aria-hidden="true" className="space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-2 w-full" />
        </div>
        <Skeleton className="h-7 w-3/4" />
        <Skeleton className="h-16 w-full rounded-lg" />
        <Skeleton className="h-20 w-full rounded-lg" />
        <Skeleton className="h-20 w-full rounded-lg" />
        <Skeleton className="h-12 w-48 rounded-md" />
      </div>
    </div>
  );
}
