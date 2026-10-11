"use client";

/**
 * Onboarding: the narrative, the optional structured helpers, and the submit.
 *
 * The flow is two steps on purpose:
 *
 * 1. Submit runs intake only - extraction, no retrieval and no eligibility - and
 *    shows what we understood. It is the cheap step, and it is the only moment
 *    where a misreading can be corrected by the one person who knows.
 * 2. Confirming starts the real run and hands off to the results screen, which
 *    polls it, asks any clarification questions and shows what came back.
 *
 * Everything typed here is mirrored to `localStorage` as it is typed, so a
 * dropped connection, a back gesture or a failed request never costs someone
 * the paragraph they just wrote about the worst month of their year. The draft
 * is deleted as soon as the run has been accepted.
 */

import { useRouter } from "next/navigation";
import { useCallback, useId, useState } from "react";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { useAnnouncer } from "@/lib/announcer";
import { useDraft, useIntake, useStartWorkflow } from "@/lib/hooks";
import type { IntakeResponse, ProfileDraft } from "@/lib/types";

import { NarrativeField } from "./narrative-field";
import { NARRATIVE_MIN_LENGTH } from "./options";
import { StructuredHelpers } from "./structured-helpers";
import {
  EMPTY_ANSWERS,
  HOUSEHOLD_SIZE_MAX,
  HOUSEHOLD_SIZE_MIN,
  toProfileDraft,
  type StructuredAnswers,
} from "./structured-answers";
import { UnderstoodSummary } from "./understood-summary";

interface OnboardingDraft {
  readonly narrative: string;
  readonly answers: StructuredAnswers;
}

const INITIAL_DRAFT: OnboardingDraft = { narrative: "", answers: EMPTY_ANSWERS };

interface FieldErrors {
  readonly narrative: string | null;
  readonly household: string | null;
}

const NO_ERRORS: FieldErrors = { narrative: null, household: null };

function validate(draft: OnboardingDraft): FieldErrors {
  const narrative = draft.narrative.trim();
  const size = draft.answers.householdSize;
  return {
    narrative:
      narrative.length === 0
        ? "Tell us what is going on before we look anything up. A sentence or two is enough."
        : narrative.length < NARRATIVE_MIN_LENGTH
          ? "Please write a little more so we have something to work with."
          : null,
    household:
      size !== null && (size < HOUSEHOLD_SIZE_MIN || size > HOUSEHOLD_SIZE_MAX)
        ? `Enter a number between ${HOUSEHOLD_SIZE_MIN} and ${HOUSEHOLD_SIZE_MAX}, or leave it blank.`
        : null,
  };
}

export function OnboardingForm(): ReactNode {
  const router = useRouter();
  const { announce } = useAnnouncer();
  const baseId = useId();
  const narrativeId = `${baseId}-narrative`;

  const draft = useDraft<OnboardingDraft>("onboarding", INITIAL_DRAFT);
  const intake = useIntake();
  const workflow = useStartWorkflow();

  const [errors, setErrors] = useState<FieldErrors>(NO_ERRORS);
  const [understood, setUnderstood] = useState<IntakeResponse | null>(null);

  const isBusy = intake.isPending || workflow.isPending;

  const prefilled = (): ProfileDraft | null => toProfileDraft(draft.value.answers);

  /** Start the run and hand off to the screen that follows it. */
  const startRun = useCallback(
    async (narrative: string, structured: ProfileDraft | null): Promise<void> => {
      announce("Looking for programs. This usually takes under a minute.");
      const created = await workflow.start(narrative, structured);
      if (created === null) return;
      // The run is durable server-side now, so the local copy has done its job.
      draft.clear();
      router.push(ROUTES.results(created.run_id));
    },
    [announce, workflow, draft, router],
  );

  const onSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (isBusy) return;

    const found = validate(draft.value);
    setErrors(found);
    if (found.narrative !== null || found.household !== null) {
      announce(found.narrative ?? found.household ?? "Please check the form.", "assertive");
      const target = document.getElementById(
        found.narrative !== null ? narrativeId : `${baseId}-household`,
      );
      target?.focus();
      return;
    }

    const narrative = draft.value.narrative.trim();
    announce("Reading what you wrote.");
    void intake.submit(narrative, prefilled()).then((response) => {
      if (response === null) return;
      setUnderstood(response);
      announce("We read what you wrote. Check it is right, then continue.");
    });
  };

  // ------------------------------------------------------------- confirm step
  if (understood !== null) {
    return (
      <div className="space-y-8">
        <UnderstoodSummary intake={understood} />

        {workflow.error !== null ? (
          <ErrorState
            error={workflow.error}
            title="We could not start the search"
            onRetry={() => void startRun(draft.value.narrative.trim(), prefilled())}
          />
        ) : null}

        <div className="flex flex-col gap-3 sm:flex-row">
          <Button
            size="lg"
            disabled={isBusy}
            onClick={() => void startRun(draft.value.narrative.trim(), prefilled())}
            className="touch-row w-full px-6 text-base sm:w-auto"
          >
            {workflow.isPending ? "Starting..." : "That is right - find programs"}
          </Button>
          <Button
            size="lg"
            variant="outline"
            disabled={isBusy}
            onClick={() => {
              setUnderstood(null);
              intake.reset();
              announce("Back to your answers.");
            }}
            className="touch-row w-full px-6 text-base sm:w-auto"
          >
            Go back and change something
          </Button>
        </div>
        <p aria-live="polite" className="text-muted-foreground text-sm empty:hidden">
          {workflow.isPending ? "Starting the search. Please stay on this page." : ""}
        </p>
      </div>
    );
  }

  // ---------------------------------------------------------------- edit step
  return (
    <form noValidate onSubmit={onSubmit} className="space-y-10">
      {draft.wasRestored ? (
        <p role="status" className="bg-muted rounded-lg border p-3 text-sm">
          We kept what you had already written on this device. You can carry on or clear it.
        </p>
      ) : null}

      <NarrativeField
        value={draft.value.narrative}
        onChange={(next) => draft.patch({ narrative: next })}
        error={errors.narrative}
        fieldId={narrativeId}
        disabled={isBusy}
      />

      <section aria-labelledby={`${baseId}-helpers-heading`} className="space-y-4 border-t pt-8">
        <h2 id={`${baseId}-helpers-heading`} className="text-xl font-semibold">
          A few optional questions
        </h2>
        <p className="text-muted-foreground reading-width">
          You can skip all of these and still get your list. They only make the match better, and we
          will ask about anything important that is missing rather than assume it.
        </p>
        <StructuredHelpers
          value={draft.value.answers}
          onChange={(patch) => draft.patch({ answers: { ...draft.value.answers, ...patch } })}
          idPrefix={baseId}
          disabled={isBusy}
          householdError={errors.household}
        />
      </section>

      {intake.error !== null ? (
        <div className="space-y-3">
          <ErrorState
            error={intake.error}
            title="We could not read that just now"
            onRetry={() => {
              intake.reset();
              void intake.submit(draft.value.narrative.trim(), prefilled()).then((response) => {
                if (response !== null) setUnderstood(response);
              });
            }}
          />
          <Button
            variant="outline"
            disabled={isBusy}
            onClick={() => void startRun(draft.value.narrative.trim(), prefilled())}
            className="touch-row"
          >
            Skip the check and search anyway
          </Button>
        </div>
      ) : null}

      <div className="space-y-3 border-t pt-8">
        <Button
          type="submit"
          size="lg"
          disabled={isBusy}
          className="touch-row w-full px-6 text-base sm:w-auto"
        >
          {intake.isPending ? "Reading what you wrote..." : "Continue"}
        </Button>
        <p aria-live="polite" className="text-muted-foreground text-sm empty:hidden">
          {intake.isPending ? "Reading what you wrote. This takes a few seconds." : ""}
        </p>
        <p className="text-muted-foreground text-sm">
          {draft.isHydrated && draft.value.narrative.length > 0
            ? "Saved on this device as you type. It is only sent when you press Continue."
            : "Nothing is sent until you press Continue."}
        </p>
        {draft.value.narrative.length > 0 ? (
          <Button
            type="button"
            variant="ghost"
            disabled={isBusy}
            onClick={() => {
              draft.clear();
              setErrors(NO_ERRORS);
              announce("Your answers were cleared.");
            }}
            className="touch-row"
          >
            Clear what I have written
          </Button>
        ) : null}
      </div>
    </form>
  );
}
