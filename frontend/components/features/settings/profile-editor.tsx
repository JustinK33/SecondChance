"use client";

/**
 * "Here is what we understood about you" - and the ability to correct it.
 *
 * Correcting a profile is not a save; it is a new evaluation. The backend has no
 * profile endpoint on purpose: the profile the user should see is the one on
 * their most recent run, and a correction is a re-match over the edited fields.
 * So the button says what actually happens - we check again - and the warning
 * above it says why the list they have will change.
 *
 * The edited value is derived during render from the loaded profile plus the
 * user's pending edits, rather than copied into state by an effect. That keeps
 * a freshly loaded profile from silently overwriting something half-typed.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { TriangleAlert } from "lucide-react";

import {
  HOUSEHOLD_SIZE_MAX,
  HOUSEHOLD_SIZE_MIN,
  StructuredHelpers,
  fromProfile,
  toProfileDraft,
  type StructuredAnswers,
} from "@/components/features/intake";
import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingState } from "@/components/ui/loading-state";
import { useAnnouncer } from "@/lib/announcer";
import { useProfile } from "@/lib/hooks";

export function ProfileEditor(): ReactNode {
  const router = useRouter();
  const { announce } = useAnnouncer();
  const profile = useProfile();

  const base = useMemo<StructuredAnswers>(() => fromProfile(profile.profile), [profile.profile]);
  const [edits, setEdits] = useState<Partial<StructuredAnswers>>({});
  const answers = useMemo<StructuredAnswers>(() => ({ ...base, ...edits }), [base, edits]);

  const isDirty = useMemo(() => JSON.stringify(answers) !== JSON.stringify(base), [answers, base]);
  const householdError =
    answers.householdSize !== null &&
    (answers.householdSize < HOUSEHOLD_SIZE_MIN || answers.householdSize > HOUSEHOLD_SIZE_MAX)
      ? `Enter a number between ${HOUSEHOLD_SIZE_MIN} and ${HOUSEHOLD_SIZE_MAX}, or leave it blank.`
      : null;

  if (profile.isLoading) {
    return <LoadingState message="Loading your information..." />;
  }

  if (profile.error !== null) {
    return (
      <ErrorState
        error={profile.error}
        title="We could not load your information"
        onRetry={profile.refetch}
      />
    );
  }

  if (profile.profile === null) {
    return (
      <EmptyState
        title="We do not hold anything about you yet"
        description="Once you describe your situation and we look for programs, what we understood will appear here for you to check and correct."
        action={
          <Button asChild className="touch-row">
            <Link href={ROUTES.start}>Describe your situation</Link>
          </Button>
        }
      />
    );
  }

  const onSave = (): void => {
    if (householdError !== null) {
      announce(householdError, "assertive");
      return;
    }
    announce("Saving your changes and checking the programs again.");
    void profile.saveChanges(toProfileDraft(answers) ?? {}).then((runId) => {
      if (runId === null) return;
      setEdits({});
      router.push(ROUTES.results(runId));
    });
  };

  return (
    <section aria-labelledby="profile-editor-heading" className="space-y-6">
      <div className="space-y-2">
        <h2 id="profile-editor-heading" className="text-xl font-semibold">
          What we understood about you
        </h2>
        <p className="text-muted-foreground">
          This is what your last search was based on. Anything blank is treated as &ldquo;not
          known&rdquo; - never as a no.
        </p>
      </div>

      {profile.unknownFields.length > 0 ? (
        <p className="text-muted-foreground text-sm">
          Some things were never filled in. That is fine: it means some programs will say we need a
          bit more information rather than ruling you out.
        </p>
      ) : null}

      <StructuredHelpers
        value={answers}
        onChange={(patch) => setEdits((previous) => ({ ...previous, ...patch }))}
        idPrefix="profile"
        disabled={profile.isSaving}
        householdError={householdError}
      />

      <div
        className="bg-notice text-notice-foreground border-notice-border flex items-start gap-3 rounded-lg border p-4 text-sm"
        role="note"
      >
        <TriangleAlert aria-hidden="true" className="mt-0.5 size-5 shrink-0" />
        <div className="reading-width space-y-1">
          <p className="font-medium">Changing this starts a new search</p>
          <p>
            We check the programs again with your corrections, so the list and the plan you have now
            will be replaced by a new one. Your old plan stays in your history.
          </p>
        </div>
      </div>

      {profile.saveError !== null ? (
        <ErrorState error={profile.saveError} title="We could not save that" onRetry={onSave} />
      ) : null}

      <div className="flex flex-col gap-3 sm:flex-row">
        <Button
          size="lg"
          onClick={onSave}
          disabled={profile.isSaving || !isDirty || householdError !== null}
          className="touch-row w-full px-6 text-base sm:w-auto"
        >
          {profile.isSaving ? "Checking again..." : "Save and check the programs again"}
        </Button>
        {isDirty ? (
          <Button
            size="lg"
            variant="outline"
            disabled={profile.isSaving}
            onClick={() => {
              setEdits({});
              announce("Your changes were discarded.");
            }}
            className="touch-row w-full px-6 text-base sm:w-auto"
          >
            Undo my changes
          </Button>
        ) : null}
      </div>

      <p aria-live="polite" className="text-muted-foreground text-sm empty:hidden">
        {profile.isSaving ? "Saving your changes and checking the programs again." : ""}
      </p>
    </section>
  );
}
