"use client";

/**
 * The one field that matters.
 *
 * Everything else on the onboarding screen is optional. This is the field the
 * whole product is built around, so it gets a real label, a generous target, an
 * example of what a good answer looks like, and an error that is attached to the
 * field rather than floating above the form.
 *
 * The example is shown rather than offered as a fill-in button: someone who is
 * stressed and unsure what to write needs to see the *shape* of an answer, and
 * a button that pastes a stranger's situation into their form invites them to
 * submit it.
 */

import type { ReactNode } from "react";

import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { PrivacyHint } from "@/components/ui/privacy-note";

import { NARRATIVE_EXAMPLE, NARRATIVE_MAX_LENGTH } from "./options";

/** Show the remaining-characters count only once it is plausibly relevant. */
const COUNTER_THRESHOLD = NARRATIVE_MAX_LENGTH - 500;

export function NarrativeField({
  value,
  onChange,
  error,
  fieldId,
  disabled = false,
}: {
  value: string;
  onChange: (next: string) => void;
  /** Plain-language problem with the current value, or `null`. */
  error: string | null;
  /** Stable id, so the parent can move focus here when it reports an error. */
  fieldId: string;
  disabled?: boolean;
}): ReactNode {
  const hintId = `${fieldId}-hint`;
  const errorId = `${fieldId}-error`;
  const counterId = `${fieldId}-counter`;
  const remaining = NARRATIVE_MAX_LENGTH - value.length;
  const showCounter = value.length > COUNTER_THRESHOLD;

  const describedBy = [hintId, showCounter ? counterId : null, error !== null ? errorId : null]
    .filter((id): id is string => id !== null)
    .join(" ");

  return (
    <div className="space-y-2">
      <Label htmlFor={fieldId} className="text-base">
        What is going on right now?
      </Label>
      <p id={hintId} className="text-muted-foreground text-sm">
        Write it however you like - the way you would tell a friend. A few sentences is plenty.
        Include anything you think matters: income, housing, food, education, health care, children.
      </p>

      <Textarea
        id={fieldId}
        name="narrative"
        value={value}
        onChange={(event) => onChange(event.target.value.slice(0, NARRATIVE_MAX_LENGTH))}
        disabled={disabled}
        rows={8}
        maxLength={NARRATIVE_MAX_LENGTH}
        required
        aria-invalid={error !== null ? true : undefined}
        aria-describedby={describedBy}
        placeholder="For example: I lost my job last month and rent is due soon..."
        className="min-h-44 text-base"
      />

      <p id={errorId} role="alert" className="text-destructive text-sm empty:hidden">
        {error ?? ""}
      </p>

      {showCounter ? (
        <p id={counterId} aria-live="polite" className="text-muted-foreground text-sm">
          {remaining === 0
            ? "You have reached the length limit."
            : `About ${remaining} characters left.`}
        </p>
      ) : null}

      <details className="bg-muted/40 rounded-lg border p-3 text-sm">
        <summary className="touch-row flex cursor-pointer items-center font-medium">
          Not sure what to write? See an example
        </summary>
        <blockquote className="text-muted-foreground mt-2 border-l-2 pl-3 italic">
          {NARRATIVE_EXAMPLE}
        </blockquote>
        <p className="mt-2">
          That is enough for us to work with. You do not need to write it that way, and you do not
          need to include everything.
        </p>
      </details>

      <PrivacyHint>
        Never include your Social Security number, bank details or a password. We do not need them
        and we will never ask.
      </PrivacyHint>
    </div>
  );
}
