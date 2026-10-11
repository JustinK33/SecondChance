"use client";

/**
 * Income as a band, never as a figure.
 *
 * This is the field people stall on, so three things are done deliberately:
 *
 * 1. It asks for a range. Every rule the engine can check compares income
 *    against a threshold, so a band is genuinely enough to evaluate against.
 *    Asking for a precise number would collect more than we can use.
 * 2. It says so, next to the field, in one sentence. The explanation is the
 *    reason someone answers at all.
 * 3. It is skippable. Leaving it blank means "not stated", which keeps affected
 *    programs at "need a bit more info" rather than ruling anything out.
 *
 * Native radios inside a `fieldset`/`legend` rather than a custom control: the
 * grouping, the arrow-key behaviour and the screen-reader announcement all come
 * for free and cannot be got subtly wrong.
 */

import type { ReactNode } from "react";

import type { IncomePeriod } from "@/lib/types";
import { cn } from "@/lib/utils";

import { incomeBandsFor } from "./options";

const PERIODS: readonly { value: IncomePeriod; label: string }[] = [
  { value: "MONTHLY", label: "Per month" },
  { value: "ANNUAL", label: "Per year" },
];

export function IncomeRangeField({
  period,
  bandId,
  onPeriodChange,
  onBandChange,
  idPrefix,
  disabled = false,
}: {
  period: IncomePeriod;
  /** The chosen band's id, or `null` for "not stated". */
  bandId: string | null;
  onPeriodChange: (next: IncomePeriod) => void;
  onBandChange: (next: string | null) => void;
  idPrefix: string;
  disabled?: boolean;
}): ReactNode {
  const bands = incomeBandsFor(period);
  const explainId = `${idPrefix}-income-why`;

  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-base font-medium">
        Roughly how much money does your household have coming in?
      </legend>

      <p id={explainId} className="text-muted-foreground reading-width text-sm">
        A range is enough. Programs compare your income against a cut-off line, so we only need to
        know which side of the line you are on - we never need the exact figure, and we do not ask
        for it. Count everyone you live with and share money with, before anything is taken out. If
        you are not sure, pick the closest one or leave this blank.
      </p>

      <div
        role="radiogroup"
        aria-label="Is that per month or per year?"
        className="flex flex-wrap gap-2"
      >
        {PERIODS.map((option) => (
          <label
            key={option.value}
            className={cn(
              "touch-row inline-flex cursor-pointer items-center gap-2 rounded-lg border px-4 py-2 text-sm",
              "focus-within:ring-ring focus-within:ring-3",
              period === option.value ? "border-primary bg-secondary font-medium" : "bg-background",
            )}
          >
            <input
              type="radio"
              name={`${idPrefix}-income-period`}
              value={option.value}
              checked={period === option.value}
              onChange={() => onPeriodChange(option.value)}
              className="size-4"
            />
            {option.label}
          </label>
        ))}
      </div>

      <div className="space-y-2">
        {bands.map((band) => {
          const id = `${idPrefix}-band-${band.id}`;
          return (
            <label
              key={band.id}
              htmlFor={id}
              className={cn(
                "touch-row flex w-full cursor-pointer items-center gap-3 rounded-lg border p-3 text-base",
                "focus-within:ring-ring focus-within:ring-3",
                bandId === band.id ? "border-primary bg-secondary" : "bg-background",
              )}
            >
              <input
                id={id}
                type="radio"
                name={`${idPrefix}-income-band`}
                value={band.id}
                checked={bandId === band.id}
                onChange={() => onBandChange(band.id)}
                aria-describedby={explainId}
                className="size-5 shrink-0"
              />
              <span>{band.label}</span>
            </label>
          );
        })}

        <label
          htmlFor={`${idPrefix}-band-none`}
          className={cn(
            "touch-row flex w-full cursor-pointer items-center gap-3 rounded-lg border border-dashed p-3 text-base",
            "focus-within:ring-ring focus-within:ring-3",
            bandId === null ? "border-primary bg-secondary" : "bg-background",
          )}
        >
          <input
            id={`${idPrefix}-band-none`}
            type="radio"
            name={`${idPrefix}-income-band`}
            value=""
            checked={bandId === null}
            onChange={() => onBandChange(null)}
            className="size-5 shrink-0"
          />
          <span>
            I would rather not say
            <span className="text-muted-foreground block text-sm">
              We will still show you programs. Some will say we need a bit more information.
            </span>
          </span>
        </label>
      </div>
    </fieldset>
  );
}
