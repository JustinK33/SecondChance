"use client";

/**
 * The optional structured questions that sit beside the narrative.
 *
 * Every one of them is optional and is labelled as optional. They exist because
 * a few fields - where you live, how many people you are feeding, whether you
 * are a student - decide which programs are even worth checking, and getting
 * them from a dropdown is more reliable than hoping they were mentioned in
 * passing. Anything left blank is asked about later rather than assumed.
 *
 * Native `select` and `input` elements rather than custom controls. On a phone
 * a native select opens the platform picker, which is the control the user
 * already knows how to operate, and it is keyboard- and screen-reader-correct
 * without any work from us.
 */

import type { ReactNode } from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { IncomePeriod, NeedCategory, StudentStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

import { IncomeRangeField } from "./income-range-field";
import { NeedsPicker } from "./needs-picker";
import { NJ_COUNTIES, STATE_CHOICES, STUDENT_CHOICES } from "./options";
import {
  HOUSEHOLD_SIZE_MAX,
  HOUSEHOLD_SIZE_MIN,
  type StructuredAnswers,
} from "./structured-answers";

/** Sentinel for "not stated" in a native select, which cannot hold `null`. */
const UNSET = "";

const SELECT_CLASS =
  "touch-row w-full rounded-lg border border-input bg-background px-3 py-2 text-base outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

export function StructuredHelpers({
  value,
  onChange,
  idPrefix,
  disabled = false,
  householdError,
}: {
  value: StructuredAnswers;
  /** Receives a partial patch; the parent owns the merge. */
  onChange: (patch: Partial<StructuredAnswers>) => void;
  idPrefix: string;
  disabled?: boolean;
  /** Plain-language problem with the household size, or `null`. */
  householdError?: string | null;
}): ReactNode {
  const stateId = `${idPrefix}-state`;
  const countyId = `${idPrefix}-county`;
  const countyHintId = `${countyId}-hint`;
  const householdId = `${idPrefix}-household`;
  const householdHintId = `${householdId}-hint`;
  const householdErrorId = `${householdId}-error`;
  const studentId = `${idPrefix}-student`;
  const studentHintId = `${studentId}-hint`;

  const error = householdError ?? null;

  return (
    <div className="space-y-8">
      <div className="grid gap-6 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor={stateId} className="text-base">
            Which state do you live in?
          </Label>
          <select
            id={stateId}
            name="state"
            disabled={disabled}
            value={value.state ?? UNSET}
            onChange={(event) =>
              onChange({ state: event.target.value === UNSET ? null : event.target.value })
            }
            className={SELECT_CLASS}
          >
            <option value={UNSET}>Prefer not to say</option>
            {STATE_CHOICES.map((choice) => (
              <option key={choice.value} value={choice.value}>
                {choice.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <Label htmlFor={countyId} className="text-base">
            Which county? <span className="text-muted-foreground font-normal">(optional)</span>
          </Label>
          <p id={countyHintId} className="text-muted-foreground text-sm">
            Some help is run county by county. We never need your street address.
          </p>
          {value.state === "NJ" ? (
            <select
              id={countyId}
              name="county"
              disabled={disabled}
              value={value.county ?? UNSET}
              onChange={(event) =>
                onChange({ county: event.target.value === UNSET ? null : event.target.value })
              }
              aria-describedby={countyHintId}
              className={SELECT_CLASS}
            >
              <option value={UNSET}>I am not sure</option>
              {NJ_COUNTIES.map((county) => (
                <option key={county} value={county}>
                  {county}
                </option>
              ))}
            </select>
          ) : (
            <Input
              id={countyId}
              name="county"
              type="text"
              autoComplete="address-level2"
              disabled={disabled}
              value={value.county ?? ""}
              onChange={(event) =>
                onChange({
                  county: event.target.value.trim().length === 0 ? null : event.target.value,
                })
              }
              aria-describedby={countyHintId}
              className="touch-row h-12 text-base"
            />
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor={householdId} className="text-base">
            How many people live with you, including you?{" "}
            <span className="text-muted-foreground font-normal">(optional)</span>
          </Label>
          <p id={householdHintId} className="text-muted-foreground text-sm">
            Count everyone you share food and bills with. A rough number is fine.
          </p>
          <Input
            id={householdId}
            name="household_size"
            type="number"
            inputMode="numeric"
            min={HOUSEHOLD_SIZE_MIN}
            max={HOUSEHOLD_SIZE_MAX}
            step={1}
            disabled={disabled}
            value={value.householdSize === null ? "" : String(value.householdSize)}
            onChange={(event) => {
              const raw = event.target.value.trim();
              if (raw.length === 0) {
                onChange({ householdSize: null });
                return;
              }
              const parsed = Number.parseInt(raw, 10);
              onChange({ householdSize: Number.isNaN(parsed) ? null : parsed });
            }}
            aria-invalid={error !== null ? true : undefined}
            aria-describedby={
              error !== null ? `${householdHintId} ${householdErrorId}` : householdHintId
            }
            className={cn("touch-row h-12 max-w-32 text-base")}
          />
          <p id={householdErrorId} role="alert" className="text-destructive text-sm empty:hidden">
            {error ?? ""}
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor={studentId} className="text-base">
            Are you a student? <span className="text-muted-foreground font-normal">(optional)</span>
          </Label>
          <p id={studentHintId} className="text-muted-foreground text-sm">
            Students can get help that other people cannot, and the other way round.
          </p>
          <select
            id={studentId}
            name="student_status"
            disabled={disabled}
            value={value.studentStatus ?? UNSET}
            onChange={(event) =>
              onChange({
                studentStatus:
                  event.target.value === UNSET ? null : (event.target.value as StudentStatus),
              })
            }
            aria-describedby={studentHintId}
            className={SELECT_CLASS}
          >
            <option value={UNSET}>Prefer not to say</option>
            {STUDENT_CHOICES.map((choice) => (
              <option key={choice.value} value={choice.value}>
                {choice.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <IncomeRangeField
        period={value.incomePeriod}
        bandId={value.incomeBandId}
        onPeriodChange={(next: IncomePeriod) =>
          // The band ids differ per period, so a stale selection has to go.
          onChange({ incomePeriod: next, incomeBandId: null })
        }
        onBandChange={(next: string | null) => onChange({ incomeBandId: next })}
        idPrefix={idPrefix}
        disabled={disabled}
      />

      <NeedsPicker
        value={value.needs}
        onChange={(next: NeedCategory[]) => onChange({ needs: next })}
        idPrefix={idPrefix}
        disabled={disabled}
      />
    </div>
  );
}
