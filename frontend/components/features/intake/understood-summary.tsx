"use client";

/**
 * "Here is what we understood. Is that right?"
 *
 * Shown between intake and the actual run, because confirming a cheap mistake
 * now is far better than explaining an expensive one afterwards: a misread
 * household size changes which programs come back, and the user is the only
 * person who can catch it.
 *
 * It renders only what was understood. There is no status, no program and no
 * judgement here - nothing has been evaluated yet - and nothing about how the
 * reading was produced, which is not the user's problem.
 */

import type { ReactNode } from "react";

import { formatIncomeRange } from "@/lib/format";
import type { IntakeResponse } from "@/lib/types";

import { NEED_CHOICES, STATE_CHOICES, STUDENT_CHOICES } from "./options";

function stateLabel(code: string | null | undefined): string | null {
  if (code === null || code === undefined) return null;
  return STATE_CHOICES.find((choice) => choice.value === code)?.label ?? code;
}

function studentLabel(value: string | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  return STUDENT_CHOICES.find((choice) => choice.value === value)?.label ?? null;
}

function needLabels(values: readonly string[]): string[] {
  return values
    .map((value) => NEED_CHOICES.find((choice) => choice.value === value)?.label ?? null)
    .filter((label): label is string => label !== null);
}

export function UnderstoodSummary({ intake }: { intake: IntakeResponse }): ReactNode {
  const profile = intake.profile;
  const place = [stateLabel(profile.state), profile.county].filter(
    (part): part is string => typeof part === "string" && part.length > 0,
  );
  const needs = needLabels(profile.needs ?? []);
  const questionCount = intake.clarification_questions?.length ?? 0;

  const rows: readonly { label: string; value: string }[] = [
    { label: "Where you live", value: place.length > 0 ? place.join(", ") : "Not provided" },
    {
      label: "People in your household",
      value:
        profile.household_size === null || profile.household_size === undefined
          ? "Not provided"
          : String(profile.household_size),
    },
    { label: "Student", value: studentLabel(profile.student_status) ?? "Not provided" },
    {
      label: "Income",
      value: formatIncomeRange(
        profile.income_min_cents,
        profile.income_max_cents,
        profile.income_period,
      ),
    },
    { label: "What would help", value: needs.length > 0 ? needs.join(", ") : "Not provided" },
  ];

  return (
    <section aria-labelledby="understood-heading" className="space-y-4">
      <h2 id="understood-heading" className="text-xl font-semibold">
        Here is what we understood
      </h2>
      <p className="text-muted-foreground">
        Nothing has been checked yet. If any of this is wrong, go back and fix it - it changes which
        programs we look at.
      </p>

      <dl className="divide-y rounded-lg border">
        {rows.map((row) => (
          <div key={row.label} className="grid gap-1 p-3 sm:grid-cols-[14rem_1fr] sm:gap-4">
            <dt className="text-muted-foreground text-sm">{row.label}</dt>
            <dd className="text-base">{row.value}</dd>
          </div>
        ))}
      </dl>

      {intake.ambiguities !== undefined && intake.ambiguities.length > 0 ? (
        <div className="space-y-2">
          <h3 className="text-base font-semibold">We were not sure about this</h3>
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {intake.ambiguities.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <p className="text-muted-foreground text-sm">
        {questionCount > 0
          ? `We will ask you ${questionCount === 1 ? "one short question" : `${questionCount} short questions`} while we work, so we do not have to guess. You can skip any of them.`
          : "Anything we are missing, we will ask about rather than guess."}
      </p>
    </section>
  );
}
