"use client";

/**
 * One clarification question, rendered the way the question asked to be.
 *
 * The backend declares an `input_kind` and, for the select kinds, the exact
 * options and their machine values. This component renders that declaration and
 * invents nothing: no extra options, no reordering, no re-wording. If a question
 * type ever arrives that we do not know how to draw, it falls back to a plain
 * text field rather than silently dropping the question.
 *
 * Every kind offers the same escape hatch: "I'm not sure". That is a real
 * answer, not a missing one - it records the field as unknown, stops us asking
 * again, and leaves the affected programs at "need a bit more info". Forcing a
 * guess instead would put a number into an eligibility calculation that the user
 * never actually claimed.
 */

import type { ReactNode } from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ClarificationAnswer, ClarificationQuestion } from "@/lib/types";
import { cn } from "@/lib/utils";

/** The sentinel radio value for "I'm not sure" in a single-select question. */
const NOT_SURE = "__not_sure__";

const CHOICE_CLASS =
  "touch-row flex w-full cursor-pointer items-start gap-3 rounded-lg border p-4 text-base focus-within:ring-ring focus-within:ring-3";

function answerFor(
  question: ClarificationQuestion,
  value: ClarificationAnswer["value"],
): ClarificationAnswer {
  return { question_id: question.id, field: question.field, value, skipped: false };
}

function skipAnswer(question: ClarificationQuestion): ClarificationAnswer {
  return { question_id: question.id, field: question.field, value: null, skipped: true };
}

/** The current value as a string array, for a multi-select. */
function selectedValues(answer: ClarificationAnswer | null): string[] {
  if (answer === null || answer.skipped === true) return [];
  return Array.isArray(answer.value) ? [...answer.value] : [];
}

function numberValue(answer: ClarificationAnswer | null): string {
  if (answer === null || answer.skipped === true) return "";
  return typeof answer.value === "number" ? String(answer.value) : "";
}

function rangeBound(answer: ClarificationAnswer | null, bound: "min" | "max"): string {
  if (answer === null || answer.skipped === true) return "";
  const value = answer.value;
  if (value === null || typeof value !== "object" || Array.isArray(value)) return "";
  const bounded = (value as { min?: number | null; max?: number | null })[bound];
  return typeof bounded === "number" ? String(bounded) : "";
}

function textValue(answer: ClarificationAnswer | null): string {
  if (answer === null || answer.skipped === true) return "";
  return typeof answer.value === "string" ? answer.value : "";
}

/** Parse a numeric field, treating an empty box as "nothing entered". */
function parseNumber(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export function ClarificationQuestionCard({
  question,
  answer,
  onChange,
  error,
  idPrefix,
  disabled = false,
}: {
  question: ClarificationQuestion;
  /** The answer so far, or `null` if untouched. */
  answer: ClarificationAnswer | null;
  onChange: (next: ClarificationAnswer) => void;
  /** Plain-language problem with the current answer, or `null`. */
  error: string | null;
  idPrefix: string;
  disabled?: boolean;
}): ReactNode {
  const groupId = `${idPrefix}-${question.id}`;
  const whyId = `${groupId}-why`;
  const errorId = `${groupId}-error`;
  const isSkipped = answer?.skipped === true;
  const describedBy = error !== null ? `${whyId} ${errorId}` : whyId;

  const options = question.options ?? [];

  const notSureRow = (
    <label
      htmlFor={`${groupId}-not-sure`}
      className={cn(
        CHOICE_CLASS,
        "border-dashed",
        isSkipped ? "border-primary bg-secondary" : "bg-background",
      )}
    >
      <input
        id={`${groupId}-not-sure`}
        type={question.input_kind === "single_select" ? "radio" : "checkbox"}
        name={question.input_kind === "single_select" ? groupId : `${groupId}-not-sure`}
        value={NOT_SURE}
        checked={isSkipped}
        disabled={disabled}
        onChange={(event) => {
          if (question.input_kind === "single_select" || event.target.checked) {
            onChange(skipAnswer(question));
          } else {
            onChange(answerFor(question, null));
          }
        }}
        aria-describedby={describedBy}
        className="mt-0.5 size-5 shrink-0"
      />
      <span>
        I am not sure
        <span className="text-muted-foreground block text-sm">
          We will leave this blank rather than guess. You will still get your list.
        </span>
      </span>
    </label>
  );

  let control: ReactNode;

  switch (question.input_kind) {
    case "single_select": {
      control = (
        <div role="group" aria-describedby={describedBy} className="space-y-2">
          {options.map((option) => {
            const id = `${groupId}-${option.value}`;
            const checked = !isSkipped && answer?.value === option.value;
            return (
              <label
                key={option.value}
                htmlFor={id}
                className={cn(
                  CHOICE_CLASS,
                  checked ? "border-primary bg-secondary" : "bg-background",
                )}
              >
                <input
                  id={id}
                  type="radio"
                  name={groupId}
                  value={option.value}
                  checked={checked}
                  disabled={disabled}
                  onChange={() => onChange(answerFor(question, option.value))}
                  aria-describedby={describedBy}
                  className="mt-0.5 size-5 shrink-0"
                />
                <span>
                  {option.label}
                  {option.help_text !== null && option.help_text !== undefined ? (
                    <span className="text-muted-foreground block text-sm">{option.help_text}</span>
                  ) : null}
                </span>
              </label>
            );
          })}
          {notSureRow}
        </div>
      );
      break;
    }

    case "multi_select": {
      const selected = selectedValues(answer);
      control = (
        <div role="group" aria-describedby={describedBy} className="space-y-2">
          {options.map((option) => {
            const id = `${groupId}-${option.value}`;
            const checked = selected.includes(option.value);
            return (
              <label
                key={option.value}
                htmlFor={id}
                className={cn(
                  CHOICE_CLASS,
                  checked ? "border-primary bg-secondary" : "bg-background",
                )}
              >
                <input
                  id={id}
                  type="checkbox"
                  name={groupId}
                  value={option.value}
                  checked={checked}
                  disabled={disabled}
                  onChange={() => {
                    const next = checked
                      ? selected.filter((item) => item !== option.value)
                      : [...selected, option.value];
                    onChange(answerFor(question, next));
                  }}
                  aria-describedby={describedBy}
                  className="mt-0.5 size-5 shrink-0"
                />
                <span>
                  {option.label}
                  {option.help_text !== null && option.help_text !== undefined ? (
                    <span className="text-muted-foreground block text-sm">{option.help_text}</span>
                  ) : null}
                </span>
              </label>
            );
          })}
          {notSureRow}
        </div>
      );
      break;
    }

    case "number": {
      const inputId = `${groupId}-number`;
      control = (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Input
              id={inputId}
              type="number"
              inputMode="numeric"
              value={numberValue(answer)}
              min={question.min_value ?? undefined}
              max={question.max_value ?? undefined}
              disabled={disabled || isSkipped}
              onChange={(event) => {
                const parsed = parseNumber(event.target.value);
                onChange(answerFor(question, parsed));
              }}
              aria-invalid={error !== null ? true : undefined}
              aria-describedby={describedBy}
              aria-labelledby={`${groupId}-label`}
              className="touch-row h-12 max-w-40 text-base"
            />
            {question.unit !== null && question.unit !== undefined ? (
              <span className="text-muted-foreground text-base">{question.unit}</span>
            ) : null}
          </div>
          {notSureRow}
        </div>
      );
      break;
    }

    case "range": {
      control = (
        <div className="space-y-3">
          <div className="flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <Label htmlFor={`${groupId}-min`}>At least</Label>
              <Input
                id={`${groupId}-min`}
                type="number"
                inputMode="numeric"
                value={rangeBound(answer, "min")}
                min={question.min_value ?? undefined}
                max={question.max_value ?? undefined}
                disabled={disabled || isSkipped}
                onChange={(event) =>
                  onChange(
                    answerFor(question, {
                      min: parseNumber(event.target.value),
                      max: parseNumber(rangeBound(answer, "max")),
                    }),
                  )
                }
                aria-describedby={describedBy}
                className="touch-row h-12 max-w-40 text-base"
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor={`${groupId}-max`}>At most</Label>
              <Input
                id={`${groupId}-max`}
                type="number"
                inputMode="numeric"
                value={rangeBound(answer, "max")}
                min={question.min_value ?? undefined}
                max={question.max_value ?? undefined}
                disabled={disabled || isSkipped}
                onChange={(event) =>
                  onChange(
                    answerFor(question, {
                      min: parseNumber(rangeBound(answer, "min")),
                      max: parseNumber(event.target.value),
                    }),
                  )
                }
                aria-describedby={describedBy}
                className="touch-row h-12 max-w-40 text-base"
              />
            </div>
            {question.unit !== null && question.unit !== undefined ? (
              <span className="text-muted-foreground pb-3 text-base">{question.unit}</span>
            ) : null}
          </div>
          {notSureRow}
        </div>
      );
      break;
    }

    default: {
      // `text`, and anything the backend adds that this build predates.
      const inputId = `${groupId}-text`;
      control = (
        <div className="space-y-3">
          <Input
            id={inputId}
            type="text"
            value={textValue(answer)}
            disabled={disabled || isSkipped}
            onChange={(event) => onChange(answerFor(question, event.target.value))}
            aria-invalid={error !== null ? true : undefined}
            aria-describedby={describedBy}
            aria-labelledby={`${groupId}-label`}
            className="touch-row h-12 text-base"
          />
          {notSureRow}
        </div>
      );
      break;
    }
  }

  return (
    <div className="space-y-4">
      <h2 id={`${groupId}-label`} className="text-xl font-semibold">
        {question.question}
      </h2>
      <p id={whyId} className="bg-muted text-muted-foreground rounded-lg border p-3 text-sm">
        <span className="text-foreground font-medium">Why we are asking: </span>
        {question.why_it_matters}
      </p>
      {control}
      <p id={errorId} role="alert" className="text-destructive text-sm empty:hidden">
        {error ?? ""}
      </p>
    </div>
  );
}
