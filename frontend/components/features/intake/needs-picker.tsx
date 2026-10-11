"use client";

/**
 * "What would help most?" as large tappable cards.
 *
 * A multi-select on a phone, used one-handed, by someone who may be tired. So:
 * whole-card targets well over the 44px minimum, two columns at most, the label
 * first and the explanation second, and a visible checked state that does not
 * rely on colour alone.
 *
 * Native checkboxes inside a `fieldset`/`legend`. The group name, the checked
 * state and the keyboard behaviour are then the browser's job rather than ours.
 */

import { Check } from "lucide-react";
import type { ReactNode } from "react";

import type { NeedCategory } from "@/lib/types";
import { cn } from "@/lib/utils";

import { NEED_CHOICES } from "./options";

export function NeedsPicker({
  value,
  onChange,
  idPrefix,
  disabled = false,
}: {
  value: readonly NeedCategory[];
  onChange: (next: NeedCategory[]) => void;
  idPrefix: string;
  disabled?: boolean;
}): ReactNode {
  const hintId = `${idPrefix}-needs-hint`;

  const toggle = (need: NeedCategory): void => {
    onChange(value.includes(need) ? value.filter((item) => item !== need) : [...value, need]);
  };

  return (
    <fieldset disabled={disabled} className="space-y-3">
      <legend className="text-base font-medium">What would help most? Pick any that apply.</legend>
      <p id={hintId} className="text-muted-foreground text-sm">
        This only decides what we look at first. You can pick several, or none.
      </p>

      <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {NEED_CHOICES.map((choice) => {
          const id = `${idPrefix}-need-${choice.value}`;
          const checked = value.includes(choice.value);
          return (
            <li key={choice.value}>
              <label
                htmlFor={id}
                className={cn(
                  "flex w-full cursor-pointer items-start gap-3 rounded-lg border p-4",
                  "min-h-[4rem] focus-within:ring-ring focus-within:ring-3",
                  checked ? "border-primary bg-secondary" : "bg-background hover:bg-muted/50",
                )}
              >
                <input
                  id={id}
                  type="checkbox"
                  name={`${idPrefix}-needs`}
                  value={choice.value}
                  checked={checked}
                  onChange={() => toggle(choice.value)}
                  aria-describedby={hintId}
                  className="mt-0.5 size-5 shrink-0"
                />
                <span className="min-w-0">
                  <span className="flex items-center gap-1.5 text-base font-medium">
                    {choice.label}
                    {checked ? <Check aria-hidden="true" className="size-4 shrink-0" /> : null}
                  </span>
                  {choice.help !== undefined ? (
                    <span className="text-muted-foreground block text-sm">{choice.help}</span>
                  ) : null}
                </span>
              </label>
            </li>
          );
        })}
      </ul>
    </fieldset>
  );
}
