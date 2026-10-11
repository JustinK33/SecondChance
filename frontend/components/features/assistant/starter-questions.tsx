"use client";

import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

/**
 * The questions people actually arrive with.
 *
 * A blank chat box is intimidating and it is also a worse prompt: someone who
 * does not know what the system can answer tends to ask something it cannot.
 * These three are the ones the sources on file can genuinely answer, phrased the
 * way a person would say them.
 */
export const STARTER_QUESTIONS: readonly string[] = [
  "What does household income mean here?",
  "Do I need proof of address?",
  "What should I do first?",
] as const;

export function StarterQuestions({
  onPick,
  disabled = false,
}: {
  onPick: (question: string) => void;
  disabled?: boolean;
}): ReactNode {
  return (
    <section aria-labelledby="starter-questions" className="space-y-3">
      <h2 id="starter-questions" className="text-sm font-semibold">
        Not sure what to ask? Try one of these
      </h2>
      <ul className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
        {STARTER_QUESTIONS.map((question) => (
          <li key={question}>
            <Button
              variant="outline"
              size="lg"
              className="touch-row h-auto w-full py-2 text-left whitespace-normal sm:w-auto"
              disabled={disabled}
              onClick={() => onPick(question)}
            >
              {question}
            </Button>
          </li>
        ))}
      </ul>
    </section>
  );
}
