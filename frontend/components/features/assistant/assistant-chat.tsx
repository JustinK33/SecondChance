"use client";

import { RotateCcw, Send, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { InlineError } from "@/components/ui/error-state";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { useAnnouncer } from "@/lib/announcer";
import { useAssistant } from "@/lib/hooks";

import { AnswerCard } from "./answer-card";
import { StarterQuestions } from "./starter-questions";

/**
 * The question-and-answer screen.
 *
 * Calm on purpose. No typing indicator theatre, no streaming cursor, no avatar:
 * a question, a visible wait, and an answer that shows its sources. The
 * transcript is local to this component, as the hook documents, so nothing is
 * stored server-side and a reload is a fresh start.
 *
 * On failure the conversation is kept. The question stays on screen and the
 * retry re-asks it, because losing what you typed is the one thing that makes
 * people give up.
 */
export function AssistantChat({
  programId = null,
  runId = null,
}: {
  /** Scope retrieval to one program's sources. */
  programId?: string | null;
  /** Scope the answer to a run the user owns. Checked server-side. */
  runId?: string | null;
}): ReactNode {
  const assistant = useAssistant({ programId, runId });
  const { announce } = useAnnouncer();
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const answeredRef = useRef(0);

  const assistantTurns = assistant.turns.filter((turn) => turn.role === "assistant");
  const lastQuestion =
    [...assistant.turns].reverse().find((turn) => turn.role === "user")?.text ?? null;

  useEffect(() => {
    if (assistantTurns.length === answeredRef.current) return;
    answeredRef.current = assistantTurns.length;
    const latest = assistantTurns[assistantTurns.length - 1];
    if (latest === undefined) return;
    announce(
      latest.response?.is_grounded === true
        ? "An answer is ready, with its sources."
        : "An answer is ready. It has no verified source, so it is marked as general guidance.",
    );
  }, [assistantTurns, announce]);

  async function ask(question: string): Promise<void> {
    const text = question.trim();
    if (text.length === 0 || assistant.isPending) return;
    setDraft("");
    announce("Looking through the sources.");
    await assistant.ask(text);
    inputRef.current?.focus();
  }

  function onSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void ask(draft);
  }

  return (
    <div className="space-y-6">
      {assistant.turns.length === 0 ? (
        <StarterQuestions
          onPick={(question) => void ask(question)}
          disabled={assistant.isPending}
        />
      ) : null}

      <ol className="space-y-4">
        {assistant.turns.map((turn) =>
          turn.role === "user" ? (
            <li key={turn.id}>
              <div className="bg-muted flex items-start gap-2 rounded-lg p-3">
                <User aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
                <div>
                  <h3 className="sr-only">Your question</h3>
                  <p className="text-sm font-medium">{turn.text}</p>
                </div>
              </div>
            </li>
          ) : (
            <li key={turn.id}>
              <Card>
                <CardContent>
                  <h3 className="sr-only">Answer</h3>
                  {turn.response === null ? (
                    <p className="text-sm">{turn.text}</p>
                  ) : (
                    <AnswerCard response={turn.response} />
                  )}
                </CardContent>
              </Card>
            </li>
          ),
        )}
      </ol>

      {assistant.isPending ? (
        <div role="status" className="space-y-2 rounded-lg border p-4">
          <p className="text-muted-foreground text-sm">
            Looking through the sources we have on file...
          </p>
          <Skeleton className="h-4 w-full" aria-hidden="true" />
          <Skeleton className="h-4 w-5/6" aria-hidden="true" />
          <Skeleton className="h-4 w-2/3" aria-hidden="true" />
        </div>
      ) : null}

      {assistant.error !== null ? (
        <div className="space-y-2">
          <InlineError error={assistant.error} />
          <div className="flex flex-wrap gap-2">
            {lastQuestion !== null ? (
              <Button
                variant="outline"
                size="lg"
                className="touch-row"
                onClick={() => void ask(lastQuestion)}
              >
                <RotateCcw aria-hidden="true" className="size-4" />
                Ask that again
              </Button>
            ) : null}
            <Button variant="ghost" size="lg" className="touch-row" onClick={assistant.clearError}>
              Dismiss
            </Button>
          </div>
        </div>
      ) : null}

      <form onSubmit={onSubmit} className="space-y-3">
        <Label htmlFor="assistant-question">Your question</Label>
        <Textarea
          id="assistant-question"
          ref={inputRef}
          value={draft}
          rows={3}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="For example: what counts as proof of address?"
          aria-describedby="assistant-question-hint"
          disabled={assistant.isPending}
        />
        <p id="assistant-question-hint" className="text-muted-foreground text-xs">
          Answers come only from the sources we have on file, and every claim shows where it came
          from. Please do not include your Social Security number or bank details - no program here
          needs them.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button
            type="submit"
            size="lg"
            className="touch-row"
            disabled={assistant.isPending || draft.trim().length === 0}
          >
            <Send aria-hidden="true" className="size-4" />
            {assistant.isPending ? "Asking..." : "Ask"}
          </Button>
          {assistant.turns.length > 0 ? (
            <Button
              type="button"
              variant="ghost"
              size="lg"
              className="touch-row"
              onClick={assistant.reset}
              disabled={assistant.isPending}
            >
              Start a new conversation
            </Button>
          ) : null}
        </div>
      </form>
    </div>
  );
}
