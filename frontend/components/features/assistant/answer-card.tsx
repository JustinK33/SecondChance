import { BookOpen, CircleAlert, Lightbulb, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";

import { CitationLink } from "@/components/ui/source-citation";
import type { ChatResponse, Citation, ConfidenceLevel } from "@/lib/types";

/**
 * One grounded answer.
 *
 * The separation is the point. Two things are being shown and they carry
 * different weight:
 *
 * - *What the sources say* - claims with at least one citation. The excerpt is
 *   quoted, not paraphrased, and the link goes to the publisher.
 * - *Our reading* - uncited reasoning. It is useful, and it is not evidence, so
 *   it sits in its own labelled block and never borrows the sources' styling.
 *
 * An answer with no citations at all gets an explicit notice rather than being
 * presented as fact. The backend computes `is_grounded` and writes
 * `ungrounded_notice`; we render it, we do not decide it.
 *
 * Confidence is shown in words. "Low" with no explanation would be worse than
 * nothing, so each level says what to do about it.
 */
const CONFIDENCE_WORDS: Record<ConfidenceLevel, string> = {
  HIGH: "We found clear support for this in the sources on file.",
  MEDIUM: "The sources support most of this. Worth confirming the details officially.",
  LOW: "The sources only partly cover this. Please check the official page before relying on it.",
  UNKNOWN: "We cannot say how well the sources cover this. Treat it as a starting point.",
};

function citationsFor(sourceIds: readonly string[], citations: readonly Citation[]): Citation[] {
  return sourceIds
    .map((id) => citations.find((citation) => citation.source_id === id))
    .filter((citation): citation is Citation => citation !== undefined);
}

export function AnswerCard({ response }: { response: ChatResponse }): ReactNode {
  const facts = response.facts ?? [];
  const interpretation = response.interpretation ?? [];
  const citations = response.citations ?? [];
  const notice = response.ungrounded_notice ?? null;
  // The backend repeats the ungrounded notice as a caveat. It is already on
  // screen in its own block, and saying the same sentence twice reads like a
  // bug rather than emphasis.
  const caveats = (response.caveats ?? []).filter((caveat) => caveat !== notice);
  const nextSteps = response.suggested_next_steps ?? [];
  const isGrounded = response.is_grounded === true;

  return (
    <div className="space-y-4">
      <p className="reading-width text-sm leading-relaxed whitespace-pre-line">{response.answer}</p>

      {!isGrounded ? (
        <div
          role="note"
          className="border-notice-border bg-notice text-notice-foreground flex items-start gap-2 rounded-md border p-3 text-sm"
        >
          <TriangleAlert aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
          <p>
            {notice ??
              "We do not have a verified source for this answer, so please treat it as general guidance and check the program's official page."}
          </p>
        </div>
      ) : null}

      {facts.length > 0 ? (
        <section className="border-notice-border bg-notice/40 space-y-3 rounded-md border p-3">
          <h4 className="flex items-center gap-2 text-sm font-semibold">
            <BookOpen aria-hidden="true" className="size-4" />
            What the sources say
          </h4>
          <ul className="space-y-3">
            {facts.map((fact) => {
              const supporting = citationsFor(fact.source_ids, citations);
              return (
                <li key={fact.text} className="space-y-1 text-sm">
                  <p>{fact.text}</p>
                  {supporting.length > 0 ? (
                    <p className="text-muted-foreground text-xs">
                      From{" "}
                      {supporting.map((citation, index) => (
                        <span key={citation.source_id}>
                          {index > 0 ? ", " : ""}
                          <a
                            href={citation.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="link"
                          >
                            {citation.title}
                          </a>
                        </span>
                      ))}
                      <span className="sr-only"> (opens in a new tab)</span>
                    </p>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      {interpretation.length > 0 ? (
        <section className="border-border space-y-2 rounded-md border border-dashed p-3">
          <h4 className="flex items-center gap-2 text-sm font-semibold">
            <Lightbulb aria-hidden="true" className="size-4" />
            Our reading of that
          </h4>
          <p className="text-muted-foreground text-xs">
            This part is our interpretation, not a quote from a source.
          </p>
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {interpretation.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <p className="text-muted-foreground flex items-start gap-1.5 text-xs">
        <CircleAlert aria-hidden="true" className="mt-0.5 size-3.5 shrink-0" />
        <span>
          How sure we are: {confidenceLabel(response.confidence)}.{" "}
          {CONFIDENCE_WORDS[response.confidence] ?? CONFIDENCE_WORDS.UNKNOWN}
        </span>
      </p>

      {caveats.length > 0 ? (
        <section className="space-y-1">
          <h4 className="text-sm font-semibold">Worth knowing</h4>
          <ul className="text-muted-foreground list-disc space-y-1 pl-5 text-sm">
            {caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {nextSteps.length > 0 ? (
        <section className="space-y-1">
          <h4 className="text-sm font-semibold">What you could do next</h4>
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {nextSteps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {citations.length > 0 ? (
        <section className="space-y-3">
          <h4 className="text-sm font-semibold">Sources for this answer</h4>
          <ul className="space-y-3">
            {citations.map((citation) => (
              <li
                key={`${citation.source_id}-${citation.url}`}
                className="border-border rounded-md border p-3"
              >
                <CitationLink citation={citation} />
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function confidenceLabel(confidence: ConfidenceLevel): string {
  switch (confidence) {
    case "HIGH":
      return "fairly sure";
    case "MEDIUM":
      return "reasonably sure";
    case "LOW":
      return "not very sure";
    default:
      return "unsure";
  }
}
