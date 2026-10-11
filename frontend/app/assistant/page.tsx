import type { Metadata } from "next";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { AssistantChat } from "@/components/features/assistant";
import { PrivacyNote } from "@/components/ui/privacy-note";
import { LoadingText } from "@/components/ui/loading-state";

export const metadata: Metadata = {
  title: "Ask a question",
  description:
    "Answers drawn only from the sources we have on file, with the quote and the link for every claim.",
};

/**
 * Grounded question answering.
 *
 * `?program=<id>` scopes retrieval to one program's sources, which is how the
 * program detail page hands a question over. `?run=<id>` scopes an answer to a
 * run; ownership is checked server-side, so the id in the URL grants nothing.
 */
export default function AssistantPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}): ReactNode {
  return (
    <div className="page-container space-y-6 py-8 sm:py-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold sm:text-3xl">Ask a question</h1>
        <p className="text-muted-foreground reading-width">
          Every answer separates what the official sources actually say from our reading of them,
          and shows you the passage it came from. If we have no source for something, we say so
          rather than guessing.
        </p>
      </header>
      <PrivacyNote>
        <p>
          Questions are answered from the program sources we have on file. Nothing you type here is
          written into our logs, and no program needs your Social Security number or bank details.
        </p>
      </PrivacyNote>
      <Suspense fallback={<LoadingText lines={3} />}>
        <Chat searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

async function Chat({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}): Promise<ReactNode> {
  const query = await searchParams;
  const program = query.program;
  const run = query.run;
  return (
    <AssistantChat
      programId={typeof program === "string" && program.length > 0 ? program : null}
      runId={typeof run === "string" && run.length > 0 ? run : null}
    />
  );
}
