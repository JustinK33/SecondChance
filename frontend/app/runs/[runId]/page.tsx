import type { Metadata } from "next";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { ClarificationFlow } from "@/components/features/intake";
import { LoadingState } from "@/components/ui/loading-state";

/**
 * Prerendering is blocking rather than streaming, for the same reason as the
 * results route: the shared header calls `usePathname()`, which cannot be known
 * while prerendering a dynamic route, and every byte of this page comes from a
 * client poll anyway.
 */
export const instant = false;

export const metadata: Metadata = {
  title: "A couple of questions",
};

/**
 * `ROUTES.run` - a run in flight, and the questions it is waiting on.
 *
 * This is the screen the results view links to with "answer the questions", and
 * the one a run sitting at `AWAITING_INPUT` needs. It is a thin shell:
 * `ClarificationFlow` decides every state from what the run reports, including
 * the case where there is nothing left to answer, in which case it points at the
 * results instead.
 */
export default function RunPage({ params }: { params: Promise<{ runId: string }> }): ReactNode {
  return (
    <div className="page-container py-8 sm:py-12">
      <div className="reading-width space-y-6">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            A couple of quick questions
          </h1>
          <p className="text-muted-foreground">
            These change what we can check for you. You can answer &ldquo;I am not sure&rdquo; to
            any of them and still get your list.
          </p>
        </header>
        <Suspense fallback={<LoadingState message="Loading your questions..." />}>
          <Flow params={params} />
        </Suspense>
      </div>
    </div>
  );
}

async function Flow({ params }: { params: Promise<{ runId: string }> }): Promise<ReactNode> {
  const { runId } = await params;
  return <ClarificationFlow runId={runId} />;
}
