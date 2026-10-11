import type { Metadata } from "next";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { ResultsView } from "@/components/features/results";
import { LoadingCards } from "@/components/ui/loading-state";

/**
 * Prerendering is blocking for this route rather than streaming.
 *
 * The shared header calls `usePathname()`, which cannot be known while
 * prerendering a dynamic route, so Next refuses to produce a static shell for
 * it. There is nothing to gain from one here - every byte on this page comes
 * from a client fetch - so the route renders at request time instead.
 */
export const instant = false;

export const metadata: Metadata = {
  title: "Your results",
  description:
    "Programs that may be worth exploring for your situation, with what we could check and what we still need.",
};

/**
 * The results of one run.
 *
 * A thin shell: every state - still working, waiting on an answer, finished,
 * failed - is decided by `ResultsView` from what `useWorkflow` reports, because
 * the run is polled in the browser and there is nothing useful to render on the
 * server.
 *
 * `params` is awaited inside a `Suspense` boundary, which is what lets the rest
 * of the page prerender.
 */
export default function ResultsPage({ params }: { params: Promise<{ runId: string }> }): ReactNode {
  return (
    <div className="page-container py-8 sm:py-10">
      <Suspense fallback={<LoadingCards count={3} />}>
        <Results params={params} />
      </Suspense>
    </div>
  );
}

async function Results({ params }: { params: Promise<{ runId: string }> }): Promise<ReactNode> {
  const { runId } = await params;
  return <ResultsView runId={runId} />;
}
