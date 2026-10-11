import type { Metadata } from "next";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { ProgramDetailView } from "@/components/features/programs";
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
  title: "Program details",
  description:
    "What a program is, who runs it, what it asks for, and the official sources those requirements came from.",
};

/**
 * One program.
 *
 * `?run=<run id>` is optional. When it is present the page also shows that
 * run's evaluation of this program - the status, the evidence behind it and
 * what is still missing - so a card on the results screen can link straight
 * here without losing the reasoning. Ownership of the run is checked
 * server-side by the API, so an id in the URL grants nothing.
 */
export default function ProgramDetailPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}): ReactNode {
  return (
    <div className="page-container py-8 sm:py-10">
      <Suspense fallback={<LoadingCards count={3} />}>
        <Detail params={params} searchParams={searchParams} />
      </Suspense>
    </div>
  );
}

async function Detail({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}): Promise<ReactNode> {
  const [{ id }, query] = await Promise.all([params, searchParams]);
  const run = query.run;
  const runId = typeof run === "string" && run.length > 0 ? run : null;
  return <ProgramDetailView programId={id} runId={runId} />;
}
