import type { Metadata } from "next";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { ActionPlanView } from "@/components/features/plan";
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
  title: "Your action plan",
  description:
    "The documents to gather once, then the steps for each program with its official application link.",
};

/** One plan: the checklist dashboard for a run. */
export default function PlanPage({ params }: { params: Promise<{ planId: string }> }): ReactNode {
  return (
    <div className="page-container py-8 sm:py-10">
      <Suspense fallback={<LoadingCards count={2} />}>
        <Plan params={params} />
      </Suspense>
    </div>
  );
}

async function Plan({ params }: { params: Promise<{ planId: string }> }): Promise<ReactNode> {
  const { planId } = await params;
  return <ActionPlanView planId={planId} />;
}
