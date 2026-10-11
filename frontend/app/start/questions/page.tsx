"use client";

/**
 * The clarification questions for a run that is waiting on an answer.
 *
 * The run id comes from the query string (`?run=...`) rather than the path,
 * because this screen is a step inside the intake flow rather than a view of
 * the run itself - `/runs/[runId]` owns that. A client component, since
 * `useSearchParams` is how the id arrives.
 */

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { ClarificationFlow } from "@/components/features/intake/clarification-flow";
import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { LoadingState } from "@/components/ui/loading-state";

function QuestionsInner(): ReactNode {
  const params = useSearchParams();
  const runId = params.get("run");

  if (runId === null || runId.length === 0) {
    return (
      <div className="space-y-4">
        <p>
          We do not know which search these questions belong to. That usually means the link was
          opened without its full address.
        </p>
        <Button asChild className="touch-row">
          <Link href={ROUTES.start}>Start again</Link>
        </Button>
      </div>
    );
  }

  return <ClarificationFlow runId={runId} />;
}

export default function ClarificationQuestionsPage(): ReactNode {
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
          <QuestionsInner />
        </Suspense>
      </div>
    </div>
  );
}
