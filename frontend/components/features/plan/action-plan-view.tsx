"use client";

import { RefreshCw } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo } from "react";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState, InlineError } from "@/components/ui/error-state";
import { LoadingCards } from "@/components/ui/loading-state";
import { formatDate } from "@/lib/format";
import { useActionItems, usePlan, useReevaluate } from "@/lib/hooks";
import { useAnnouncer } from "@/lib/announcer";
import type { ActionItem } from "@/lib/types";

import { PlanProgress } from "./plan-progress";
import { ProgramSteps, groupByProgram } from "./program-steps";
import { SharedDocuments } from "./shared-documents";

/**
 * The action plan: one checklist dashboard for a run.
 *
 * Structure, in the order it is useful:
 *
 * 1. Progress, so the work already done is visible before the work remaining.
 * 2. Shared documents, collapsed once. Five programs wanting proof of address
 *    is one errand, and presenting it as five is how a plan starts to feel
 *    impossible.
 * 3. Per-program steps, each with the official application link.
 *
 * The plan's `version` is shown, because a plan is a snapshot of a profile at a
 * point in time: if the profile has changed, the honest move is to say which
 * version this is and offer to rebuild it. "Refresh my plan" re-runs matching
 * from the profile already on file - nobody retypes anything - and lands on the
 * new run so the rebuild is visible rather than silent.
 *
 * Nothing here offers to submit an application or to contact an organisation on
 * the user's behalf, because SecondChance does neither.
 */
export function ActionPlanView({ planId }: { planId: string }): ReactNode {
  const router = useRouter();
  const { plan, isLoading, error, refetch } = usePlan(planId);
  const items = useActionItems(plan);
  const reevaluation = useReevaluate();
  const { announce } = useAnnouncer();

  const sharedKeys = useMemo<ReadonlySet<string>>(
    () => new Set(items.sharedDocuments.map((document) => document.dedup_key)),
    [items.sharedDocuments],
  );
  const itemsByKey = useMemo<ReadonlyMap<string, ActionItem>>(
    () => new Map(items.items.map((item) => [item.dedup_key, item])),
    [items.items],
  );
  const groups = useMemo(() => groupByProgram(items.items, sharedKeys), [items.items, sharedKeys]);

  async function onToggle(itemId: string, next: boolean): Promise<void> {
    const total = items.totalCount;
    const expected = Math.max(0, Math.min(total, items.completedCount + (next ? 1 : -1)));
    const ok = await items.setComplete(itemId, next);
    if (!ok) {
      announce("We could not save that step. It has been put back as it was.", "assertive");
      return;
    }
    announce(
      next
        ? expected === total
          ? `Step marked done. That is all ${total} steps.`
          : `Step marked done. ${expected} of ${total} done.`
        : `Step put back. ${expected} of ${total} done.`,
    );
  }

  async function onRefresh(): Promise<void> {
    if (plan === null) return;
    const response = await reevaluation.reevaluate(plan.run_id, {});
    if (response !== null) router.push(ROUTES.results(response.run_id));
  }

  if (isLoading && plan === null) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">Your plan</h1>
        <LoadingCards count={2} />
      </div>
    );
  }

  if (plan === null) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-semibold sm:text-3xl">Your plan</h1>
        <ErrorState error={error} title="We could not open that plan" onRetry={refetch} />
        <Button asChild variant="outline" size="lg" className="touch-row">
          <Link href={ROUTES.plans}>See your plans</Link>
        </Button>
      </div>
    );
  }

  const created = formatDate(plan.created_at);

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <h1 className="text-2xl font-semibold sm:text-3xl">{plan.title}</h1>
        <p className="text-muted-foreground reading-width">{plan.summary}</p>
        <p className="text-muted-foreground text-xs">
          Plan version {plan.version}
          {created === null ? "" : ` · built ${created}`}
        </p>
        <PlanProgress completed={items.completedCount} total={items.totalCount} />
      </header>

      {items.error !== null ? (
        <div className="space-y-2">
          <InlineError error={items.error} />
          <Button variant="ghost" size="lg" className="touch-row" onClick={items.clearError}>
            Dismiss
          </Button>
        </div>
      ) : null}

      {items.totalCount === 0 ? (
        <EmptyState
          title="This plan has no steps yet"
          description="That can happen when a run found nothing it could check. Browsing the programs directly is usually the fastest next move."
          action={
            <Button asChild variant="outline" size="lg" className="touch-row">
              <Link href={ROUTES.programs}>Browse programs</Link>
            </Button>
          }
        />
      ) : (
        <>
          <SharedDocuments
            documents={items.sharedDocuments}
            itemsByKey={itemsByKey}
            isSaving={items.isSaving}
            onToggle={(itemId, next) => void onToggle(itemId, next)}
          />

          <section aria-labelledby="per-program-steps" className="space-y-4">
            <h2 id="per-program-steps" className="text-xl font-semibold">
              Step by step, program by program
            </h2>
            {groups.map((group) => (
              <ProgramSteps
                key={group.programId}
                group={group}
                isSaving={items.isSaving}
                onToggle={(itemId, next) => void onToggle(itemId, next)}
              />
            ))}
          </section>
        </>
      )}

      <section aria-labelledby="refresh-plan" className="space-y-2 border-t pt-6">
        <h2 id="refresh-plan" className="text-base font-semibold">
          Changed something about your situation?
        </h2>
        <p className="text-muted-foreground reading-width text-sm">
          Rebuilding checks the programs again using the answers already on file and keeps the steps
          you have ticked off. You will not have to retype anything.
        </p>
        <Button
          variant="outline"
          size="lg"
          className="touch-row"
          onClick={() => void onRefresh()}
          disabled={reevaluation.isPending}
        >
          <RefreshCw aria-hidden="true" className="size-4" />
          {reevaluation.isPending ? "Rebuilding..." : "Refresh my plan"}
        </Button>
        {reevaluation.error !== null ? <InlineError error={reevaluation.error} /> : null}
      </section>

      <p className="text-muted-foreground reading-width text-xs">
        SecondChance cannot apply for you or contact a program on your behalf. Every application
        link goes to the organisation that runs the program.
      </p>
    </div>
  );
}
