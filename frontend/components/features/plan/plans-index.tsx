"use client";

import { ArrowRight } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingCards } from "@/components/ui/loading-state";
import { formatDate, pluralize } from "@/lib/format";
import { usePlans } from "@/lib/hooks";
import { useAuth } from "@/lib/auth";

import { PlanProgress } from "./plan-progress";

/**
 * Every plan this person has, newest first.
 *
 * A plan is a snapshot of one run, so somebody who has described their
 * situation twice has two, and both are worth keeping: the older one records
 * what they were told last month. The list therefore shows the version and the
 * date rather than silently collapsing to the newest.
 *
 * Signed out, this says so plainly instead of showing an empty list, because
 * "you have no plans" and "we do not know who you are" are different facts.
 */
export function PlansIndex(): ReactNode {
  const auth = useAuth();
  const { plans, isLoading, error, refetch, isEmpty } = usePlans({ limit: 20 });

  if (auth.isLoading) {
    return <LoadingCards count={2} />;
  }

  if (!auth.isAuthenticated) {
    return (
      <EmptyState
        title="Your plans live with your answers"
        description="Once you have described your situation, the steps we suggest are saved here so you can work through them over several days."
        action={
          <Button asChild size="lg" className="touch-row">
            <Link href={ROUTES.start}>Start</Link>
          </Button>
        }
      />
    );
  }

  if (isLoading && plans.length === 0) {
    return <LoadingCards count={2} />;
  }

  if (error !== null && plans.length === 0) {
    return <ErrorState error={error} title="We could not load your plans" onRetry={refetch} />;
  }

  if (isEmpty) {
    return (
      <EmptyState
        title="No plans yet"
        description="Describe your situation and we will put together a checklist with the official link for each program."
        action={
          <Button asChild size="lg" className="touch-row">
            <Link href={ROUTES.start}>Start</Link>
          </Button>
        }
      />
    );
  }

  return (
    <ol className="space-y-4">
      {plans.map((plan) => {
        const items = plan.items ?? [];
        const completed = items.filter((item) => item.is_complete === true).length;
        const created = formatDate(plan.created_at);
        return (
          <li key={plan.id}>
            <Card className="gap-3">
              <CardHeader className="gap-1">
                <h2 className="text-base font-semibold">{plan.title}</h2>
                <p className="text-muted-foreground text-sm">
                  Version {plan.version}
                  {created === null ? "" : ` · built ${created}`} ·{" "}
                  {pluralize(items.length, "step")}
                </p>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="reading-width text-sm">{plan.summary}</p>
                <PlanProgress completed={completed} total={items.length} />
                <Button asChild variant="outline" size="lg" className="touch-row">
                  <Link href={ROUTES.plan(plan.id)}>
                    Open this plan
                    <ArrowRight aria-hidden="true" className="size-4" />
                  </Link>
                </Button>
              </CardContent>
            </Card>
          </li>
        );
      })}
    </ol>
  );
}
