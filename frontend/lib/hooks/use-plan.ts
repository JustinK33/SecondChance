"use client";

/**
 * Reading an action plan.
 *
 * A plan is immutable apart from the completion flag on its items, which is why
 * mutation lives in `use-action-items` and this file only reads.
 */

import { useMemo } from "react";

import { getPlan, listPlans } from "../api-client";
import type { ActionPlan } from "../types";
import { useQuery } from "./use-async";
import type { QueryResult } from "./use-async";

export interface PlanResult extends QueryResult<ActionPlan> {
  readonly plan: ActionPlan | null;
}

export function usePlan(planId: string | null | undefined): PlanResult {
  const id = typeof planId === "string" && planId.length > 0 ? planId : null;
  const result = useQuery((signal) => getPlan(id as string, { signal }), [id], {
    enabled: id !== null,
  });
  return useMemo(() => ({ ...result, plan: result.data }), [result]);
}

export interface PlansResult extends QueryResult<{ items: ActionPlan[]; total: number }> {
  readonly plans: ActionPlan[];
  readonly total: number;
  readonly isEmpty: boolean;
}

export function usePlans(query?: { limit?: number; offset?: number }): PlansResult {
  const key = JSON.stringify(query ?? {});
  const result = useQuery(
    (signal) =>
      listPlans(query, { signal }).then((page) => ({
        items: page.items ?? [],
        total: page.total,
      })),
    [key],
  );
  return useMemo(
    () => ({
      ...result,
      plans: result.data?.items ?? [],
      total: result.data?.total ?? 0,
      isEmpty: result.data !== null && result.data.items.length === 0,
    }),
    [result],
  );
}
