"use client";

import { ExternalLink } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { pluralize } from "@/lib/format";
import type { ActionItem } from "@/lib/types";

import { ActionItemRow } from "./action-item-row";

export interface ProgramStepGroup {
  readonly programId: string;
  readonly programName: string;
  readonly items: ActionItem[];
  /** The application link from this program's `APPLY` step, when it has one. */
  readonly applicationUrl: string | null;
}

/**
 * Group a plan's items by program, in the backend's `sort_order`.
 *
 * Shared documents are excluded by key, because they are rendered once at the
 * top of the plan. Showing them here as well is exactly the repetition the
 * grouping exists to remove.
 */
export function groupByProgram(
  items: readonly ActionItem[],
  sharedKeys: ReadonlySet<string>,
): ProgramStepGroup[] {
  interface MutableGroup {
    programId: string;
    programName: string;
    items: ActionItem[];
    applicationUrl: string | null;
  }
  const groups = new Map<string, MutableGroup>();
  for (const item of [...items].sort((a, b) => a.sort_order - b.sort_order)) {
    if (sharedKeys.has(item.dedup_key)) continue;
    const url = typeof item.url === "string" && item.url.length > 0 ? item.url : null;
    const applicationUrl = item.kind === "APPLY" ? url : null;
    const existing = groups.get(item.program_id);
    if (existing === undefined) {
      groups.set(item.program_id, {
        programId: item.program_id,
        programName: item.program_name,
        items: [item],
        applicationUrl,
      });
      continue;
    }
    existing.items.push(item);
    if (existing.applicationUrl === null) existing.applicationUrl = applicationUrl;
  }
  return [...groups.values()];
}

/** One program's steps, under the program's name. */
export function ProgramSteps({
  group,
  isSaving,
  onToggle,
}: {
  group: ProgramStepGroup;
  isSaving: (itemId: string) => boolean;
  onToggle: (itemId: string, next: boolean) => void;
}): ReactNode {
  const done = group.items.filter((item) => item.is_complete === true).length;
  const headingId = `program-steps-${group.programId}`;

  return (
    <Card aria-labelledby={headingId}>
      <CardHeader>
        <CardTitle className="text-base">
          <h3 id={headingId}>{group.programName}</h3>
        </CardTitle>
        <p className="text-muted-foreground text-sm">
          {done} of {pluralize(group.items.length, "step")} done.
        </p>
        <div className="flex flex-wrap items-center gap-3 pt-1">
          <Link href={ROUTES.program(group.programId)} className="link touch-row text-sm">
            What this program is
          </Link>
          {group.applicationUrl !== null ? (
            <a
              href={group.applicationUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="link touch-row inline-flex items-center gap-1 text-sm font-medium"
            >
              Official application
              <ExternalLink aria-hidden="true" className="size-3.5" />
              <span className="sr-only">
                for {group.programName} (opens in a new tab, leaving SecondChance)
              </span>
            </a>
          ) : null}
        </div>
      </CardHeader>
      <CardContent>
        <ul className="-my-1">
          {group.items.map((item) => (
            <ActionItemRow
              key={item.id}
              item={item}
              isSaving={isSaving(item.id)}
              onToggle={onToggle}
            />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
