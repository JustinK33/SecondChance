"use client";

import { ExternalLink, Loader2 } from "lucide-react";
import type { MouseEvent, ReactNode } from "react";

import { Checkbox } from "@/components/ui/checkbox";
import { humanizeEnum } from "@/lib/format";
import type { ActionItem, ActionItemKind } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * One step on the checklist.
 *
 * The checkbox is the only control in the row, named by the step's title
 * through `aria-labelledby` and described by its detail. The surrounding text
 * is *also* clickable as a convenience for a thumb on a phone - a plain `label`
 * cannot do that job here, because the primitive renders a `button` and
 * `label[for]` only forwards clicks to native form controls. Keyboard and
 * screen-reader users operate the one real control, so there is no duplicate
 * tab stop and nothing hidden.
 *
 * Completion is optimistic in `useActionItems`: the tick moves immediately and
 * rolls back if the server disagrees. This component only reflects state.
 *
 * Where a step has a link it goes to the organisation that runs the program and
 * says so. SecondChance never submits an application and never contacts anyone
 * on a user's behalf, so opening the official page is the only thing on offer.
 */
const KIND_WORDS: Record<ActionItemKind, string> = {
  DOCUMENT: "Document to find",
  APPLY: "Application",
  CONFIRM_INFO: "Check a detail",
  CONTACT: "Someone to contact",
  FOLLOW_UP: "Follow up",
  LEARN: "Read up",
};

function kindWord(kind: ActionItemKind): string {
  return KIND_WORDS[kind] ?? humanizeEnum(kind);
}

export function ActionItemRow({
  item,
  isSaving,
  onToggle,
  /** Shown when the row is read outside its program's own section. */
  showProgram = false,
  /** Extra context rendered under the row, inside the same list item. */
  footer,
}: {
  item: ActionItem;
  isSaving: boolean;
  onToggle: (itemId: string, next: boolean) => void;
  showProgram?: boolean;
  footer?: ReactNode;
}): ReactNode {
  const isComplete = item.is_complete === true;
  const titleId = `item-title-${item.id}`;
  const detailId = `item-detail-${item.id}`;
  const url = typeof item.url === "string" && item.url.length > 0 ? item.url : null;

  // The checkbox fires its own change event; swallow the bubbled click so one
  // tap is one toggle.
  function onRowClick(event: MouseEvent<HTMLDivElement>): void {
    const target = event.target as HTMLElement | null;
    if (target !== null && target.closest('[data-slot="checkbox"]') !== null) return;
    onToggle(item.id, !isComplete);
  }

  return (
    <li className="border-border border-b last:border-b-0">
      <div className="flex items-start gap-2">
        <div
          role="presentation"
          onClick={onRowClick}
          className="touch-row flex flex-1 cursor-pointer items-start gap-3 py-3"
        >
          <span className="touch-target -my-2 flex items-center justify-center">
            <Checkbox
              checked={isComplete}
              aria-labelledby={titleId}
              aria-describedby={detailId}
              onCheckedChange={(next) => onToggle(item.id, next === true)}
              className="size-5"
            />
          </span>
          <span className="flex-1 space-y-1">
            <span
              id={titleId}
              className={cn(
                "block text-sm font-medium",
                isComplete && "text-muted-foreground line-through",
              )}
            >
              {item.title}
            </span>
            <span id={detailId} className="text-muted-foreground block text-sm">
              {item.detail}
            </span>
            <span className="text-muted-foreground block text-xs">
              {kindWord(item.kind)}
              {showProgram ? ` · ${item.program_name}` : ""}
              {isComplete ? " · done" : ""}
            </span>
          </span>
        </div>
        <span className="flex shrink-0 items-center gap-2 py-3">
          {isSaving ? (
            <span className="text-muted-foreground flex items-center gap-1 text-xs">
              <Loader2
                aria-hidden="true"
                className="size-3.5 animate-spin motion-reduce:animate-none"
              />
              Saving
            </span>
          ) : null}
          {url !== null ? (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="link touch-target inline-flex items-center justify-center gap-1 px-2 text-sm font-medium"
            >
              Open
              <ExternalLink aria-hidden="true" className="size-3.5" />
              <span className="sr-only">
                {item.title} on {item.program_name}&apos;s official site (opens in a new tab,
                leaving SecondChance)
              </span>
            </a>
          ) : null}
        </span>
      </div>
      {footer === undefined ? null : <div className="-mt-1 pb-3 pl-11">{footer}</div>}
    </li>
  );
}
