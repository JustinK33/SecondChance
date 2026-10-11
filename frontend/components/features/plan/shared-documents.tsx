"use client";

import { FolderOpen } from "lucide-react";
import type { ReactNode } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ActionItem, SharedDocument } from "@/lib/types";

import { ActionItemRow } from "./action-item-row";

/**
 * The documents several programs ask for, gathered once.
 *
 * Without this group, "proof of a New Jersey address" appears under every
 * program that wants it and the list looks five times as long as the errand
 * actually is. The backend has already done the collapsing - items are keyed by
 * a canonical subject, so one `dedup_key` is one real-world document - and this
 * component renders that group and names the programs it covers, so nobody
 * wonders whether the single row is the whole story.
 *
 * Ticking a shared document ticks the one underlying item, which is why the
 * per-program sections must exclude these keys.
 */
export function SharedDocuments({
  documents,
  itemsByKey,
  isSaving,
  onToggle,
}: {
  documents: readonly SharedDocument[];
  /** The plan's items, indexed by `dedup_key`. */
  itemsByKey: ReadonlyMap<string, ActionItem>;
  isSaving: (itemId: string) => boolean;
  onToggle: (itemId: string, next: boolean) => void;
}): ReactNode {
  if (documents.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <FolderOpen aria-hidden="true" className="size-4" />
          <h2>Gather these once</h2>
        </CardTitle>
        <p className="text-muted-foreground text-sm">
          More than one program asks for each of these, so you only need to find them once.
        </p>
      </CardHeader>
      <CardContent>
        <ul className="divide-border -my-1">
          {documents.map((document) => {
            const item = itemsByKey.get(document.dedup_key);
            if (item === undefined) {
              return (
                <li
                  key={document.dedup_key}
                  className="border-border border-b py-3 last:border-b-0"
                >
                  <p className="text-sm font-medium">{document.title}</p>
                  <p className="text-muted-foreground text-sm">{document.detail}</p>
                </li>
              );
            }
            return (
              <ActionItemRow
                key={document.dedup_key}
                item={item}
                isSaving={isSaving(item.id)}
                onToggle={onToggle}
                footer={
                  <p className="text-muted-foreground text-xs">
                    Needed for {document.program_names.join(", ")}
                  </p>
                }
              />
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}
