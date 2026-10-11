"use client";

/**
 * "Delete everything you have about me."
 *
 * Three things make this a promise rather than a gesture:
 *
 * 1. The confirmation names exactly what will be removed, before it happens.
 *    "Are you sure?" with no list is not informed consent.
 * 2. It is a hard delete on the server, not a flag, and the destructive action
 *    is never the default focus in the dialog.
 * 3. Afterwards the user sees the receipt: how many rows went, and what was
 *    deliberately kept and why. A deletion you cannot verify is a rumour.
 */

import { useState } from "react";
import type { ReactNode } from "react";
import { Trash2 } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { useAnnouncer } from "@/lib/announcer";
import { formatDate } from "@/lib/format";
import { useAccountDeletion } from "@/lib/hooks";

/**
 * Plain-language names for the rows the receipt counts.
 *
 * The keys are database table names, which are not the user's vocabulary. Rows
 * that mean the same thing to a person share a label and are summed, and
 * anything this build has not seen before is folded into one honest catch-all
 * rather than shown as a table name.
 */
const ROW_LABELS: Record<string, string> = {
  profiles: "records of what we understood about you",
  workflow_runs: "searches you ran",
  program_matches: "programs we matched for you, and why",
  action_plans: "action plans",
  action_items: "steps on your plans",
  audit_events: "records that you used the site",
  checkpoints: "saved progress from your searches",
  checkpoint_writes: "saved progress from your searches",
  checkpoint_blobs: "saved progress from your searches",
};

const UNKNOWN_ROW_LABEL = "other records attached to you";

/** Counts folded onto their plain-language labels, largest first. */
function summarise(counts: Record<string, number> | undefined): [string, number][] {
  const totals = new Map<string, number>();
  for (const [key, count] of Object.entries(counts ?? {})) {
    if (count <= 0) continue;
    const label = ROW_LABELS[key] ?? UNKNOWN_ROW_LABEL;
    totals.set(label, (totals.get(label) ?? 0) + count);
  }
  return [...totals.entries()].sort((left, right) => right[1] - left[1]);
}

/** What the user is told will go, in the order they would think of it. */
const WILL_BE_REMOVED: readonly string[] = [
  "Everything you wrote about your situation.",
  "Every answer you gave us, including your income range and household size.",
  "Every search you ran and the programs it matched.",
  "Every action plan and every step you ticked off.",
];

export function DeleteDataDialog(): ReactNode {
  const { announce } = useAnnouncer();
  const deletion = useAccountDeletion();
  const [isOpen, setIsOpen] = useState(false);

  if (deletion.receipt !== null) {
    const counts = summarise(deletion.receipt.deleted_counts);
    const deletedAt = formatDate(deletion.receipt.deleted_at);
    return (
      <section
        aria-labelledby="deletion-receipt-heading"
        role="status"
        aria-live="polite"
        className="bg-card space-y-3 rounded-lg border p-4"
      >
        <h2 id="deletion-receipt-heading" className="text-base font-semibold">
          Your information has been deleted
        </h2>
        <p className="text-sm">
          {deletedAt === null
            ? "This has been done."
            : `This was done on ${deletedAt}. Here is exactly what was removed.`}
        </p>
        {counts.length > 0 ? (
          <ul className="list-disc space-y-1 pl-5 text-sm">
            {counts.map(([label, count]) => (
              <li key={label}>
                {count} {label}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground text-sm">
            There was nothing stored under your id, so there was nothing to remove.
          </p>
        )}
        {deletion.receipt.retained !== undefined && deletion.receipt.retained.length > 0 ? (
          <>
            <h3 className="text-sm font-semibold">What we kept, and why</h3>
            <ul className="text-muted-foreground list-disc space-y-1 pl-5 text-sm">
              {deletion.receipt.retained.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </>
        ) : null}
        <p className="text-muted-foreground text-sm">
          You can start again any time. Nothing from before will be used.
        </p>
      </section>
    );
  }

  return (
    <section aria-labelledby="delete-data-heading" className="space-y-3">
      <h2 id="delete-data-heading" className="text-xl font-semibold">
        Delete everything we hold about you
      </h2>
      <p className="text-muted-foreground reading-width">
        This removes your information from our database for good. We cannot undo it and we cannot
        get it back for you.
      </p>

      {deletion.error !== null ? (
        <ErrorState error={deletion.error} title="We could not delete your information" />
      ) : null}

      <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
        <AlertDialogTrigger asChild>
          <Button variant="destructive" size="lg" className="touch-row px-6 text-base">
            <Trash2 aria-hidden="true" className="size-4" />
            Delete all my data
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent className="max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete everything we hold about you?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2 text-left">
                <p>This cannot be undone. We will remove:</p>
                <ul className="list-disc space-y-1 pl-5">
                  {WILL_BE_REMOVED.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
                <p>
                  Programs you have already applied for are not affected - those applications are
                  held by the programs themselves, not by us.
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="touch-row">Keep my information</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              className="touch-row"
              disabled={deletion.isPending}
              onClick={(event) => {
                // Keep the dialog open until the request has actually settled,
                // so the user is not told it worked before it did.
                event.preventDefault();
                announce("Deleting your information.", "assertive");
                void deletion.deleteEverything().then((receipt) => {
                  if (receipt === null) return;
                  setIsOpen(false);
                  announce("Your information has been deleted.", "assertive");
                });
              }}
            >
              {deletion.isPending ? "Deleting..." : "Yes, delete everything"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
