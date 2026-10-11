import type { Metadata } from "next";
import type { ReactNode } from "react";

import { ProgramCatalog } from "@/components/features/programs";

export const metadata: Metadata = {
  title: "Browse programs",
  description:
    "Every New Jersey and Rutgers assistance program we have checked, with the official source for each one.",
};

/**
 * The catalog.
 *
 * Open to everyone: `GET /api/v1/programs` is the one unauthenticated data
 * route, so somebody can look through real programs before deciding whether to
 * tell us anything about themselves.
 */
export default function ProgramsPage(): ReactNode {
  return (
    <div className="page-container space-y-6 py-8 sm:py-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold sm:text-3xl">Programs we have checked</h1>
        <p className="text-muted-foreground reading-width">
          Search or filter the list. Each program shows who runs it, what it is in plain language,
          and when we last checked its details against the official source.
        </p>
      </header>
      <ProgramCatalog />
    </div>
  );
}
