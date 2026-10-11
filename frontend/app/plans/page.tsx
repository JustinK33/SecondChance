import type { Metadata } from "next";
import type { ReactNode } from "react";

import { PlansIndex } from "@/components/features/plan";

export const metadata: Metadata = {
  title: "My plan",
  description: "The steps from your runs, and what you have finished so far.",
};

/** Every plan the signed-in person has, newest first. */
export default function PlansPage(): ReactNode {
  return (
    <div className="page-container space-y-6 py-8 sm:py-10">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold sm:text-3xl">Your plans</h1>
        <p className="text-muted-foreground reading-width">
          Each plan is the checklist from one time you described your situation. Ticking things off
          is saved, so you can work through it over several days.
        </p>
      </header>
      <PlansIndex />
    </div>
  );
}
