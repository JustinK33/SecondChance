import type { Metadata } from "next";
import type { ReactNode } from "react";

import { OnboardingForm } from "@/components/features/intake/onboarding-form";
import { PreliminaryNotice } from "@/components/features/intake/preliminary-notice";
import { WhatWeCollect } from "@/components/features/intake/what-we-collect";

export const metadata: Metadata = {
  title: "Tell us what is going on",
};

/**
 * The onboarding screen.
 *
 * One `h1`, one primary action, and the privacy statement in sight of the field
 * it is about rather than linked from a footer. The form itself is a client
 * component; everything around it is static.
 */
export default function StartPage(): ReactNode {
  return (
    <div className="page-container py-8 sm:py-12">
      <div className="reading-width space-y-8">
        <header className="space-y-3">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Tell us what is going on
          </h1>
          <p className="text-muted-foreground">
            In your own words. We will read it, ask about anything important that is missing, and
            show you programs that may be worth exploring.
          </p>
        </header>

        <WhatWeCollect />
        <OnboardingForm />
        <PreliminaryNotice />
      </div>
    </div>
  );
}
