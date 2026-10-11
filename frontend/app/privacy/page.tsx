import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

import { PreliminaryNotice } from "@/components/features/intake/preliminary-notice";
import { WhatWeCollect } from "@/components/features/intake/what-we-collect";
import { PrivacyExplainer } from "@/components/features/settings";
import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";

export const metadata: Metadata = {
  title: "How we handle your information",
  description:
    "What SecondChance collects, what it never asks for, what it never writes down, and how to delete everything it holds.",
};

/**
 * `ROUTES.privacy` - the page the footer link on every screen points at.
 *
 * It is deliberately public and carries no identity: somebody deciding whether
 * to type anything at all must be able to read this first. The same components
 * render it as render the privacy copy inside onboarding and "your
 * information", so there is one wording rather than a policy page that drifts
 * away from what the product actually does.
 */
export default function PrivacyPage(): ReactNode {
  return (
    <div className="page-container py-8 sm:py-12">
      <div className="reading-column space-y-10">
        <header className="space-y-3">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            How we handle your information
          </h1>
          <p className="text-muted-foreground">
            The short version, in plain sentences. Every line here is a statement about how this
            site actually works, not a general assurance.
          </p>
        </header>

        <WhatWeCollect />
        <PrivacyExplainer />
        <PreliminaryNotice />

        <section aria-labelledby="privacy-delete-heading" className="space-y-4 border-t pt-8">
          <h2 id="privacy-delete-heading" className="text-xl font-semibold">
            Removing everything we hold
          </h2>
          <p className="text-muted-foreground">
            Deletion is immediate and we show you exactly what was removed. You do not have to ask
            us or wait for a reply.
          </p>
          <Button asChild size="lg" className="touch-row">
            <Link href={ROUTES.account}>Go to your information</Link>
          </Button>
        </section>
      </div>
    </div>
  );
}
