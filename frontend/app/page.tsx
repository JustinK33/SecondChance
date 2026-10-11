import Link from "next/link";
import type { ReactNode } from "react";

import { HowItWorks } from "@/components/features/intake/how-it-works";
import { LandingHero } from "@/components/features/intake/landing-hero";
import { PreliminaryNotice } from "@/components/features/intake/preliminary-notice";
import { WhatWeCollect } from "@/components/features/intake/what-we-collect";
import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";

/**
 * The landing screen.
 *
 * Order is the argument: what this is, how it works, what we collect, what it
 * is not, then the action again for anyone who scrolled to the bottom before
 * deciding. A server component - there is no state here, so there is no reason
 * to ship it as JavaScript.
 *
 * The layout is a centred 64rem column rather than the full 72rem page: the
 * hero is one reading column, and the sections below it widen into a grid so a
 * desktop window is used rather than leaving its right half empty.
 */
export default function HomePage(): ReactNode {
  return (
    <div className="page-container py-10 sm:py-14">
      <div className="mx-auto max-w-5xl space-y-14">
        <LandingHero />

        <HowItWorks />

        {/* Two notices of similar weight. Side by side above `lg`, stacked
            below it, so neither is buried under a long scroll on a phone. They
            stretch to a common height rather than leaving a ragged gap under
            the shorter one. */}
        <div className="grid gap-6 lg:grid-cols-2 [&>*]:h-full">
          <WhatWeCollect />
          <PreliminaryNotice />
        </div>

        <section
          aria-labelledby="start-again-heading"
          className="reading-column space-y-4 border-t pt-10 text-center"
        >
          <h2 id="start-again-heading" className="text-xl font-semibold">
            Ready when you are
          </h2>
          <p className="text-muted-foreground">
            You can stop part way through and come back. Your answers stay on your own device until
            you send them.
          </p>
          <Button asChild size="lg" className="touch-row w-full px-8 text-base sm:w-auto">
            <Link href={ROUTES.start}>Start</Link>
          </Button>
        </section>
      </div>
    </div>
  );
}
