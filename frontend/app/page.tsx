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
 */
export default function HomePage(): ReactNode {
  return (
    <div className="page-container space-y-12 py-10 sm:py-14">
      <LandingHero />
      <HowItWorks className="reading-width sm:max-w-none" />
      <WhatWeCollect className="reading-width" />
      <PreliminaryNotice className="reading-width" />

      <section aria-labelledby="start-again-heading" className="reading-width space-y-4">
        <h2 id="start-again-heading" className="text-xl font-semibold">
          Ready when you are
        </h2>
        <p className="text-muted-foreground">
          You can stop part way through and come back. Your answers stay on your own device until
          you send them.
        </p>
        <Button asChild size="lg" className="touch-row w-full px-6 text-base sm:w-auto">
          <Link href={ROUTES.start}>Start</Link>
        </Button>
      </section>
    </div>
  );
}
