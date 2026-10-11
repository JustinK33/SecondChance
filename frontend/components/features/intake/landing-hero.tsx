import Link from "next/link";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";

/**
 * The first screen, and the only chance to be believed.
 *
 * One honest sentence about what this does, one about who it is for, and one
 * primary action. The secondary action is deliberately quieter: browsing the
 * catalog is useful, but it is not what most people arrive needing.
 */
export function LandingHero(): ReactNode {
  return (
    <section className="reading-width space-y-6">
      <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
        Find help you may be able to get in New Jersey
      </h1>
      <p className="text-lg">
        Describe what is going on in your own words. We check it against the published rules for
        real New Jersey and Rutgers assistance programs and give you a short list worth exploring,
        with the official link for each one.
      </p>
      <p className="text-muted-foreground">
        It is built for people who are not sure where to start: students, parents, people between
        jobs, anyone who has been told to &ldquo;apply online&rdquo; and found twelve different
        websites. You do not need to know the name of any program.
      </p>
      <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
        <Button asChild size="lg" className="touch-row w-full px-6 text-base sm:w-auto">
          <Link href={ROUTES.start}>Start - it takes about 5 minutes</Link>
        </Button>
        <Button
          asChild
          size="lg"
          variant="outline"
          className="touch-row w-full px-6 text-base sm:w-auto"
        >
          <Link href={ROUTES.programs}>Browse programs first</Link>
        </Button>
      </div>
      <p className="text-muted-foreground text-sm">Free. No account needed to get your list.</p>
    </section>
  );
}
