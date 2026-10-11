import Link from "next/link";

import { Button } from "@/components/ui/button";
import { PrivacyNote } from "@/components/ui/privacy-note";
import { ROUTES } from "@/components/layout/routes";

/**
 * PLACEHOLDER - owned by the landing-screen agent, not by the foundation.
 *
 * It exists so the route tree builds and so the shell can be seen working. The
 * copy is deliberately plain; replace the whole file.
 */
export default function HomePage() {
  return (
    <div className="page-container py-12 sm:py-16">
      <div className="reading-width space-y-6">
        <h1 className="text-3xl font-semibold sm:text-4xl">
          Tell us what is going on, in your own words
        </h1>
        <p className="text-muted-foreground text-lg">
          SecondChance reads your description, finds New Jersey and Rutgers assistance programs that
          may be worth exploring, and gives you the official link and the steps for each one.
        </p>
        <div className="flex flex-wrap gap-3">
          <Button asChild size="lg" className="touch-row">
            <Link href={ROUTES.start}>Start</Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="touch-row">
            <Link href={ROUTES.programs}>Browse programs first</Link>
          </Button>
        </div>
        <PrivacyNote />
      </div>
    </div>
  );
}
