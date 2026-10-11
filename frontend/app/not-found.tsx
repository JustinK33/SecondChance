import Link from "next/link";

import { Button } from "@/components/ui/button";
import { ROUTES } from "@/components/layout/routes";

export const metadata = { title: "Page not found" };

/**
 * A 404 that helps rather than scolds.
 *
 * Most arrivals here are a stale bookmark or a link from a message, so the page
 * offers the two things that are almost always what was wanted.
 */
export default function NotFound() {
  return (
    <div className="page-container py-16">
      <div className="reading-width space-y-5">
        <h1 className="text-2xl font-semibold">We could not find that page</h1>
        <p>
          The link may be out of date, or the page may have moved. Nothing is wrong with your
          account and nothing has been lost.
        </p>
        <div className="flex flex-wrap gap-3">
          <Button asChild className="touch-row">
            <Link href={ROUTES.start}>Start finding help</Link>
          </Button>
          <Button asChild variant="outline" className="touch-row">
            <Link href={ROUTES.programs}>Browse all programs</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
