import Link from "next/link";

import { ROUTES } from "./routes";

/**
 * The footer, which carries the disclaimer the whole product depends on.
 *
 * This text is a product requirement, not decoration. It must stay on every
 * page, it must stay in plain language, and it must not be collapsed behind a
 * toggle. Two separate claims are being disclaimed:
 *
 * 1. We are not a government agency and not affiliated with one.
 * 2. Nothing here is an eligibility decision. Only the program can decide.
 */
export function SiteFooter() {
  return (
    <footer className="bg-muted/40 mt-auto border-t">
      <div className="page-container flex flex-col gap-6 py-8 md:flex-row md:items-start md:justify-between">
        <div className="reading-width space-y-3">
          <p className="text-foreground text-sm font-medium">
            SecondChance is an independent student project. It is not a government agency and is not
            affiliated with, endorsed by, or acting on behalf of any government agency or university
            office.
          </p>
          <p className="text-muted-foreground text-sm">
            Nothing on this site is an official eligibility decision. We show programs that may be
            worth exploring based on what you tell us, and we link to the official page so you can
            check and apply there. Only the organisation running a program can decide whether you
            qualify.
          </p>
          <p className="text-muted-foreground text-sm">
            We never ask for your Social Security number, bank details or passwords. No program
            listed here needs them from us.
          </p>
        </div>

        <nav aria-label="Footer" className="flex shrink-0 flex-col gap-2 text-sm">
          <Link href={ROUTES.programs} className="touch-row link flex items-center">
            Browse all programs
          </Link>
          <Link href={ROUTES.privacy} className="touch-row link flex items-center">
            How we handle your information
          </Link>
          <Link href={ROUTES.account} className="touch-row link flex items-center">
            Delete my information
          </Link>
        </nav>
      </div>
    </footer>
  );
}
