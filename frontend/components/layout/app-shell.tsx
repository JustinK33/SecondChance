import type { ReactNode } from "react";

import { SiteFooter } from "./site-footer";
import { SiteHeader } from "./site-header";
import { SkipLink } from "./skip-link";
import { StatusAnnouncer } from "./status-announcer";

/**
 * Header, main, footer, skip link and live regions.
 *
 * `main` carries the id the skip link targets and `tabIndex={-1}` so moving
 * focus to it actually works in Safari.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <>
      <SkipLink />
      <StatusAnnouncer />
      <SiteHeader />
      <main id="main-content" tabIndex={-1} className="flex-1 focus-visible:outline-none">
        {children}
      </main>
      <SiteFooter />
    </>
  );
}
