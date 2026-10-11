"use client";

/**
 * The application header.
 *
 * A wordmark, not a seal. No crest, no flag, no shield - nothing that could be
 * mistaken for a government agency's identity. The identity state is shown
 * plainly, including the fact that the local development identity is not a
 * login, because quietly implying an account exists would be dishonest.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

import { MobileNav } from "./mobile-nav";
import { PRIMARY_NAV, ROUTES, isActivePath } from "./routes";

export function SiteHeader() {
  const pathname = usePathname();
  const { isAuthenticated, isDevIdentity, isLoading, signOut } = useAuth();

  return (
    <header className="bg-background/95 supports-[backdrop-filter]:bg-background/80 sticky top-0 z-40 border-b backdrop-blur">
      <div className="page-container flex h-16 items-center gap-4">
        <Link
          href={ROUTES.home}
          className="touch-row flex items-center gap-2 rounded-md font-semibold tracking-tight"
        >
          <span className="text-primary text-lg">SecondChance</span>
          <span className="sr-only">Home</span>
        </Link>

        <nav aria-label="Main" className="ml-4 hidden items-center gap-1 md:flex">
          {PRIMARY_NAV.map((item) => {
            const active = isActivePath(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "touch-row flex items-center rounded-md px-3 text-sm font-medium transition-colors",
                  active
                    ? "bg-secondary text-secondary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          {isLoading ? null : isDevIdentity ? (
            <Badge variant="outline" className="hidden sm:inline-flex">
              Local test identity
            </Badge>
          ) : isAuthenticated ? (
            <Button
              variant="ghost"
              className="touch-row hidden text-sm sm:inline-flex"
              onClick={() => void signOut()}
            >
              Sign out
            </Button>
          ) : (
            <Button asChild variant="outline" className="touch-row hidden text-sm sm:inline-flex">
              <Link href={ROUTES.signIn}>Sign in</Link>
            </Button>
          )}
          <MobileNav />
        </div>
      </div>
    </header>
  );
}
