"use client";

/**
 * Navigation on a phone: a sheet behind one clearly-labelled button.
 *
 * Mobile is the primary case for this product, so the sheet carries the full
 * nav with its descriptions rather than a reduced version of the desktop bar.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useAuth } from "@/lib/auth";
import { cn } from "@/lib/utils";

import { PRIMARY_NAV, isActivePath } from "./routes";

export function MobileNav() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const { isAuthenticated } = useAuth();

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="touch-target md:hidden">
          <Menu aria-hidden="true" className="size-5" />
          <span className="sr-only">Open menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-[min(22rem,90vw)]">
        <SheetHeader>
          <SheetTitle>Menu</SheetTitle>
          <SheetDescription>Where would you like to go?</SheetDescription>
        </SheetHeader>
        <nav aria-label="Main" className="flex flex-col gap-1 px-4 pb-6">
          {PRIMARY_NAV.map((item) => {
            const active = isActivePath(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "touch-row flex flex-col justify-center rounded-md px-3 py-3 transition-colors",
                  active ? "bg-secondary text-secondary-foreground" : "hover:bg-muted",
                )}
              >
                <span className="font-medium">{item.label}</span>
                <span className="text-muted-foreground text-sm">{item.description}</span>
                {item.requiresAuth && !isAuthenticated ? (
                  <span className="text-muted-foreground mt-1 text-xs">Needs sign-in</span>
                ) : null}
              </Link>
            );
          })}
        </nav>
      </SheetContent>
    </Sheet>
  );
}
