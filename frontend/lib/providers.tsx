"use client";

/**
 * Every client-side provider the application needs, in one component.
 *
 * `app/layout.tsx` stays a server component and renders this once. Order
 * matters: the announcer is outermost so an auth transition can be announced,
 * and the tooltip provider is innermost because only presentation needs it.
 */

import type { ReactNode } from "react";

import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";

import { AnnouncerProvider } from "./announcer";
import { AuthProvider } from "./auth";

export function Providers({ children }: { children: ReactNode }): ReactNode {
  return (
    <AnnouncerProvider>
      <AuthProvider>
        <TooltipProvider delayDuration={200}>
          {children}
          <Toaster position="bottom-center" closeButton />
        </TooltipProvider>
      </AuthProvider>
    </AnnouncerProvider>
  );
}
