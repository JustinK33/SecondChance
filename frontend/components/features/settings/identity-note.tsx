"use client";

/**
 * Which identity this data is attached to, said plainly.
 *
 * The local development identity is not a login and must never be presented as
 * one: it is a random id in this browser, and losing the browser loses the data.
 * Someone deciding whether to type their situation into this box deserves to
 * know that before they start, not after.
 *
 * No email address is rendered here. There is no reason to put it on screen.
 */

import Link from "next/link";
import type { ReactNode } from "react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";

export function IdentityNote(): ReactNode {
  const { isLoading, isAuthenticated, isDevIdentity, signOut } = useAuth();

  if (isLoading) return null;

  if (isDevIdentity) {
    return (
      <div className="bg-card space-y-2 rounded-lg border p-4 text-sm">
        <h2 className="text-base font-semibold">This is a local test identity</h2>
        <p>
          There are no accounts on this installation. You are identified by a random id kept in this
          browser, so clearing your browser data, or using a different device, looks like a
          different person to us.
        </p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="bg-card space-y-3 rounded-lg border p-4 text-sm">
        <h2 className="text-base font-semibold">You are not signed in</h2>
        <p>
          Sign in to see and correct what we hold about you. You can still search for programs
          without an account.
        </p>
        <Button asChild className="touch-row">
          <Link href={`${ROUTES.signIn}?next=${encodeURIComponent(ROUTES.account)}`}>Sign in</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="bg-card flex flex-wrap items-center justify-between gap-3 rounded-lg border p-4 text-sm">
      <p>You are signed in. Everything below is attached to your account.</p>
      <Button variant="outline" className="touch-row" onClick={() => void signOut()}>
        Sign out
      </Button>
    </div>
  );
}
