"use client";

/**
 * The sign-in screen.
 *
 * A client component because `next` comes from the query string: the sign-in
 * link has to come back to wherever the user was heading. It is wrapped in
 * `Suspense` because `useSearchParams` requires it, and validated to a same-site
 * path so the callback cannot be pointed at another origin.
 */

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import type { ReactNode } from "react";

import { SignInForm } from "@/components/features/intake/sign-in-form";
import { ROUTES } from "@/components/layout/routes";

/** Only a same-site absolute path is an acceptable destination. */
function safeNext(value: string | null): string | undefined {
  if (value === null) return undefined;
  if (!value.startsWith("/") || value.startsWith("//")) return undefined;
  return value;
}

function SignInInner(): ReactNode {
  const params = useSearchParams();
  return <SignInForm next={safeNext(params.get("next"))} />;
}

export default function SignInPage(): ReactNode {
  return (
    <div className="page-container py-10 sm:py-14">
      <div className="reading-width space-y-6">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Sign in</h1>
        <p className="text-muted-foreground">
          Signing in is optional. You can get your list of programs without an account - an account
          only saves it for later.
        </p>
        <Suspense fallback={<SignInForm />}>
          <SignInInner />
        </Suspense>
        <p className="text-muted-foreground text-sm">
          Changed your mind?{" "}
          <a className="link" href={ROUTES.home}>
            Go back to the start
          </a>
          .
        </p>
      </div>
    </div>
  );
}
