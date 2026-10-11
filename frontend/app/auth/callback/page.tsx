"use client";

/**
 * Where a magic link lands.
 *
 * Supabase sends the user back here with either a PKCE `code` to exchange or, on
 * the implicit flow, tokens in the URL fragment that the browser client picks up
 * on its own. Both are handled: the code is exchanged explicitly, and the
 * fragment case is covered by waiting for the auth state to settle.
 *
 * This is auth plumbing rather than one of the product screens, which is why it
 * lives here rather than with the feature routes.
 *
 * `next` is validated to be a same-site path before being used, so the callback
 * cannot be turned into an open redirect.
 */

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { LoadingState } from "@/components/ui/loading-state";
import { ROUTES } from "@/components/layout/routes";
import { USES_SUPABASE_AUTH } from "@/lib/env";
import { getSupabaseClient } from "@/lib/supabase";

/** Only a same-site absolute path is an acceptable redirect target. */
function safeNext(value: string | null): string {
  if (value === null) return ROUTES.start;
  if (!value.startsWith("/") || value.startsWith("//")) return ROUTES.start;
  return value;
}

function CallbackInner() {
  const router = useRouter();
  const params = useSearchParams();
  const [exchangeFailure, setExchangeFailure] = useState<string | null>(null);

  const next = safeNext(params.get("next"));
  const code = params.get("code");
  const errorDescription = params.get("error_description") ?? params.get("error");

  // Supabase can report the failure in the URL itself, which is known at render
  // time and therefore derived rather than stored.
  const linkFailure =
    errorDescription === null
      ? null
      : "That sign-in link did not work. It may have expired or already been used.";
  // Whether sign-in is possible at all is configuration, known at render time.
  const configFailure = USES_SUPABASE_AUTH
    ? null
    : "Sign-in is not configured on this installation.";
  const failure = linkFailure ?? configFailure ?? exchangeFailure;

  useEffect(() => {
    if (errorDescription !== null || !USES_SUPABASE_AUTH) return;

    const client = getSupabaseClient();
    if (client === null) return;

    let active = true;

    const finish = async (): Promise<void> => {
      if (code !== null) {
        const { error } = await client.auth.exchangeCodeForSession(code);
        if (!active) return;
        if (error !== null) {
          setExchangeFailure(
            "That sign-in link did not work. It may have expired or already been used.",
          );
          return;
        }
        router.replace(next);
        return;
      }

      // Implicit flow: the client reads the fragment itself. Give it a session
      // check rather than assuming it has finished.
      const { data } = await client.auth.getSession();
      if (!active) return;
      if (data.session !== null) {
        router.replace(next);
        return;
      }
      setExchangeFailure("We could not complete sign-in. Please request a new link.");
    };

    void finish();
    return () => {
      active = false;
    };
  }, [code, errorDescription, next, router]);

  if (failure !== null) {
    return (
      <div className="page-container py-16">
        <div className="reading-width space-y-5">
          <h1 className="text-2xl font-semibold">That link did not work</h1>
          <p>{failure}</p>
          <Button asChild className="touch-row">
            <Link href={ROUTES.signIn}>Request a new link</Link>
          </Button>
        </div>
      </div>
    );
  }

  return <LoadingState message="Signing you in..." />;
}

export default function AuthCallbackPage() {
  return (
    <Suspense fallback={<LoadingState message="Signing you in..." />}>
      <CallbackInner />
    </Suspense>
  );
}
