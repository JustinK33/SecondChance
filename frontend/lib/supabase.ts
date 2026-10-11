/**
 * The Supabase browser client.
 *
 * Created lazily and memoised, because `createBrowserClient` installs listeners
 * and a token-refresh timer, and a second instance would race the first for the
 * same storage key.
 *
 * Returns `null` when the project is not configured. Every caller must handle
 * that: the whole point of the fallback is that a missing Supabase project
 * degrades to the local development identity rather than crashing the app.
 *
 * The key read here is the publishable (`sb_publishable_`) key, which is
 * designed to be sent to a browser. The service-role key must never appear in
 * this package.
 */

import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";

import { SUPABASE_PUBLISHABLE_KEY, SUPABASE_URL } from "./env";

let cached: SupabaseClient | null = null;

/** The memoised browser client, or `null` when Supabase is not configured. */
export function getSupabaseClient(): SupabaseClient | null {
  if (SUPABASE_URL === null || SUPABASE_PUBLISHABLE_KEY === null) return null;
  if (cached !== null) return cached;
  cached = createBrowserClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
  return cached;
}

/** Where a magic link should land. Absolute, because Supabase redirects to it. */
export function authCallbackUrl(next?: string): string {
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  const url = new URL("/auth/callback", origin || "http://localhost:3000");
  if (next !== undefined && next.startsWith("/")) url.searchParams.set("next", next);
  return url.toString();
}
