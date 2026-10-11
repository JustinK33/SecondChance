/**
 * The only module that reads `process.env` on the client.
 *
 * Next.js inlines `NEXT_PUBLIC_*` at build time, so these have to be referenced
 * as static property accesses - `process.env[name]` is not substituted. Keeping
 * every read here means there is one place to audit for a leaked secret, and one
 * place that decides what "not configured" means.
 *
 * Nothing in this file may reference a non-public variable. A value that reaches
 * the browser is public by definition.
 */

/** Trim and treat the empty string as absent, because a blank line in a .env file is not a value. */
function clean(value: string | undefined): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

/** Base URL of the SecondChance API, without a trailing slash. */
export const API_BASE_URL: string = (
  clean(process.env.NEXT_PUBLIC_API_URL) ?? "http://localhost:8000"
).replace(/\/+$/, "");

/** Supabase project URL, or `null` when the project has not been provisioned. */
export const SUPABASE_URL: string | null = clean(process.env.NEXT_PUBLIC_SUPABASE_URL);

/**
 * Supabase publishable key, or `null` when absent.
 *
 * This is the anon / `sb_publishable_` key, which is designed to ship to the
 * browser. The service-role key must never appear in this package.
 */
export const SUPABASE_PUBLISHABLE_KEY: string | null = clean(
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
);

/**
 * True when both Supabase values are present, so real sign-in is possible.
 *
 * Note that this is *capability*, not the active mode - see {@link AUTH_MODE}.
 */
export const SUPABASE_CONFIGURED: boolean =
  SUPABASE_URL !== null && SUPABASE_PUBLISHABLE_KEY !== null;

/** `supabase` means real sessions; `dev` means the local `X-Dev-User` identity. */
export type FrontendAuthMode = "supabase" | "dev";

/**
 * Which identity mechanism is live.
 *
 * Resolution order, and why it is this order:
 *
 * 1. `NEXT_PUBLIC_AUTH_MODE`, when set to `dev` or `supabase`. This exists
 *    because the two halves of the system have to agree: the API decides from
 *    its own `AUTH_MODE`, and a backend running `AUTH_MODE=dev` ignores bearer
 *    tokens entirely and requires `X-Dev-User`. Sending it a real Supabase
 *    token would 401 every request. An explicit switch is the only honest way
 *    to pin the browser to the same mode as the API it is talking to.
 * 2. Otherwise `supabase` when both Supabase values are present.
 * 3. Otherwise `dev`, which is the documented fallback: a missing Supabase
 *    project must degrade to a clearly-labelled local identity rather than
 *    crash the application.
 */
export const AUTH_MODE: FrontendAuthMode = (() => {
  const configured = clean(process.env.NEXT_PUBLIC_AUTH_MODE);
  if (configured === "dev") return "dev";
  if (configured === "supabase") return "supabase";
  return SUPABASE_CONFIGURED ? "supabase" : "dev";
})();

/** True when requests carry a Supabase access token. */
export const USES_SUPABASE_AUTH: boolean = AUTH_MODE === "supabase" && SUPABASE_CONFIGURED;

/** Human-readable label for the active identity mode. Shown in the UI, never hidden. */
export const AUTH_MODE_LABEL: string = USES_SUPABASE_AUTH
  ? "Signed-in account"
  : "Local development identity";
