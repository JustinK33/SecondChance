/**
 * The local development identity.
 *
 * When Supabase is not configured the backend is expected to be running
 * `AUTH_MODE=dev`, which accepts an unsigned `X-Dev-User: <uuid>` header. We mint
 * one UUID per browser and keep it in `localStorage` so a reload lands on the
 * same data instead of a fresh empty account.
 *
 * This is never a security boundary and the UI says so wherever it appears. It
 * exists so the product can be exercised end to end before a Supabase project
 * is provisioned.
 */

const STORAGE_KEY = "secondchance.dev-user-id";

/** RFC 4122 shape check. Rejects a hand-edited localStorage value. */
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function randomUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback for a context without randomUUID. Still cryptographically random.
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = ((bytes[6] ?? 0) & 0x0f) | 0x40;
  bytes[8] = ((bytes[8] ?? 0) & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join("-");
}

/**
 * The stable dev user id for this browser, creating it on first use.
 *
 * Returns `null` on the server, where there is no `localStorage` and no single
 * caller to attribute a request to.
 */
export function getDevUserId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const existing = window.localStorage.getItem(STORAGE_KEY);
    if (existing !== null && UUID_RE.test(existing)) return existing;
    const minted = randomUuid();
    window.localStorage.setItem(STORAGE_KEY, minted);
    return minted;
  } catch {
    // Private browsing with storage disabled. Fall back to a per-session id so
    // the app still works; the user simply starts fresh on reload.
    return randomUuid();
  }
}

/** Forget the local identity. Used after "delete my data" so the slate is really clean. */
export function clearDevUserId(): void {
  cached = undefined;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Nothing to clear if storage is unavailable.
    }
  }
  for (const listener of listeners) listener();
}

// ---------------------------------------------------------------------------
// React integration
//
// `localStorage` is an external store, so this is exposed as one rather than
// mirrored into component state in an effect. `useSyncExternalStore` then gives
// us the two things that pattern gets wrong by hand: a server snapshot of `null`
// so hydration matches, and a re-render when the identity is reset.
// ---------------------------------------------------------------------------

let cached: string | null | undefined;
const listeners = new Set<() => void>();

/** Subscribe to identity changes. Returns the unsubscribe function. */
export function subscribeDevUserId(onChange: () => void): () => void {
  listeners.add(onChange);
  return () => {
    listeners.delete(onChange);
  };
}

/**
 * The identity, memoised.
 *
 * `useSyncExternalStore` compares snapshots by identity and will loop forever if
 * this returns a fresh value each call, so the result is cached until something
 * invalidates it.
 */
export function devUserIdSnapshot(): string | null {
  if (cached === undefined) cached = getDevUserId();
  return cached;
}

/** The server snapshot. Always `null`: there is no browser storage to read. */
export function devUserIdServerSnapshot(): null {
  return null;
}
