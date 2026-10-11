/**
 * The seam between the auth context and the API client.
 *
 * `lib/api-client.ts` must not import React, and `lib/auth.tsx` must not own
 * request building, so the credential is passed through a single registered
 * resolver instead. The auth provider installs its resolver on mount; until then
 * the default resolver returns the local development identity, which is what
 * makes a fetch from outside the React tree still work.
 */

import { getDevUserId } from "./dev-identity";
import { USES_SUPABASE_AUTH } from "./env";

/**
 * What to put on an outgoing request.
 *
 * `supabase` becomes `Authorization: Bearer <token>`; `dev` becomes
 * `X-Dev-User: <uuid>`. `null` means send neither, which is correct for the
 * unauthenticated program catalog.
 */
export type Credential =
  | { readonly kind: "supabase"; readonly token: string }
  | { readonly kind: "dev"; readonly userId: string }
  | null;

export type CredentialResolver = () => Credential | Promise<Credential>;

function defaultResolver(): Credential {
  if (USES_SUPABASE_AUTH) {
    // The provider has not mounted yet and we have no session to read here.
    // Returning null is honest: the request goes out unauthenticated and the
    // backend answers 401, which the UI turns into "sign in again".
    return null;
  }
  const userId = getDevUserId();
  return userId === null ? null : { kind: "dev", userId };
}

let resolver: CredentialResolver = defaultResolver;

/** Install the live resolver. Called by the auth provider. */
export function setCredentialResolver(next: CredentialResolver): void {
  resolver = next;
}

/** Restore the default resolver. Called when the auth provider unmounts. */
export function resetCredentialResolver(): void {
  resolver = defaultResolver;
}

/** The credential for the next request, or `null` when there is none. */
export async function resolveCredential(): Promise<Credential> {
  try {
    return await resolver();
  } catch {
    // A failure to read the session must not take out the request path; the
    // backend's 401 is a better signal than a thrown error here.
    return null;
  }
}
