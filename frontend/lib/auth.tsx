"use client";

/**
 * Identity for the whole application.
 *
 * Two modes, chosen by configuration rather than by a build flag:
 *
 * - `supabase` - a real session. Sign-in is an emailed magic link that lands on
 *   `/auth/callback`. The access token is read from the live session on every
 *   request so a refreshed token is picked up without a reload.
 * - `dev` - no Supabase project configured. A stable UUID in `localStorage` is
 *   sent as `X-Dev-User`, which the backend accepts only under `AUTH_MODE=dev`
 *   on `APP_ENV=local`. The UI labels this plainly; it is not a login.
 *
 * The provider installs its credential resolver into `lib/auth-token`, which is
 * what `lib/api-client` reads. That indirection keeps the client free of React
 * and keeps this file free of request building.
 *
 * Nothing here logs an email, a token or a user id.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import type { ReactNode } from "react";
import type { Session, SupabaseClient } from "@supabase/supabase-js";

import { resetCredentialResolver, setCredentialResolver } from "./auth-token";
import {
  clearDevUserId,
  devUserIdServerSnapshot,
  devUserIdSnapshot,
  getDevUserId,
  subscribeDevUserId,
} from "./dev-identity";
import { USES_SUPABASE_AUTH } from "./env";
import { authCallbackUrl, getSupabaseClient } from "./supabase";

export type AuthMode = "supabase" | "dev";

export interface AuthUser {
  /** The id the backend attributes data to. `sub` in Supabase mode. */
  readonly id: string;
  /** Present only in Supabase mode. Never logged. */
  readonly email: string | null;
}

export interface AuthState {
  /** True until the initial session lookup has settled. */
  readonly isLoading: boolean;
  /** Which identity mechanism is live. */
  readonly mode: AuthMode;
  /** The current principal, or `null` when nobody is identified. */
  readonly user: AuthUser | null;
  /** True when a request will carry an identity. */
  readonly isAuthenticated: boolean;
  /**
   * True when this is the local development identity rather than a real login.
   * Screens must surface this; pretending it is a login would be dishonest.
   */
  readonly isDevIdentity: boolean;
  /** The last sign-in or sign-out failure, in plain language. */
  readonly error: string | null;
  /** Set after a magic link is sent, so the UI can say "check your email". */
  readonly magicLinkSentTo: string | null;
}

export interface AuthActions {
  /** Email a magic link. Resolves when the email has been accepted for delivery. */
  signInWithEmail(email: string, next?: string): Promise<void>;
  /** End the session. In dev mode this mints a fresh local identity. */
  signOut(): Promise<void>;
  /** Forget the local dev identity. Used after account deletion. */
  resetLocalIdentity(): void;
  /** Clear a displayed error without retrying. */
  clearError(): void;
}

export type AuthContextValue = AuthState & AuthActions;

const AuthContext = createContext<AuthContextValue | null>(null);

function userFromSession(session: Session | null): AuthUser | null {
  if (session === null) return null;
  const id = session.user.id;
  if (typeof id !== "string" || id.length === 0) return null;
  return { id, email: session.user.email ?? null };
}

/** Plain-language text for a Supabase auth failure. Never the raw provider string. */
function authErrorMessage(kind: "send" | "signout"): string {
  return kind === "send"
    ? "We could not send that sign-in link. Check the address and try again."
    : "We could not sign you out. Please try again.";
}

export function AuthProvider({ children }: { children: ReactNode }): ReactNode {
  const mode: AuthMode = USES_SUPABASE_AUTH ? "supabase" : "dev";

  // Built once, lazily. `createBrowserClient` installs listeners and a refresh
  // timer, so a second instance would race the first for the same storage key.
  const [client] = useState<SupabaseClient | null>(() =>
    mode === "supabase" ? getSupabaseClient() : null,
  );

  const [isLoading, setIsLoading] = useState<boolean>(mode === "supabase" && client !== null);
  const [sessionUser, setSessionUser] = useState<AuthUser | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [magicLinkSentTo, setMagicLinkSentTo] = useState<string | null>(null);

  // Mode `dev`: `localStorage` is an external store, so it is subscribed to
  // rather than copied into state inside an effect. The server snapshot is
  // `null`, which is what keeps hydration from mismatching.
  const devUserId = useSyncExternalStore(
    subscribeDevUserId,
    mode === "dev" ? devUserIdSnapshot : devUserIdServerSnapshot,
    devUserIdServerSnapshot,
  );

  // Memoised so the context value below is not a fresh object every render.
  const user = useMemo<AuthUser | null>(
    () =>
      mode === "dev" ? (devUserId === null ? null : { id: devUserId, email: null }) : sessionUser,
    [mode, devUserId, sessionUser],
  );

  // Mode `supabase`: adopt the stored session, then follow it.
  useEffect(() => {
    if (mode !== "supabase" || client === null) return;
    let active = true;

    void client.auth
      .getSession()
      .then(({ data }) => {
        if (!active) return;
        setSessionUser(userFromSession(data.session));
      })
      .catch(() => {
        if (active) setSessionUser(null);
      })
      .finally(() => {
        if (active) setIsLoading(false);
      });

    const { data } = client.auth.onAuthStateChange((_event, session) => {
      if (!active) return;
      setSessionUser(userFromSession(session));
      setIsLoading(false);
    });

    return () => {
      active = false;
      data.subscription.unsubscribe();
    };
  }, [mode, client]);

  // Install the resolver the API client reads. Reading the session on every
  // request rather than caching the token here means a refresh is picked up
  // without a re-render.
  useEffect(() => {
    if (mode === "dev") {
      setCredentialResolver(() => {
        const id = getDevUserId();
        return id === null ? null : { kind: "dev", userId: id };
      });
    } else {
      setCredentialResolver(async () => {
        if (client === null) return null;
        const { data } = await client.auth.getSession();
        const token = data.session?.access_token;
        return typeof token === "string" && token.length > 0 ? { kind: "supabase", token } : null;
      });
    }
    return () => resetCredentialResolver();
  }, [mode, client]);

  const signInWithEmail = useCallback(
    async (email: string, next?: string): Promise<void> => {
      setError(null);
      if (mode === "dev") {
        // Honest refusal rather than a fake success screen.
        setError("Sign-in is not available in local development mode.");
        return;
      }
      if (client === null) {
        setError("Sign-in is not configured.");
        return;
      }
      const { error: sendError } = await client.auth.signInWithOtp({
        email: email.trim(),
        options: { emailRedirectTo: authCallbackUrl(next) },
      });
      if (sendError !== null) {
        setError(authErrorMessage("send"));
        return;
      }
      setMagicLinkSentTo(email.trim());
    },
    [mode, client],
  );

  const signOut = useCallback(async (): Promise<void> => {
    setError(null);
    setMagicLinkSentTo(null);
    if (mode === "dev") {
      // Clearing notifies the store, which re-renders with a freshly minted id.
      clearDevUserId();
      getDevUserId();
      return;
    }
    if (client === null) return;
    const { error: signOutError } = await client.auth.signOut();
    if (signOutError !== null) {
      setError(authErrorMessage("signout"));
      return;
    }
    setSessionUser(null);
  }, [mode, client]);

  const resetLocalIdentity = useCallback((): void => {
    if (mode !== "dev") return;
    clearDevUserId();
    getDevUserId();
  }, [mode]);

  const clearError = useCallback((): void => setError(null), []);

  const value = useMemo<AuthContextValue>(
    () => ({
      isLoading,
      mode,
      user,
      isAuthenticated: user !== null,
      isDevIdentity: mode === "dev",
      error,
      magicLinkSentTo,
      signInWithEmail,
      signOut,
      resetLocalIdentity,
      clearError,
    }),
    [
      isLoading,
      mode,
      user,
      error,
      magicLinkSentTo,
      signInWithEmail,
      signOut,
      resetLocalIdentity,
      clearError,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/** The auth context. Throws if used outside the provider, which is a wiring bug. */
export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (value === null) {
    throw new Error("useAuth must be used inside <AuthProvider>.");
  }
  return value;
}

/**
 * The context without throwing, for a component that may render outside the tree.
 * Prefer {@link useAuth}.
 */
export function useOptionalAuth(): AuthContextValue | null {
  return useContext(AuthContext);
}
