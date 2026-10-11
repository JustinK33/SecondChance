/**
 * The request proxy: security headers, and Supabase session upkeep.
 *
 * This is the `proxy.ts` convention, which replaced `middleware.ts` in Next 16
 * - the old filename still works but warns on every build, and a build that
 * warns is a build people stop reading.
 *
 * Two jobs, and nothing else. It runs on every request, so anything expensive
 * here is paid for on every navigation.
 *
 * **Security headers.** Set here rather than in `next.config.ts` so they also
 * cover responses the config's `headers()` does not, and so the connect-src
 * value can be derived from the configured API origin instead of hard-coded.
 * The CSP permits `'unsafe-inline'` for scripts because Next's App Router
 * bootstraps with inline scripts and a nonce-based policy needs every rendered
 * page to be dynamic - a trade this project is not ready to make. It is still
 * a meaningful policy: no `object-src`, no framing, and a closed `connect-src`.
 *
 * **Session upkeep.** `@supabase/ssr` keeps the session in cookies, which means
 * a refresh has to be written back on a response. Doing it here keeps a long
 * session alive across navigations. It is skipped entirely when the browser is
 * in local-development identity mode, because there is no cookie to refresh and
 * paying for a Supabase client per request would be pure waste.
 *
 * Deliberately *not* done here: route protection. In development-identity mode
 * the identity lives in `localStorage`, which this layer cannot see, so a
 * cookie-based redirect would bounce every authenticated page to sign-in. The
 * screens gate their own content on `useAuth()`, which is correct in both modes.
 */

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { createServerClient } from "@supabase/ssr";

const API_ORIGIN = (() => {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (raw === undefined || raw.length === 0) return "http://localhost:8000";
  try {
    return new URL(raw).origin;
  } catch {
    return "http://localhost:8000";
  }
})();

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL?.trim() ?? "";
const SUPABASE_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim() ?? "";
const AUTH_MODE = process.env.NEXT_PUBLIC_AUTH_MODE?.trim() ?? "";

/** True when there is a cookie-backed Supabase session worth refreshing. */
const SUPABASE_SESSIONS_ACTIVE =
  SUPABASE_URL.length > 0 && SUPABASE_KEY.length > 0 && AUTH_MODE !== "dev";

function contentSecurityPolicy(): string {
  const supabaseOrigin = SUPABASE_URL.length > 0 ? ` ${SUPABASE_URL}` : "";
  // The dev server's hot reloader compiles modules with `eval`. Allowing it in
  // production would defeat most of the point of a script policy, so it is
  // scoped to development only.
  const devEval = process.env.NODE_ENV === "production" ? "" : " 'unsafe-eval'";
  return [
    "default-src 'self'",
    // Next's bootstrap and the font loader both inline.
    `script-src 'self' 'unsafe-inline'${devEval}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    // Websocket for the dev server's hot reload, same scoping as above.
    `connect-src 'self' ${API_ORIGIN}${supabaseOrigin}${
      process.env.NODE_ENV === "production" ? "" : " ws: wss:"
    }`,
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "upgrade-insecure-requests",
  ].join("; ");
}

function withSecurityHeaders(response: NextResponse): NextResponse {
  response.headers.set("Content-Security-Policy", contentSecurityPolicy());
  response.headers.set("X-Content-Type-Options", "nosniff");
  response.headers.set("X-Frame-Options", "DENY");
  // Do not leak the path - which can contain a run id - to an external site.
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  response.headers.set(
    "Permissions-Policy",
    "camera=(), microphone=(), geolocation=(), payment=()",
  );
  return response;
}

export default async function proxy(request: NextRequest): Promise<NextResponse> {
  let response = NextResponse.next({ request });

  if (!SUPABASE_SESSIONS_ACTIVE) {
    return withSecurityHeaders(response);
  }

  const client = createServerClient(SUPABASE_URL, SUPABASE_KEY, {
    cookies: {
      getAll: () => request.cookies.getAll(),
      setAll: (cookies) => {
        for (const { name, value } of cookies) {
          request.cookies.set(name, value);
        }
        response = NextResponse.next({ request });
        for (const { name, value, options } of cookies) {
          response.cookies.set(name, value, options);
        }
      },
    },
  });

  // Reading the user is what triggers a refresh and therefore the cookie write.
  // A failure here must not break the page: the client-side provider will see
  // the same thing and render the signed-out state.
  try {
    await client.auth.getUser();
  } catch {
    // Supabase unreachable. Continue unauthenticated.
  }

  return withSecurityHeaders(response);
}

export const config = {
  /**
   * Skip static assets and the image optimiser. Matching them would add a
   * proxy invocation to every icon request for no benefit.
   */
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};
