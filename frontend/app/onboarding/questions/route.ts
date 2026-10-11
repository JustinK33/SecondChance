import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * `/onboarding/questions` is an alias for `/start/questions`.
 *
 * A route handler rather than a page, for one concrete reason: the redirect has
 * to carry the `run` query parameter across, and a *page* that reads
 * `searchParams` cannot be prerendered. A handler is the right shape for
 * something whose entire job is to issue a 308 and never render anything.
 */
export function GET(request: NextRequest): NextResponse {
  const runId = request.nextUrl.searchParams.get("run");
  const target = new URL("/start/questions", request.nextUrl.origin);
  if (runId !== null && runId.length > 0) target.searchParams.set("run", runId);
  return NextResponse.redirect(target, 308);
}
