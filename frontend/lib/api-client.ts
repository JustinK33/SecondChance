/**
 * The only place in the application that calls `fetch`.
 *
 * One exported function per endpoint. Every one of them:
 *
 * - resolves against `NEXT_PUBLIC_API_URL`;
 * - attaches the Supabase access token, or the `X-Dev-User` header in dev mode;
 * - sends `X-Request-ID` so a user can quote a correlation id back to us;
 * - sends `Idempotency-Key` on the two endpoints that create work;
 * - parses the standard error envelope into an {@link ApiError} carrying `code`;
 * - gives up after a timeout rather than hanging a screen forever;
 * - retries *only* idempotent GETs, and only on a transport failure.
 *
 * What it deliberately does not do: retry a POST. `POST /workflows` is
 * idempotent only when the caller supplies a key, and a blind retry of
 * `/assistant/chat` would bill a second model call for one question.
 *
 * Privacy: no request body, narrative, income figure or token is ever written to
 * the console. The only thing logged on failure is the code and the request id.
 */

import { ApiError } from "./errors";
import { resolveCredential } from "./auth-token";
import { API_BASE_URL } from "./env";
import type {
  AccountDeletionReceipt,
  ActionItem,
  ActionItemPatch,
  ActionPlan,
  ActionPlanPage,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  IntakeRequest,
  IntakeResponse,
  ListProgramsQuery,
  MatchRequest,
  PaginationQuery,
  ProgramDetail,
  ProgramPage,
  ReadinessResponse,
  WorkflowCreateRequest,
  WorkflowCreateResponse,
  WorkflowResumeRequest,
  WorkflowRun,
  WorkflowRunPage,
} from "./types";

// ---------------------------------------------------------------------------
// tuning
// ---------------------------------------------------------------------------

/** Wall-clock budget for one attempt. The backend's own ceiling is 30s. */
const DEFAULT_TIMEOUT_MS = 20_000;

/** A poll must fail fast so the next tick is not queued behind it. */
const POLL_TIMEOUT_MS = 8_000;

/** Transport-failure retries for GETs only. Three attempts total. */
const GET_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 300;

export interface RequestOptions {
  /** Abort the request when this signal fires. Produces an `aborted` `ApiError`. */
  readonly signal?: AbortSignal;
  /** Override the per-attempt timeout. */
  readonly timeoutMs?: number;
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

function newRequestId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "");
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * A key the backend can deduplicate on, derived from nothing personal.
 *
 * The contract wants 8-64 characters. A random 32-char hex string satisfies that
 * and, because the caller holds it for the lifetime of the attempt, a retry of a
 * failed submission lands on the original run instead of starting a second one.
 */
export function newIdempotencyKey(): string {
  return newRequestId();
}

type QueryValue = string | number | boolean | readonly string[] | undefined | null;

function buildUrl(path: string, query?: Record<string, QueryValue>): string {
  const url = new URL(`${API_BASE_URL}${path}`);
  if (query !== undefined) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null) continue;
      if (Array.isArray(value)) {
        for (const member of value) url.searchParams.append(key, member);
      } else {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

/** Combine the caller's signal with our timeout so whichever fires first wins. */
function linkSignals(
  timeoutMs: number,
  external?: AbortSignal,
): {
  signal: AbortSignal;
  cleanup: () => void;
  timedOut: () => boolean;
} {
  const controller = new AbortController();
  let timedOut = false;

  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const onExternalAbort = (): void => controller.abort();
  if (external !== undefined) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", onExternalAbort, { once: true });
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      clearTimeout(timer);
      external?.removeEventListener("abort", onExternalAbort);
    },
    timedOut: () => timedOut,
  };
}

async function authHeaders(): Promise<Record<string, string>> {
  const credential = await resolveCredential();
  if (credential === null) return {};
  if (credential.kind === "supabase") {
    return { Authorization: `Bearer ${credential.token}` };
  }
  return { "X-Dev-User": credential.userId };
}

/** Turn a non-2xx response into an `ApiError`, preferring the server's envelope. */
async function errorFromResponse(response: Response): Promise<ApiError> {
  const headerRequestId = response.headers.get("X-Request-ID");
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (
    typeof body === "object" &&
    body !== null &&
    "error" in body &&
    typeof (body as { error: unknown }).error === "object" &&
    (body as { error: unknown }).error !== null
  ) {
    const detail = (body as { error: Record<string, unknown> }).error;
    const code = detail["code"];
    const message = detail["message"];
    if (typeof code === "string" && typeof message === "string") {
      return new ApiError({
        // The generated `ErrorCode` union is the contract; a value outside it
        // means the backend grew a code we have not regenerated types for, and
        // the honest thing is to carry it through rather than mislabel it.
        code: code as ApiError["code"],
        message,
        status: response.status,
        requestId:
          typeof detail["request_id"] === "string" ? detail["request_id"] : headerRequestId,
        detail: (detail["detail"] as Record<string, unknown> | null | undefined) ?? null,
      });
    }
  }

  // No envelope: a proxy, a gateway or a static error page answered.
  return new ApiError({
    code: response.status >= 500 ? "upstream_unavailable" : "internal_error",
    status: response.status,
    requestId: headerRequestId,
  });
}

interface FetchSpec {
  readonly method: "GET" | "POST" | "PATCH" | "DELETE";
  readonly path: string;
  readonly query?: Record<string, QueryValue>;
  readonly body?: unknown;
  readonly headers?: Record<string, string>;
  readonly options?: RequestOptions;
  /** Retry on transport failure. Only ever true for a GET. */
  readonly retries?: number;
  /**
   * Statuses to parse as a normal body instead of raising.
   *
   * Exists for `/readyz`, which answers 503 with a full `ReadinessResponse`
   * describing which dependency is down. That is information, not a failure.
   */
  readonly allowStatuses?: readonly number[];
}

async function request<T>(spec: FetchSpec): Promise<T> {
  const url = buildUrl(spec.path, spec.query);
  const timeoutMs = spec.options?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const retries = spec.method === "GET" ? (spec.retries ?? GET_RETRIES) : 0;
  // One id for the whole logical call, so a retried GET correlates to one entry.
  const requestId = newRequestId();

  let lastError: ApiError | null = null;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "X-Request-ID": requestId,
      ...(await authHeaders()),
      ...spec.headers,
    };
    if (spec.body !== undefined) headers["Content-Type"] = "application/json";

    const linked = linkSignals(timeoutMs, spec.options?.signal);
    try {
      const response = await fetch(url, {
        method: spec.method,
        headers,
        body: spec.body === undefined ? undefined : JSON.stringify(spec.body),
        signal: linked.signal,
        cache: "no-store",
        credentials: "omit",
      });

      const allowed = response.ok || (spec.allowStatuses?.includes(response.status) ?? false);
      if (!allowed) throw await errorFromResponse(response);

      if (response.status === 204) return undefined as T;
      const text = await response.text();
      if (text.length === 0) return undefined as T;
      try {
        return JSON.parse(text) as T;
      } catch (cause) {
        throw new ApiError({ code: "malformed_response", status: response.status, cause });
      }
    } catch (caught) {
      const error = classify(caught, linked.timedOut(), spec.options?.signal);
      lastError = error;
      // A transport failure is the only thing worth repeating, and only for a
      // GET. Anything the server answered is a decision, not a glitch.
      const transient = error.code === "network_error" || error.code === "client_timeout";
      if (attempt < retries && transient && !error.isAborted) {
        await sleep(RETRY_BASE_DELAY_MS * 2 ** attempt);
        continue;
      }
      throw error;
    } finally {
      linked.cleanup();
    }
  }

  throw lastError ?? new ApiError({ code: "internal_error" });
}

function classify(caught: unknown, timedOut: boolean, external?: AbortSignal): ApiError {
  if (caught instanceof ApiError) return caught;
  const isAbort =
    caught instanceof DOMException
      ? caught.name === "AbortError"
      : caught instanceof Error && caught.name === "AbortError";
  if (isAbort) {
    if (external?.aborted === true) return new ApiError({ code: "aborted", cause: caught });
    if (timedOut) return new ApiError({ code: "client_timeout", cause: caught });
    return new ApiError({ code: "aborted", cause: caught });
  }
  // `fetch` rejects with a TypeError for DNS, TLS, CORS and offline failures.
  return new ApiError({ code: "network_error", cause: caught });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// intake
// ---------------------------------------------------------------------------

/** Extract a structured profile from a narrative. One model call, no matching. */
export function createIntake(
  body: IntakeRequest,
  options?: RequestOptions,
): Promise<IntakeResponse> {
  return request<IntakeResponse>({ method: "POST", path: "/api/v1/intake", body, options });
}

// ---------------------------------------------------------------------------
// workflows
// ---------------------------------------------------------------------------

/**
 * Start a run. Returns 202 with a durable `run_id` to poll.
 *
 * `idempotencyKey` is sent as the `Idempotency-Key` header. Reuse the same key
 * when the user resubmits after a failure and the original run comes back with
 * `deduplicated: true` instead of a second run being created.
 */
export function createWorkflow(
  body: WorkflowCreateRequest,
  idempotencyKey: string,
  options?: RequestOptions,
): Promise<WorkflowCreateResponse> {
  return request<WorkflowCreateResponse>({
    method: "POST",
    path: "/api/v1/workflows",
    body,
    headers: { "Idempotency-Key": idempotencyKey },
    options,
  });
}

export function listWorkflows(
  query?: PaginationQuery,
  options?: RequestOptions,
): Promise<WorkflowRunPage> {
  return request<WorkflowRunPage>({
    method: "GET",
    path: "/api/v1/workflows",
    query: { limit: query?.limit, offset: query?.offset },
    options,
  });
}

export function getWorkflow(runId: string, options?: RequestOptions): Promise<WorkflowRun> {
  return request<WorkflowRun>({
    method: "GET",
    path: `/api/v1/workflows/${encodeURIComponent(runId)}`,
    options,
  });
}

/** The poll variant: a shorter timeout and no retry, because the next tick is the retry. */
export function pollWorkflow(runId: string, options?: RequestOptions): Promise<WorkflowRun> {
  return request<WorkflowRun>({
    method: "GET",
    path: `/api/v1/workflows/${encodeURIComponent(runId)}`,
    retries: 0,
    options: { timeoutMs: POLL_TIMEOUT_MS, ...options },
  });
}

/** Answer the clarification questions a run is parked on. 409 if it is not awaiting input. */
export function resumeWorkflow(
  runId: string,
  body: WorkflowResumeRequest,
  options?: RequestOptions,
): Promise<WorkflowRun> {
  return request<WorkflowRun>({
    method: "POST",
    path: `/api/v1/workflows/${encodeURIComponent(runId)}/resume`,
    body,
    options,
  });
}

export function cancelWorkflow(runId: string, options?: RequestOptions): Promise<WorkflowRun> {
  return request<WorkflowRun>({
    method: "POST",
    path: `/api/v1/workflows/${encodeURIComponent(runId)}/cancel`,
    options,
  });
}

/** Re-run matching after the user corrected their profile. Returns a new run to poll. */
export function createMatches(
  body: MatchRequest,
  idempotencyKey: string,
  options?: RequestOptions,
): Promise<WorkflowCreateResponse> {
  return request<WorkflowCreateResponse>({
    method: "POST",
    path: "/api/v1/matches",
    body,
    headers: { "Idempotency-Key": idempotencyKey },
    options,
  });
}

// ---------------------------------------------------------------------------
// programs - the only unauthenticated data route
// ---------------------------------------------------------------------------

export function listPrograms(
  query?: ListProgramsQuery,
  options?: RequestOptions,
): Promise<ProgramPage> {
  return request<ProgramPage>({
    method: "GET",
    path: "/api/v1/programs",
    query: {
      category: query?.category ?? undefined,
      state: query?.state ?? undefined,
      county: query?.county ?? undefined,
      q: query?.q ?? undefined,
      verification_status: query?.verification_status ?? undefined,
      limit: query?.limit,
      offset: query?.offset,
    },
    options,
  });
}

export function getProgram(programId: string, options?: RequestOptions): Promise<ProgramDetail> {
  return request<ProgramDetail>({
    method: "GET",
    path: `/api/v1/programs/${encodeURIComponent(programId)}`,
    options,
  });
}

// ---------------------------------------------------------------------------
// plans and action items
// ---------------------------------------------------------------------------

export function listPlans(
  query?: PaginationQuery,
  options?: RequestOptions,
): Promise<ActionPlanPage> {
  return request<ActionPlanPage>({
    method: "GET",
    path: "/api/v1/plans",
    query: { limit: query?.limit, offset: query?.offset },
    options,
  });
}

export function getPlan(planId: string, options?: RequestOptions): Promise<ActionPlan> {
  return request<ActionPlan>({
    method: "GET",
    path: `/api/v1/plans/${encodeURIComponent(planId)}`,
    options,
  });
}

export function patchActionItem(
  itemId: string,
  body: ActionItemPatch,
  options?: RequestOptions,
): Promise<ActionItem> {
  return request<ActionItem>({
    method: "PATCH",
    path: `/api/v1/action-items/${encodeURIComponent(itemId)}`,
    body,
    options,
  });
}

// ---------------------------------------------------------------------------
// assistant
// ---------------------------------------------------------------------------

/** Ask a grounded question. Never retried: a retry is a second model call. */
export function sendChat(body: ChatRequest, options?: RequestOptions): Promise<ChatResponse> {
  return request<ChatResponse>({ method: "POST", path: "/api/v1/assistant/chat", body, options });
}

// ---------------------------------------------------------------------------
// account and system
// ---------------------------------------------------------------------------

/** Irreversibly delete everything stored for the signed-in user. */
export function deleteAccountData(options?: RequestOptions): Promise<AccountDeletionReceipt> {
  return request<AccountDeletionReceipt>({
    method: "DELETE",
    path: "/api/v1/account/data",
    options,
  });
}

export function getHealth(options?: RequestOptions): Promise<HealthResponse> {
  return request<HealthResponse>({ method: "GET", path: "/healthz", options });
}

/** Readiness. A 503 body is parsed rather than thrown: it names the failing dependency. */
export function getReadiness(options?: RequestOptions): Promise<ReadinessResponse> {
  return request<ReadinessResponse>({
    method: "GET",
    path: "/readyz",
    allowStatuses: [503],
    options,
  });
}

export { ApiError } from "./errors";
export type { AnyErrorCode, ClientErrorCode } from "./errors";
