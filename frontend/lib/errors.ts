/**
 * The typed error the data layer throws, and the words we show for it.
 *
 * The backend already returns a plain-language `message` that is safe to render
 * verbatim, and that message is always preferred. The local map exists for the
 * cases where there is no envelope to read: the network never answered, the
 * request timed out client-side, or a proxy returned HTML.
 */

import type { ApiErrorDetail, ErrorCode } from "./types";

/** Codes the backend marks retryable. Mirrors `AppError.is_retryable`. */
const RETRYABLE_CODES: ReadonlySet<string> = new Set<ErrorCode>([
  "rate_limited",
  "invalid_model_output",
  "upstream_unavailable",
  "upstream_timeout",
  "workflow_failed",
]);

/**
 * Client-side pseudo-codes for failures that never reached the application.
 *
 * They are deliberately not `ErrorCode` members - inventing a backend code
 * would let a UI branch on something the backend cannot produce.
 */
export type ClientErrorCode = "network_error" | "client_timeout" | "malformed_response" | "aborted";

export type AnyErrorCode = ErrorCode | ClientErrorCode;

/** Fallbacks used only when there is no server message to show. */
const FALLBACK_MESSAGES: Record<AnyErrorCode, string> = {
  validation_error: "Some of the information sent was not valid. Please check it and try again.",
  unauthorized: "You need to sign in again before we can do that.",
  forbidden: "That is not something this account can open.",
  not_found: "We could not find that. It may have been removed.",
  conflict: "That action has already been taken, so we did not repeat it.",
  rate_limited: "We are getting a lot of requests right now. Please wait a moment and try again.",
  upstream_timeout: "This is taking longer than expected. Please try again.",
  upstream_unavailable:
    "A service we rely on is temporarily unavailable. Please try again shortly.",
  invalid_model_output: "We could not read that result clearly. Please try again.",
  workflow_failed: "Something went wrong while working through your situation. You can try again.",
  internal_error: "Something went wrong on our side. Please try again.",
  network_error: "We could not reach the server. Check your connection and try again.",
  client_timeout: "The request took too long and we stopped waiting. Please try again.",
  malformed_response: "The server sent something we could not read. Please try again.",
  aborted: "That request was cancelled.",
};

/** Short, non-blaming guidance shown under the message where a screen wants it. */
const HINTS: Partial<Record<AnyErrorCode, string>> = {
  unauthorized: "Sign in again to pick up where you left off.",
  rate_limited: "Waiting about a minute is usually enough.",
  network_error: "Your answers are saved on this device, so nothing is lost.",
  client_timeout: "Your answers are saved on this device, so nothing is lost.",
  workflow_failed: "You can start again without retyping anything.",
};

export interface ApiErrorOptions {
  readonly code: AnyErrorCode;
  readonly message?: string | null;
  readonly status?: number;
  readonly requestId?: string | null;
  readonly detail?: Record<string, unknown> | null;
  readonly cause?: unknown;
}

/**
 * A failed API call, carrying the contract's `code` so a screen can branch on it.
 *
 * `message` is always populated and always safe to render: either the server's
 * plain-language text or our own fallback for the same code.
 */
export class ApiError extends Error {
  readonly code: AnyErrorCode;
  readonly status: number | null;
  readonly requestId: string | null;
  readonly detail: Record<string, unknown> | null;

  constructor(options: ApiErrorOptions) {
    const message =
      typeof options.message === "string" && options.message.trim().length > 0
        ? options.message.trim()
        : FALLBACK_MESSAGES[options.code];
    super(message);
    this.name = "ApiError";
    this.code = options.code;
    this.status = options.status ?? null;
    this.requestId = options.requestId ?? null;
    this.detail = options.detail ?? null;
    if (options.cause !== undefined) {
      this.cause = options.cause;
    }
  }

  /** Build from a parsed error envelope. */
  static fromEnvelope(detail: ApiErrorDetail, status: number): ApiError {
    return new ApiError({
      code: detail.code,
      message: detail.message,
      status,
      requestId: detail.request_id ?? null,
      detail: (detail.detail as Record<string, unknown> | null | undefined) ?? null,
    });
  }

  /** True when trying the same thing again could plausibly succeed. */
  get isRetryable(): boolean {
    if (this.code === "network_error" || this.code === "client_timeout") return true;
    if (this.code === "malformed_response") return true;
    return RETRYABLE_CODES.has(this.code);
  }

  /** True when the right response is to send the user back through sign-in. */
  get isAuthFailure(): boolean {
    return this.code === "unauthorized";
  }

  /** True when the user cancelled or navigated away; screens should stay quiet. */
  get isAborted(): boolean {
    return this.code === "aborted";
  }

  /** Optional second line of guidance, or `null` when there is nothing useful to add. */
  get hint(): string | null {
    return HINTS[this.code] ?? null;
  }

  /** The field a validation error points at, when the backend named one. */
  get field(): string | null {
    const value = this.detail?.["field"];
    return typeof value === "string" ? value : null;
  }
}

/** The human sentence for any thrown value, including ones that are not `ApiError`. */
export function messageForError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.message.trim().length > 0) {
    // Not a contract error: do not show the raw text, it may be a stack-ish string.
    return FALLBACK_MESSAGES.internal_error;
  }
  return FALLBACK_MESSAGES.internal_error;
}

/** Coerce any thrown value into an `ApiError` so callers have one shape to handle. */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  return new ApiError({ code: "internal_error", cause: error });
}

/** The fallback sentence for a code, for UI that wants it without an instance. */
export function fallbackMessageFor(code: AnyErrorCode): string {
  return FALLBACK_MESSAGES[code];
}
