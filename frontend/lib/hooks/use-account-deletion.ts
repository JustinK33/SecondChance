"use client";

/**
 * "Delete everything you have about me."
 *
 * The backend hard-deletes the user's rows and returns a receipt listing what it
 * removed and what it kept, so the promise is auditable rather than rhetorical.
 * In local development mode the stored dev identity is forgotten too, otherwise
 * the next request would re-create data under the same id and the deletion would
 * look like it had not worked.
 */

import { useCallback } from "react";

import { deleteAccountData } from "../api-client";
import type { ApiError } from "../errors";
import { useAuth } from "../auth";
import type { AccountDeletionReceipt } from "../types";
import { useAction } from "./use-async";

export interface AccountDeletionResult {
  /** Perform the deletion. Resolves with the receipt, or `null` on failure. */
  deleteEverything(): Promise<AccountDeletionReceipt | null>;
  readonly receipt: AccountDeletionReceipt | null;
  readonly isPending: boolean;
  readonly error: ApiError | null;
  /** True once the deletion has succeeded, for a confirmation screen. */
  readonly isDeleted: boolean;
  reset(): void;
}

export function useAccountDeletion(): AccountDeletionResult {
  const { isDevIdentity, resetLocalIdentity, signOut } = useAuth();
  const action = useAction(() => deleteAccountData());

  const deleteEverything = useCallback(async (): Promise<AccountDeletionReceipt | null> => {
    const receipt = await action.run();
    if (receipt === null) return null;
    if (isDevIdentity) resetLocalIdentity();
    else await signOut();
    return receipt;
  }, [action, isDevIdentity, resetLocalIdentity, signOut]);

  return {
    deleteEverything,
    receipt: action.data,
    isPending: action.isPending,
    error: action.error,
    isDeleted: action.data !== null,
    reset: action.reset,
  };
}
