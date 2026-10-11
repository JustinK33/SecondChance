"use client";

/**
 * Reading the program catalog.
 *
 * `GET /api/v1/programs` is the only unauthenticated data route, so these work
 * before anyone is identified.
 */

import { useMemo } from "react";

import { getProgram, listPrograms } from "../api-client";
import type { ListProgramsQuery, Program, ProgramDetail } from "../types";
import { useQuery } from "./use-async";
import type { QueryResult } from "./use-async";

export interface ProgramsResult extends QueryResult<{ items: Program[]; total: number }> {
  readonly programs: Program[];
  readonly total: number;
  /** True when the request succeeded and matched nothing, as opposed to not having run. */
  readonly isEmpty: boolean;
}

/**
 * A page of the catalog, optionally filtered.
 *
 * The query is compared by value, so a caller may pass a fresh object literal on
 * every render without causing a fetch loop.
 */
export function usePrograms(query?: ListProgramsQuery): ProgramsResult {
  const key = JSON.stringify(query ?? {});
  const result = useQuery(
    (signal) =>
      listPrograms(query, { signal }).then((page) => ({
        items: page.items ?? [],
        total: page.total,
      })),
    [key],
  );

  return useMemo(
    () => ({
      ...result,
      programs: result.data?.items ?? [],
      total: result.data?.total ?? 0,
      isEmpty: result.data !== null && result.data.items.length === 0,
    }),
    [result],
  );
}

export interface ProgramResult extends QueryResult<ProgramDetail> {
  readonly program: ProgramDetail | null;
}

/** One program with its sources, rendered requirements and rule version. */
export function useProgram(programId: string | null | undefined): ProgramResult {
  const id = typeof programId === "string" && programId.length > 0 ? programId : null;
  const result = useQuery((signal) => getProgram(id as string, { signal }), [id], {
    enabled: id !== null,
  });
  return useMemo(() => ({ ...result, program: result.data }), [result]);
}
