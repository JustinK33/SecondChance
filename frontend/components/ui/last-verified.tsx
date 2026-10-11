import { ShieldCheck, ShieldQuestion } from "lucide-react";
import type { ReactNode } from "react";

import { formatDate, formatRelativeAge } from "@/lib/format";
import { verificationPresentation } from "@/lib/status";
import type { VerificationStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * When we last checked a program against its official source.
 *
 * Shown next to every program, because "we checked this in March" and "we have
 * never been able to check this" are very different things to be acting on, and
 * hiding the difference would be the most damaging kind of polish.
 *
 * An unverified program is still displayed. The product rule is that unverified
 * data is labelled and excluded from eligibility claims, not concealed.
 */
export function LastVerified({
  status,
  lastVerifiedAt,
  className,
}: {
  status: VerificationStatus;
  lastVerifiedAt?: string | null;
  className?: string;
}): ReactNode {
  const presentation = verificationPresentation(status);
  const exact = formatDate(lastVerifiedAt);
  const relative = formatRelativeAge(lastVerifiedAt);
  const Icon = presentation.supportsClaims ? ShieldCheck : ShieldQuestion;

  return (
    <p
      className={cn("text-muted-foreground flex flex-wrap items-center gap-1.5 text-xs", className)}
    >
      <Icon aria-hidden="true" className="size-3.5 shrink-0" />
      <span>{presentation.label}</span>
      {exact !== null ? (
        <span>
          {"·"} last checked <time dateTime={lastVerifiedAt ?? undefined}>{exact}</time>
          {relative === null ? "" : ` (${relative})`}
        </span>
      ) : (
        <span>{"·"} no check on record</span>
      )}
    </p>
  );
}
