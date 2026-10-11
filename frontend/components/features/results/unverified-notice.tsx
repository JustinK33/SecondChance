import { ExternalLink, ShieldQuestion } from "lucide-react";
import type { ReactNode } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { verificationPresentation } from "@/lib/status";
import type { VerificationStatus } from "@/lib/types";

/**
 * The banner that says "we could not confirm this".
 *
 * A `NEEDS_REVIEW` or `UNVERIFIED` program is still shown - hiding a real
 * program helps nobody - but the honest thing is to say plainly that its
 * requirements are not confirmed and point at the official page. The backend
 * already excludes these programs from eligibility claims; this is the same
 * fact, visible.
 *
 * Renders nothing for a `VERIFIED` program, so callers can drop it in
 * unconditionally.
 */
export function UnverifiedNotice({
  status,
  programName,
  officialUrl,
}: {
  status: VerificationStatus;
  programName?: string;
  officialUrl?: string | null;
}): ReactNode {
  const presentation = verificationPresentation(status);
  if (presentation.supportsClaims) return null;

  return (
    <Alert>
      <ShieldQuestion aria-hidden="true" className="size-4" />
      <AlertTitle>
        {status === "NEEDS_REVIEW"
          ? "We are re-checking these requirements"
          : "We could not confirm these requirements"}
      </AlertTitle>
      <AlertDescription>
        <p>{presentation.description}</p>
        <p>
          Please check
          {typeof programName === "string" && programName.length > 0
            ? ` ${programName}'s`
            : " the program's"}{" "}
          official page before you rely on anything here.
        </p>
        {typeof officialUrl === "string" && officialUrl.length > 0 ? (
          <a
            href={officialUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="link touch-row inline-flex items-center gap-1 text-sm font-medium"
          >
            Open the official page
            <ExternalLink aria-hidden="true" className="size-3.5" />
            <span className="sr-only">(opens in a new tab, leaving SecondChance)</span>
          </a>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}
