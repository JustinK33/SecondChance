import type { Metadata } from "next";
import type { ReactNode } from "react";

import {
  DeleteDataDialog,
  IdentityNote,
  PrivacyExplainer,
  ProfileEditor,
} from "@/components/features/settings";

export const metadata: Metadata = {
  title: "Your information",
};

/**
 * The "your information" screen: review, correct, understand, delete.
 *
 * Order matters. What we hold comes first, because that is what someone came to
 * check. How we handle it comes second. Deleting it comes last, separated by a
 * rule, so it is never the thing a thumb lands on by accident.
 */
export default function AccountPage(): ReactNode {
  return (
    <div className="page-container py-8 sm:py-12">
      <div className="reading-column space-y-10">
        <header className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Your information</h1>
          <p className="text-muted-foreground">
            Everything we hold about you, in one place. You can correct it or remove all of it.
          </p>
        </header>

        <IdentityNote />
        <ProfileEditor />
        <PrivacyExplainer />

        <div className="border-t pt-10">
          <DeleteDataDialog />
        </div>
      </div>
    </div>
  );
}
