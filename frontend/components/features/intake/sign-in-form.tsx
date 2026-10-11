"use client";

/**
 * Sign-in, which is optional on purpose.
 *
 * An account buys exactly one thing - your plan is still there tomorrow, on
 * whatever device you pick up - and the screen says so in one line rather than
 * selling it. The core flow is never behind it: there is always a way forward
 * without signing in, and that way forward is a real button, not a link in
 * small print.
 *
 * Sign-in is a magic link, so there is no password to remember or to leak. What
 * is possible depends on how the installation is configured, and the two cases
 * are told apart honestly rather than papered over:
 *
 * - Supabase configured: the email is sent and the link lands on `/auth/callback`.
 * - Local development identity: there is no sign-in to offer, so the form is not
 *   shown at all and the local identity is explained instead of being dressed up
 *   as an account.
 */

import Link from "next/link";
import { useId, useState } from "react";
import type { ReactNode } from "react";
import { MailCheck } from "lucide-react";

import { ROUTES } from "@/components/layout/routes";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/lib/auth";

/** The loosest check that still catches a typo. Delivery is the real validator. */
function looksLikeEmail(value: string): boolean {
  const trimmed = value.trim();
  if (trimmed.length < 5 || trimmed.length > 254) return false;
  if (/\s/.test(trimmed)) return false;
  const at = trimmed.indexOf("@");
  if (at <= 0 || at !== trimmed.lastIndexOf("@")) return false;
  const domain = trimmed.slice(at + 1);
  return domain.includes(".") && !domain.startsWith(".") && !domain.endsWith(".");
}

export function SignInForm({ next }: { next?: string }): ReactNode {
  const { isDevIdentity, signInWithEmail, error, magicLinkSentTo, clearError } = useAuth();

  const emailId = useId();
  const emailHintId = `${emailId}-hint`;
  const emailErrorId = `${emailId}-error`;

  const [email, setEmail] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);

  const continueWithoutAccount = (
    <div className="space-y-2">
      <Button
        asChild
        variant={isDevIdentity ? "default" : "outline"}
        size="lg"
        className="touch-row w-full px-6 text-base sm:w-auto"
      >
        <Link href={isDevIdentity ? ROUTES.start : ROUTES.programs}>
          {isDevIdentity ? "Continue without an account" : "Browse programs without an account"}
        </Link>
      </Button>
      <p className="text-muted-foreground text-sm">
        {isDevIdentity
          ? "Your answers stay on this device. Nothing is emailed to you and no account is created."
          : "You can read every program and its official link without signing in. A saved plan needs an account."}
      </p>
    </div>
  );

  if (isDevIdentity) {
    return (
      <div className="space-y-6">
        <div className="bg-card space-y-2 rounded-lg border p-4">
          <h2 className="text-base font-semibold">No sign-in on this installation</h2>
          <p className="text-sm">
            This copy of SecondChance is running in local test mode, so there are no accounts and no
            emails. You are identified only by a random id stored in this browser, and your
            information is kept on this device and in this local database.
          </p>
        </div>
        {continueWithoutAccount}
      </div>
    );
  }

  if (magicLinkSentTo !== null) {
    return (
      <div className="space-y-6">
        <div
          role="status"
          aria-live="polite"
          className="bg-card flex items-start gap-3 rounded-lg border p-4"
        >
          <MailCheck aria-hidden="true" className="text-primary mt-0.5 size-5 shrink-0" />
          <div className="space-y-2">
            <h2 className="text-base font-semibold">Check your email</h2>
            <p className="text-sm">
              We sent a sign-in link to that address. Open it on this device and you will be signed
              in - there is no code to type and no password to pick.
            </p>
            <p className="text-muted-foreground text-sm">
              The link works once and expires after about an hour. If it does not arrive in a few
              minutes, check your spam folder, then request another one.
            </p>
          </div>
        </div>
        {continueWithoutAccount}
      </div>
    );
  }

  const shownError = localError ?? error;

  const onSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    clearError();
    if (!looksLikeEmail(email)) {
      setLocalError("Enter an email address we can send the link to, like you@example.com.");
      return;
    }
    setLocalError(null);
    setIsSending(true);
    void signInWithEmail(email, next).finally(() => setIsSending(false));
  };

  return (
    <div className="space-y-8">
      <form noValidate onSubmit={onSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor={emailId} className="text-base">
            Your email address
          </Label>
          <p id={emailHintId} className="text-muted-foreground text-sm">
            We email you a link that signs you in. No password to remember.
          </p>
          <Input
            id={emailId}
            name="email"
            type="email"
            inputMode="email"
            autoComplete="email"
            autoCapitalize="none"
            spellCheck={false}
            required
            value={email}
            onChange={(event) => {
              setEmail(event.target.value);
              if (localError !== null) setLocalError(null);
            }}
            aria-invalid={shownError !== null ? true : undefined}
            aria-describedby={shownError !== null ? `${emailHintId} ${emailErrorId}` : emailHintId}
            className="touch-row h-12 text-base"
          />
          <p
            id={emailErrorId}
            role="alert"
            aria-live="assertive"
            className="text-destructive text-sm empty:hidden"
          >
            {shownError ?? ""}
          </p>
        </div>
        <Button
          type="submit"
          size="lg"
          disabled={isSending}
          className="touch-row w-full px-6 text-base sm:w-auto"
        >
          {isSending ? "Sending the link..." : "Email me a sign-in link"}
        </Button>
        <p className="text-muted-foreground text-sm">
          An account saves your plan so you can come back to it later or on another phone. That is
          all it does.
        </p>
      </form>

      <div className="space-y-2 border-t pt-6">
        <h2 className="text-base font-semibold">You do not have to sign in</h2>
        {continueWithoutAccount}
      </div>
    </div>
  );
}
