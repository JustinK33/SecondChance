import type { ReactNode } from "react";

import { PrivacyNote } from "@/components/ui/privacy-note";

/**
 * Exactly what is collected, and the things that are never asked for.
 *
 * Specific rather than reassuring. "We take your privacy seriously" tells
 * someone nothing; a list of the four things we store and the three things we
 * will never ask for lets them decide for themselves.
 *
 * Every claim here is structurally true of the system, not a promise bolted on
 * afterwards: there is no field anywhere in the application for a Social
 * Security number or bank details, and the logger redacts personal content.
 */
export function WhatWeCollect({ className }: { className?: string }): ReactNode {
  return (
    <PrivacyNote className={className}>
      <h2 className="text-base font-semibold">What we collect, and what we never ask for</h2>
      <p>What you give us:</p>
      <ul className="list-disc space-y-1 pl-5">
        <li>What you write about your situation.</li>
        <li>Where you live - the state and county, not your street address.</li>
        <li>How many people are in your household.</li>
        <li>A rough income range. A range, never an exact figure.</li>
      </ul>
      <p className="pt-1">
        We never ask for your Social Security number, your bank or card details, a password, or an
        immigration document number. No program we list needs those from us, and there is nowhere on
        this site to type them.
      </p>
      <p>
        What you write is used to match you to programs and nothing else. It is never written into
        our logs, never sold, and never shared with an agency. You can delete everything we hold at
        any time, and we will show you exactly what was removed.
      </p>
    </PrivacyNote>
  );
}
