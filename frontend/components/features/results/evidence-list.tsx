import { Check, CircleHelp, Minus } from "lucide-react";
import type { ReactNode } from "react";

import { trileanLabel } from "@/lib/status";
import type { ConditionEvidence, Trilean } from "@/lib/types";
import { cn } from "@/lib/utils";

import { describeMissingFields } from "./field-labels";

/**
 * The rule engine's reasoning, in human terms.
 *
 * `ConditionEvidence` is a tree: `all_of` / `any_of` / `not` carry children and
 * the leaves carry the checks that actually happened. A user does not need the
 * boolean algebra, they need the list of checks and how each one came out, so
 * the tree is flattened to its leaves and the leaves keep their author-written
 * `description` verbatim. Nothing here invents a reason.
 *
 * Every row states its outcome in words as well as with an icon, because the
 * difference between "we checked this and it matched" and "we could not check
 * this" is the whole point and must survive greyscale and a screen reader.
 */
export interface EvidenceLeaf {
  readonly key: string;
  readonly description: string;
  readonly outcome: Trilean;
  readonly thresholdApplied: string | null;
  readonly missingFields: readonly string[];
}

/** Depth-first leaves of an evidence tree, in rule order. */
export function flattenEvidence(
  evidence: readonly ConditionEvidence[] | undefined,
  prefix = "e",
): EvidenceLeaf[] {
  const leaves: EvidenceLeaf[] = [];
  (evidence ?? []).forEach((node, index) => {
    const key = `${prefix}.${index}`;
    const children = node.children ?? [];
    if (children.length > 0) {
      leaves.push(...flattenEvidence(children, key));
      return;
    }
    leaves.push({
      key,
      description: node.description,
      outcome: node.outcome,
      thresholdApplied: node.threshold_applied ?? null,
      missingFields: node.missing_fields ?? [],
    });
  });
  return leaves;
}

/** The leaves with a given outcome, which is how the UI groups them. */
export function evidenceWithOutcome(
  evidence: readonly ConditionEvidence[] | undefined,
  outcome: Trilean,
): EvidenceLeaf[] {
  return flattenEvidence(evidence).filter((leaf) => leaf.outcome === outcome);
}

function OutcomeIcon({ outcome }: { outcome: Trilean }): ReactNode {
  const Icon = outcome === "TRUE" ? Check : outcome === "UNKNOWN" ? CircleHelp : Minus;
  return <Icon aria-hidden="true" className="mt-0.5 size-4 shrink-0" />;
}

export function EvidenceRow({ leaf }: { leaf: EvidenceLeaf }): ReactNode {
  const missing = describeMissingFields(leaf.missingFields);
  return (
    <li className="flex items-start gap-2 text-sm">
      <OutcomeIcon outcome={leaf.outcome} />
      <div className="space-y-1">
        <p>
          {leaf.description}{" "}
          <span className="text-muted-foreground">({trileanLabel(leaf.outcome)})</span>
        </p>
        {leaf.thresholdApplied !== null ? (
          <p className="text-muted-foreground text-xs">Checked against {leaf.thresholdApplied}.</p>
        ) : null}
        {missing.length > 0 ? (
          <p className="text-muted-foreground text-xs">
            We would need to know {missing.join(", and ")}.
          </p>
        ) : null}
      </div>
    </li>
  );
}

/**
 * A titled list of evidence rows.
 *
 * `heading` is rendered as an `h4` so it nests under a card's `h3`; pass
 * `headingLevel` where the surrounding structure differs.
 */
export function EvidenceList({
  leaves,
  heading,
  description,
  headingLevel = 4,
  className,
}: {
  leaves: readonly EvidenceLeaf[];
  heading: string;
  description?: string;
  headingLevel?: 3 | 4 | 5;
  className?: string;
}): ReactNode {
  if (leaves.length === 0) return null;
  const Heading = `h${headingLevel}` as "h3" | "h4" | "h5";
  return (
    <section className={cn("space-y-2", className)}>
      <Heading className="text-sm font-semibold">{heading}</Heading>
      {description !== undefined ? (
        <p className="text-muted-foreground text-xs">{description}</p>
      ) : null}
      <ul className="space-y-2">
        {leaves.map((leaf) => (
          <EvidenceRow key={leaf.key} leaf={leaf} />
        ))}
      </ul>
    </section>
  );
}
