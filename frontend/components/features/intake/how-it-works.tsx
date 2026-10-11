import { ClipboardList, ListChecks, MessageSquareText } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * Three steps, in the order they happen.
 *
 * Written for someone who has never used a benefits screener and has no reason
 * to trust one. Each step says what *they* do or get, not what the system does
 * internally - there is nothing here about models, retrieval or pipelines,
 * because none of that helps anyone decide whether to start.
 */

const STEPS: readonly { icon: ReactNode; title: string; body: string }[] = [
  {
    icon: <MessageSquareText aria-hidden="true" className="size-6" />,
    title: "1. You tell us what is going on",
    body: "Write it however you like, the way you would tell a friend. A few sentences is enough. You can skip anything you would rather not share.",
  },
  {
    icon: <ClipboardList aria-hidden="true" className="size-6" />,
    title: "2. We check it against real programs",
    body: "We compare what you told us with the published rules for New Jersey and Rutgers assistance programs. If something we need is missing, we ask you instead of guessing.",
  },
  {
    icon: <ListChecks aria-hidden="true" className="size-6" />,
    title: "3. You get a plan you can act on",
    body: "A short list of programs worth exploring, why each one came up, what to bring, and the official link to apply. You can save it or print it.",
  },
];

export function HowItWorks({ className }: { className?: string }): ReactNode {
  return (
    <section aria-labelledby="how-it-works-heading" className={cn("space-y-4", className)}>
      <h2 id="how-it-works-heading" className="text-xl font-semibold sm:text-2xl">
        How it works
      </h2>
      <ol className="grid gap-4 sm:grid-cols-3">
        {STEPS.map((step) => (
          <li key={step.title} className="bg-card space-y-2 rounded-lg border p-4">
            <span className="text-primary inline-flex">{step.icon}</span>
            <h3 className="text-base font-semibold">{step.title}</h3>
            <p className="text-muted-foreground text-sm">{step.body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
