/**
 * Every route in the application, named once.
 *
 * A `Link` built from a literal string is a broken link waiting to happen, and
 * the screens are owned by several people. These helpers are the agreed spelling
 * - if a screen needs a different URL, change it here and every nav item,
 * breadcrumb and redirect follows.
 */

export const ROUTES = {
  home: "/",
  /** Onboarding: describe your situation in your own words. */
  start: "/start",
  /** A run in flight, with progress and any clarification questions. */
  run: (runId: string): string => `/runs/${encodeURIComponent(runId)}`,
  /** The results of a finished run. */
  results: (runId: string): string => `/runs/${encodeURIComponent(runId)}/results`,
  /** The browsable catalog. */
  programs: "/programs",
  program: (programId: string): string => `/programs/${encodeURIComponent(programId)}`,
  /** The action plan for a run. */
  plan: (planId: string): string => `/plans/${encodeURIComponent(planId)}`,
  plans: "/plans",
  /** Grounded question answering. */
  assistant: "/assistant",
  /** Profile review, privacy and data deletion. */
  account: "/account",
  privacy: "/privacy",
  signIn: "/sign-in",
} as const;

export interface NavItem {
  readonly href: string;
  readonly label: string;
  /** Shown under the label in the mobile sheet, where there is room for it. */
  readonly description: string;
  /** True when this item requires an identity. */
  readonly requiresAuth: boolean;
}

/** The primary navigation, in the order a new user should meet it. */
export const PRIMARY_NAV: readonly NavItem[] = [
  {
    href: ROUTES.start,
    label: "Find help",
    description: "Describe your situation and see what you may be able to get.",
    requiresAuth: false,
  },
  {
    href: ROUTES.programs,
    label: "Browse programs",
    description: "Every program we have checked, with its official source.",
    requiresAuth: false,
  },
  {
    href: ROUTES.plans,
    label: "My plan",
    description: "The steps from your last run, and what you have finished.",
    requiresAuth: true,
  },
  {
    href: ROUTES.assistant,
    label: "Ask a question",
    description: "Answers drawn only from the sources we have on file.",
    requiresAuth: true,
  },
  {
    href: ROUTES.account,
    label: "Your information",
    description: "Review what we hold, correct it, or delete all of it.",
    requiresAuth: true,
  },
] as const;

/** True when `pathname` is inside `href`, for marking the current nav item. */
export function isActivePath(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}
