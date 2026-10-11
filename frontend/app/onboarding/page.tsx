import { redirect } from "next/navigation";

/**
 * `/onboarding` is an alias for the canonical `/start`.
 *
 * `ROUTES.start` is what the header, the landing screen and the nav all link
 * to; this path exists only so an old link or a typed guess does not 404.
 */
export default function OnboardingAliasPage(): never {
  redirect("/start");
}
