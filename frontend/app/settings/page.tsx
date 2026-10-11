import { redirect } from "next/navigation";

/**
 * `/settings` is an alias for the canonical `/account`.
 *
 * `ROUTES.account` is what the header, the footer and the nav link to. This
 * path exists because "settings" is what people type when they are looking for
 * the delete button.
 */
export default function SettingsAliasPage(): never {
  redirect("/account");
}
