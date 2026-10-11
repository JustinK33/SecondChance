import { redirect } from "next/navigation";

/**
 * `/login` is an alias.
 *
 * The canonical sign-in path is `ROUTES.signIn` (`/sign-in`), which is what the
 * header, the middleware and every other screen link to. `/login` exists
 * because it is the first thing people type, and a 404 on that guess is a lost
 * user. Permanent redirect so it is not indexed as a second page.
 */
export default function LoginAliasPage(): never {
  redirect("/sign-in");
}
