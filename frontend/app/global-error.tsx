"use client";

/**
 * The last-resort boundary, for a failure in the root layout itself.
 *
 * It has to render its own `<html>` and `<body>` because the layout that would
 * normally provide them is the thing that failed. For the same reason it uses
 * inline styles rather than Tailwind: the stylesheet may be what is broken.
 */
import Link from "next/link";

export default function GlobalError({ error }: { error: Error & { digest?: string } }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          padding: "3rem 1.25rem",
          fontFamily: "system-ui, sans-serif",
          lineHeight: 1.6,
          color: "#1b2227",
          background: "#fbfdfc",
        }}
      >
        <main style={{ maxWidth: "36rem", margin: "0 auto" }}>
          <h1 style={{ fontSize: "1.5rem", marginBottom: "0.75rem" }}>
            SecondChance could not start
          </h1>
          <p>
            Something went wrong before the page could load. Please reload, and if it keeps
            happening try again in a few minutes.
          </p>
          <p>
            <Link href="/" style={{ color: "#1d5b63" }}>
              Reload SecondChance
            </Link>
          </p>
          {error.digest === undefined ? null : (
            <p style={{ fontSize: "0.75rem", color: "#5a666d" }}>Reference: {error.digest}</p>
          )}
        </main>
      </body>
    </html>
  );
}
