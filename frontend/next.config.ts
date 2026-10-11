import type { NextConfig } from "next";

/**
 * Next configuration.
 *
 * `cacheComponents` and `partialPrefetching` come from the generated scaffold
 * and are left on: every data-bearing page in this app is client-rendered
 * through `lib/hooks`, so there is no server cache for them to stale.
 *
 * Security headers live in `proxy.ts` rather than here, because the CSP
 * needs the configured API origin and `headers()` cannot read it per-request.
 */
const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // Do not leak the framework version to every response.
  poweredByHeader: false,
  // Next writes an AGENTS.md into the project on dev start. This repository
  // keeps its own documentation under docs/, so the generated file is off.
  agentRules: false,
  reactStrictMode: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  typescript: {
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
