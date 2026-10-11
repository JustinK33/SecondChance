import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { AppShell } from "@/components/layout/app-shell";
import { Providers } from "@/lib/providers";

import "./globals.css";

/**
 * Inter rather than a display face. This interface is read, often quickly and
 * often on a phone, by someone who may be stressed. Legibility wins.
 */
const sans = Inter({
  variable: "--font-app-sans",
  subsets: ["latin"],
  display: "swap",
});

const mono = JetBrains_Mono({
  variable: "--font-app-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "SecondChance - find assistance programs in New Jersey",
    template: "%s - SecondChance",
  },
  description:
    "Describe your situation in your own words and see which New Jersey and Rutgers assistance programs may be worth exploring, with the official link for each one. An independent project, not a government agency.",
  applicationName: "SecondChance",
  // No search indexing while the data is still being verified. Publishing an
  // unverified benefits directory to search results would do real harm.
  robots: { index: false, follow: false },
  openGraph: {
    title: "SecondChance",
    description:
      "Find New Jersey and Rutgers assistance programs that may be worth exploring, with official sources for every claim.",
    type: "website",
  },
  formatDetection: { telephone: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Never block zoom. Someone who needs to magnify text must be able to.
  maximumScale: 5,
  // Only the light palette is actually rendered today: `.dark` exists in the
  // stylesheet but nothing toggles it. Claiming "light dark" made the UA draw
  // native checkboxes and radios dark on a light page.
  colorScheme: "light",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbfdfc" },
    { media: "(prefers-color-scheme: dark)", color: "#13191d" },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
