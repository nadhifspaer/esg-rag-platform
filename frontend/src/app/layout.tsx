import type { Metadata } from "next";
import localFont from "next/font/local";

import { SidebarShell } from "@/components/sidebar-shell";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

/** `title.template` appends the product name to every page title; `default` covers the home page. */
export const metadata: Metadata = {
  title: {
    default: "ESG Intelligence Platform · Indonesian Banking",
    template: "%s · ESG Intelligence Platform",
  },
  description:
    "Grounded, cited answers over Indonesian banking sustainability disclosures and ESG " +
    "reporting standards.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    // `lang="en"` is the interface language; Indonesian quotes carry their own `lang="id"`.
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}>
        {/* Keyboard users land here first; the target is the <main> below. */}
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-foreground"
        >
          Skip to content
        </a>
        <div className="flex h-screen overflow-hidden">
          <SidebarShell />
          <div className="flex min-h-0 flex-1 flex-col">
            <main id="main" className="min-h-0 flex-1 overflow-y-auto">
              {children}
            </main>
            <footer className="shrink-0 border-t border-border px-6 py-4 text-xs text-muted">
              <p className="mx-auto max-w-6xl text-center">
                © 2025 Nadhif Spaer. All rights reserved.
              </p>
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
