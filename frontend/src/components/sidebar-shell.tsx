"use client";

import { useState } from "react";

import { SidebarNav } from "./sidebar-nav";

/** Owns the sidebar's open/closed state and the toggle that controls it. */
export function SidebarShell() {
  const [open, setOpen] = useState(true);

  return (
    <>
      <div
        className={`shrink-0 overflow-hidden transition-[width] duration-200 ease-in-out ${
          open ? "w-56" : "w-0"
        }`}
      >
        {open ? <SidebarNav /> : null}
      </div>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label={open ? "Close sidebar" : "Open sidebar"}
        className="mt-4 ml-3 flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border text-muted transition-colors hover:bg-surface-raised hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <svg
          viewBox="0 0 20 20"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          className="h-4 w-4"
          aria-hidden="true"
        >
          <line x1="3" y1="5.5" x2="17" y2="5.5" />
          <line x1="3" y1="10" x2="17" y2="10" />
          <line x1="3" y1="14.5" x2="17" y2="14.5" />
        </svg>
      </button>
    </>
  );
}
