"use client";

import { useCallback, useEffect, useState } from "react";

/** Copy-to-clipboard control for extracting a disclosed figure without dragging across `[n]` citation markers. */
export function CopyButton({
  text,
  label,
  copiedLabel = "Copied",
  className = "",
}: {
  text: string;
  label: string;
  copiedLabel?: string;
  className?: string;
}) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const id = window.setTimeout(() => setState("idle"), 2000);
    return () => window.clearTimeout(id);
  }, [state]);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(text);
      setState("copied");
    } catch {
      setState("failed");
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={copy}
      // The accessible name carries what is being copied; the visible label stays short.
      aria-label={state === "idle" ? label : undefined}
      className={`rounded px-1.5 py-0.5 text-[0.7rem] text-muted transition-colors hover:bg-surface hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${className}`}
    >
      {state === "copied" ? copiedLabel : state === "failed" ? "Copy failed" : label}
    </button>
  );
}
