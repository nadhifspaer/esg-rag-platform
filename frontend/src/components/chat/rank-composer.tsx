"use client";

import { useState } from "react";

/** Rank-banks composer mode, wired directly to `/rank`, skipping the classifier round trip. */
export function RankComposer({
  onSubmit,
  onStop,
  isStreaming,
  disabled,
}: {
  onSubmit: (query: string) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}) {
  const [query, setQuery] = useState("");

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || disabled) return;
    setQuery("");
    onSubmit(trimmed);
  }

  return (
    <form onSubmit={submit} className="shrink-0 space-y-2 bg-background pt-2">
      <label htmlFor="rank-query" className="sr-only">
        Metric or requirement to rank banks on
      </label>
      <div className="flex items-end gap-2">
        <input
          id="rank-query"
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="e.g. GRI 401-1, or “earliest net-zero target”"
          disabled={disabled}
          className="min-h-[3rem] flex-1 rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-60"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-lg border border-border px-4 py-2.5 text-sm text-muted transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!query.trim() || disabled}
            className="rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-accent-foreground transition-opacity hover:opacity-90 disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            Rank
          </button>
        )}
      </div>
      <p className="text-xs text-muted">Runs across all 21 banks — takes up to about a minute.</p>
    </form>
  );
}
