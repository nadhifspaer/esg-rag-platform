"use client";

import { useEffect, useState } from "react";

import { getDocuments } from "@/lib/documents";

import { CategoryFrameworkPicker } from "./category-picker";

/** Compliance-check composer mode, wired directly to `/compliance-check`, skipping the `/chat` classifier. */
export function ComplianceComposer({
  onSubmit,
  onStop,
  isStreaming,
  disabled,
}: {
  onSubmit: (
    company: string,
    category: "environmental" | "social" | "governance",
    framework: "gri" | "edgb",
  ) => void;
  onStop: () => void;
  isStreaming: boolean;
  disabled?: boolean;
}) {
  const [banks, setBanks] = useState<string[] | null>(null);
  const [bankLoadFailed, setBankLoadFailed] = useState(false);
  const [company, setCompany] = useState("");
  const [framework, setFramework] = useState<"gri" | "edgb">("gri");

  // Fetched lazily: this component only mounts once a visitor switches into Compliance-check
  // mode, so Ask mode (the default) never pays for a request it doesn't need.
  useEffect(() => {
    let cancelled = false;
    getDocuments("company_documents").then((documents) => {
      if (cancelled) return;
      if (!documents) {
        setBankLoadFailed(true);
        return;
      }
      const names = Array.from(
        new Set(
          documents.map((doc) => doc.bank).filter((bank): bank is string => Boolean(bank)),
        ),
      ).sort((a, b) => a.localeCompare(b));
      setBanks(names);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="shrink-0 space-y-2 bg-background pt-2">
      <label htmlFor="compliance-bank" className="block text-xs text-muted">
        Bank
      </label>
      {bankLoadFailed ? (
        <p role="alert" className="text-xs text-danger">
          Could not load the bank list. Reload the page to try again.
        </p>
      ) : (
        <select
          id="compliance-bank"
          value={company}
          onChange={(event) => setCompany(event.target.value)}
          disabled={disabled || banks === null}
          className="w-full rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-60"
        >
          <option value="">{banks === null ? "Loading banks…" : "Select a bank…"}</option>
          {(banks ?? []).map((bank) => (
            <option key={bank} value={bank}>
              {bank}
            </option>
          ))}
        </select>
      )}

      {company ? (
        <CategoryFrameworkPicker
          framework={framework}
          onFrameworkChange={setFramework}
          onPick={(category) => onSubmit(company, category, framework)}
          disabled={disabled}
        />
      ) : (
        <p className="text-xs text-muted">Pick a bank, then which checklist to run.</p>
      )}

      {isStreaming ? (
        <button
          type="button"
          onClick={onStop}
          className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          Stop
        </button>
      ) : null}
    </div>
  );
}
