"use client";

import { useRef } from "react";

import type { DocumentInfo } from "@/lib/types";

/** A stat card that opens a modal listing every real document in that category. */
export function DocumentCategoryModal({
  value,
  label,
  categoryTitle,
  documents,
}: {
  value: string;
  label: string;
  categoryTitle: string;
  documents: DocumentInfo[];
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  return (
    <>
      <button
        type="button"
        onClick={() => dialogRef.current?.showModal()}
        className="rounded-lg border border-border bg-surface px-4 py-3 text-left transition-colors hover:bg-surface-raised focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <span className="block text-2xl font-semibold tracking-tight text-foreground">
          {value}
        </span>
        <span className="text-xs text-muted">{label}</span>
      </button>

      <dialog
        ref={dialogRef}
        aria-labelledby={`${label}-modal-title`}
        className="w-full max-w-md rounded-lg border border-border bg-surface p-0 text-foreground backdrop:bg-foreground/20"
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 id={`${label}-modal-title`} className="text-sm font-semibold tracking-tight">
            {categoryTitle}
          </h2>
          <button
            type="button"
            onClick={() => dialogRef.current?.close()}
            aria-label="Close"
            className="rounded-md px-2 py-1 text-muted transition-colors hover:bg-surface-raised hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            ✕
          </button>
        </div>
        <ul className="max-h-[60vh] space-y-1.5 overflow-y-auto px-5 py-4 text-sm">
          {documents.map((doc) => (
            <li key={doc.source_name} className="text-foreground">
              {doc.source_name}
            </li>
          ))}
        </ul>
      </dialog>
    </>
  );
}
