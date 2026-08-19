import Link from "next/link";

import { DOMAIN_LABELS, getDocuments } from "@/lib/documents";
import type { Domain } from "@/lib/types";

export const metadata = {
  title: "Documents",
};

const FILTERS: { value: Domain | null; label: string }[] = [
  { value: null, label: "All" },
  { value: "company_documents", label: DOMAIN_LABELS.company_documents },
  { value: "standards", label: DOMAIN_LABELS.standards },
];

function isDomain(value: string | undefined): value is Domain {
  return value === "company_documents" || value === "standards";
}

/** Read-only browse view of the ingested knowledge base, filterable by domain via `?domain=`. */
export default async function DocumentsPage({
  searchParams,
}: {
  searchParams: { domain?: string };
}) {
  const domain = isDomain(searchParams.domain) ? searchParams.domain : undefined;
  const documents = await getDocuments(domain);

  return (
    <div className="mx-auto max-w-5xl px-6 py-16">
      <p className="text-xs font-medium uppercase tracking-widest text-accent">Knowledge base</p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Documents</h1>
      <p className="mt-4 text-base leading-relaxed text-muted">
        Every document ingested into the corpus retrieval draws on, read directly from Qdrant
        so this list can never drift from what citations elsewhere actually show.
      </p>

      <nav aria-label="Filter by domain" className="mt-8 flex flex-wrap gap-2">
        {FILTERS.map((filter) => {
          const isActive = filter.value === (domain ?? null);
          const href = filter.value ? `/documents?domain=${filter.value}` : "/documents";
          return (
            <Link
              key={filter.label}
              href={href}
              aria-current={isActive ? "page" : undefined}
              className={`rounded-full border px-3 py-1.5 text-sm transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                isActive
                  ? "border-accent bg-accent text-accent-foreground"
                  : "border-border bg-surface text-muted hover:bg-surface-raised hover:text-foreground"
              }`}
            >
              {filter.label}
            </Link>
          );
        })}
      </nav>

      {documents === null ? (
        <p role="alert" className="mt-12 text-sm text-danger">
          The document list could not be loaded from the backend just now. Try reloading this
          page.
        </p>
      ) : documents.length === 0 ? (
        <p className="mt-12 text-sm text-muted">No documents in this category.</p>
      ) : (
        <>
          <div className="mt-8 overflow-x-auto rounded-lg border border-border">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-border bg-surface text-xs uppercase tracking-wide text-muted">
                  <th scope="col" className="px-4 py-3 font-medium">
                    Source
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Domain
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Bank
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Year
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Chunks
                  </th>
                  <th scope="col" className="px-4 py-3 font-medium">
                    Pages
                  </th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.source_name} className="border-b border-border last:border-0">
                    <td className="px-4 py-3 text-foreground">{doc.source_name}</td>
                    <td className="px-4 py-3 text-muted">{DOMAIN_LABELS[doc.domain]}</td>
                    <td className="px-4 py-3 text-muted">{doc.bank ?? "—"}</td>
                    <td className="px-4 py-3 text-muted">{doc.year ?? "—"}</td>
                    <td className="px-4 py-3 text-muted">{doc.chunk_count}</td>
                    <td className="px-4 py-3 text-muted">{doc.page_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs text-muted">
            {documents.length} document{documents.length === 1 ? "" : "s"}
          </p>
        </>
      )}
    </div>
  );
}
