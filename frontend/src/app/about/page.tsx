import { DocumentCategoryModal } from "@/components/document-category-modal";
import { getDocuments } from "@/lib/documents";

export const metadata = {
  title: "About",
};

/** The platform info page, reached via the sidebar link (formerly the site root before `/` began redirecting to `/chat`). */
export default async function AboutPage() {
  const documents = await getDocuments();

  const companyDocs = documents?.filter((d) => d.domain === "company_documents") ?? [];
  const standards = documents?.filter((d) => d.domain === "standards") ?? [];

  return (
    <div className="mx-auto max-w-3xl px-6 py-16">
      <p className="text-xs font-medium uppercase tracking-widest text-accent">
        Indonesian Banking
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
        ESG Intelligence Platform
      </h1>
      <p className="mt-4 text-base leading-relaxed text-muted">
        Grounded, cited answers over a fixed corpus of Indonesian banking ESG disclosure
        filings and reporting standards, with every figure traced to the
        page it came from.
      </p>

      {documents === null ? (
        <p role="alert" className="mt-12 text-sm text-danger">
          The document list could not be loaded from the backend just now. Try reloading this
          page.
        </p>
      ) : (
        <div className="mt-12 grid gap-4 sm:grid-cols-2">
          <DocumentCategoryModal
            value={String(companyDocs.length)}
            label="ESG disclosure filings"
            categoryTitle="ESG disclosure filings"
            documents={companyDocs}
          />
          <DocumentCategoryModal
            value={String(standards.length)}
            label="ESG reporting standards"
            categoryTitle="ESG reporting standards"
            documents={standards}
          />
        </div>
      )}

      <section className="mt-14 space-y-2">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-muted">
          How this works
        </h2>
        <ul className="list-disc space-y-2 pl-5 text-sm leading-relaxed text-muted">
          <li>
            Retrieval is domain-filtered hybrid search, dense embeddings fused with BM25
            keyword matching, then re-ranked by a cross-encoder, not a single similarity
            search.
          </li>
          <li>
            The chat pipeline adds a bounded agentic retry loop: the model checks its own
            answer against the retrieved sources and, if unsupported, rewrites the query and
            retrieves again, capped at two retries.
          </li>
          <li>
            Compliance questions run a separate, fully deterministic checklist-driven pipeline
            that checks every requirement in a standard directly, rather than relying on
            similarity search to surface which ones apply.
          </li>
          <li>
            Cross-bank comparisons run their own aggregation and ranking pipeline over every
            bank&rsquo;s disclosed value for a metric, holding out anything ambiguous rather
            than guessing.
          </li>
        </ul>
      </section>

      <p className="mt-14 border-t border-border pt-6 text-xs leading-relaxed text-muted">
        Ask in English or Bahasa Indonesia. Answers quote source text verbatim, so an English
        answer may include Indonesian wording where that is what the report says.
      </p>
    </div>
  );
}
