/** Server-side fetch for `GET /documents`, shared by every page that reads the document catalog. */

import { API_BASE } from "./api";
import type { Domain, DocumentInfo, DocumentsResponse } from "./types";

export async function getDocuments(domain?: Domain): Promise<DocumentInfo[] | null> {
  try {
    const url = domain ? `${API_BASE}/documents?domain=${domain}` : `${API_BASE}/documents`;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return null;
    const body: DocumentsResponse = await response.json();
    return body.documents;
  } catch {
    return null;
  }
}

/** Human labels for a domain, shared so `/about`'s stat cards and `/documents`'s filter tabs
 *  and badges never drift into slightly different wording for the same three domains. */
export const DOMAIN_LABELS: Record<Domain, string> = {
  company_documents: "ESG disclosure filings",
  standards: "ESG reporting standards",
};
