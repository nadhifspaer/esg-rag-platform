/** Types mirroring the backend's `/chat` contract; hand-kept in sync with the backend models. */

/** Which backend pipeline produced a turn: compliance answers render in the chat thread too. */
export type Pipeline = "chat" | "compliance" | "rank";

/** One completed exchange, as the backend expects it back on the next request. */
export interface ConversationTurn {
  question: string;
  answer: string;
  retrievalQuery: string | null;
  pipeline: Pipeline;
}

/** Wire shape (snake_case) the backend actually accepts. */
export interface ConversationTurnWire {
  question: string;
  answer: string;
  retrieval_query: string | null;
  pipeline: Pipeline;
}

export type Domain = "company_documents" | "standards";

/** How the backend resolved an elliptical follow-up, surfaced so a user can see what was searched. */
export type ContextSource =
  | "utterance"
  | "entity_swap"
  | "inherited_entity"
  | "previous_query";

/** The `meta` SSE event: the routing decision, sent before the answer. */
export interface ChatMeta {
  query_type: "single_domain" | "cross_domain";
  domains: Domain[];
  /** Per domain in `domains`: the single report/standard retrieval was confined to, or null for a domain-wide search. */
  scoped_to: Partial<Record<Domain, string | null>>;
  model: string;
  high_accuracy: boolean;
  classifier: { method: string; confident: boolean; reason: string };
  conversation_id: string | null;
  history_turns: number;
  context_source: ContextSource;
  resolved_question: string;
  retrieval_seed: string;
  inherited_entity: string | null;
}

/** One citation; v1 is text-only by product decision, no inline page images. */
export interface Citation {
  index: number;
  label: string;
  source_name: string;
  page_number: number;
  content_type: "text" | "table" | "chart_caption";
  excerpt: string;
}

/** The `done` SSE event: the agentic loop's telemetry. */
export interface ChatDone {
  attempts: number;
  retries: number;
  stop_reason: "sufficient" | "retry_cap";
  retrieval_query: string;
}

/** A rendered message in the thread. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: ChatMeta;
  citations?: Citation[];
  done?: ChatDone;
  /** Which pipeline actually answered; defaults to `"chat"` in practice. */
  pipeline?: Pipeline;
  /** Set when `pipeline === "compliance"`: the structured result to render as a table. */
  compliance?: ComplianceResult;
  /** Set when `pipeline === "rank"`: the ranked/ambiguous/excluded breakdown, or a refusal. */
  rank?: RankResult;
  /** A resolved company but no checklist category: genuine ambiguity, rendered as quick-picks. */
  needsCategory?: { company: string };
  /** Which endpoint this message is currently waiting on, so `Waiting` can show the right copy. */
  waitingFor?: "rank" | "compliance";
  /** A genuine failure. Rendered as an error, announced, and offers a retry. */
  error?: string;
  /** A neutral outcome that is not a failure, currently only "you stopped this". */
  note?: string;
  /** The question that produced this turn, carried so Retry can re-ask it without re-reading the thread. */
  sourceQuestion?: string;
  /** When the request was sent, for the elapsed-time readout during the wait. */
  startedAt?: number;
}

/** One ingested document, from `GET /documents`; mirrors `DocumentInfo` in the backend. */
export interface DocumentInfo {
  source_name: string;
  domain: Domain;
  document_type: string;
  bank: string | null;
  year: number | null;
  chunk_count: number;
  page_count: number;
}

export interface DocumentsResponse {
  count: number;
  documents: DocumentInfo[];
}

/** The structured payload `/chat` returns in a 422's `detail` when it refuses to a routed pipeline. */
export interface RoutingRefusal {
  query_type: "aggregation_ranking" | "compliance_check";
  company?: string | null;
  category?: "environmental" | "social" | "governance" | null;
}

/** Mirrors `RequirementRow` in `backend/app/api/compliance.py`. */
export interface ComplianceRequirementRow {
  code: string;
  description: string;
  status: string;
  evidence_snippet: string | null;
  citation: string | null;
  top_score: number;
  fallback_used: boolean;
  rejected_as_insubstantial: boolean;
}

/** Mirrors `ComplianceResponse` in `backend/app/api/compliance.py`. */
export interface ComplianceResult {
  report_name: string;
  category: string;
  results: ComplianceRequirementRow[];
  summary: string;
  status_note: string;
  disclaimer: string;
  markdown: string;
}

/** Mirrors `RankedBank` in `backend/app/api/rank.py`. */
export interface RankedBank {
  bank: string;
  value: number;
  raw_value: string;
  scope_label: string | null;
  citation: string | null;
}

/** Mirrors `AmbiguousBank`: a bank disclosing several scoped values, held out of ranking. */
export interface RankAmbiguousBank {
  bank: string;
  values: string[];
  citation: string | null;
}

/** Mirrors `ExcludedBank`: a bank with no disclosed value, excluded with a stated reason. */
export interface RankExcludedBank {
  bank: string;
  reason: string;
  detail: string | null;
}

/** Mirrors `RankResponse` in `backend/app/api/rank.py`; a refusal outcome is a designed result, not an error. */
export interface RankResult {
  query: string;
  query_type: string;
  target: string | null;
  target_kind: "metric" | "requirement" | "unresolved";
  outcome: "ranked" | "refused";
  route_to?: string | null;
  reason?: string | null;
  message?: string | null;
  ranked: RankedBank[];
  ambiguous: RankAmbiguousBank[];
  excluded: RankExcludedBank[];
  bank_count: number | null;
  note?: string | null;
}
