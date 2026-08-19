import type { ChatDone, ChatMeta, ComplianceResult, ContextSource, Domain, RankResult } from "@/lib/types";

/** The routing decision and loop outcome behind one answer, shown rather than hidden. */

const CONTEXT_LABELS: Record<ContextSource, string> = {
  utterance: "Asked directly",
  entity_swap: "Follow-up · same question, different bank",
  inherited_entity: "Follow-up · bank carried over",
  previous_query: "Follow-up · same search as before",
};

const DOMAIN_LABELS: Record<Domain, string> = {
  company_documents: "Company reports",
  standards: "ESG standards",
};

// Keyed on `ChatMeta["query_type"]` itself so TypeScript rejects a missing or stale case.
const PIPELINE_LABELS: Record<ChatMeta["query_type"], string> = {
  single_domain: "Single-domain lookup",
  cross_domain: "Cross-domain comparison",
};

const ALL_DOMAINS: Domain[] = ["company_documents", "standards"];

type RoutingDetailProps =
  | {
      pipeline?: "chat";
      meta: ChatMeta;
      done?: ChatDone;
      onPinDomain?: (domain: Domain) => void;
    }
  | { pipeline: "compliance"; compliance: ComplianceResult }
  | { pipeline: "rank"; rank: RankResult };

export function RoutingDetail(props: RoutingDetailProps) {
  if (props.pipeline === "compliance") return <ComplianceRoutingDetail result={props.compliance} />;
  if (props.pipeline === "rank") return <RankRoutingDetail result={props.rank} />;
  return <ChatRoutingDetail meta={props.meta} done={props.done} onPinDomain={props.onPinDomain} />;
}

/** The chat loop's routing readout: a small always-visible badge, not a footnote. */
function ChatRoutingDetail({
  meta,
  done,
  onPinDomain,
}: {
  meta: ChatMeta;
  done?: ChatDone;
  onPinDomain?: (domain: Domain) => void;
}) {
  const isFollowUp = meta.context_source !== "utterance";
  const cappedOut = done?.stop_reason === "retry_cap";
  const others = ALL_DOMAINS.filter((d) => !meta.domains.includes(d));

  return (
    <details className="group mb-2 text-xs">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 py-1 text-muted transition-colors hover:border-accent/50 hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent [&::-webkit-details-marker]:hidden">
        <span className="font-medium text-foreground/90">{PIPELINE_LABELS[meta.query_type]}</span>
        {/* Carried on the badge itself so the caveat doesn't depend on opening the disclosure. */}
        {cappedOut ? (
          <span className="rounded bg-warning/15 px-1.5 py-0.5 font-medium text-warning">
            answer not fully verified
          </span>
        ) : null}
        <span className="text-muted group-open:hidden">details</span>
        <span className="hidden text-muted group-open:inline">hide</span>
      </summary>

      <dl className="mt-2 grid gap-x-4 gap-y-1 border-l border-border pl-3 sm:grid-cols-[auto_1fr]">
        <Row label="Pipeline">{PIPELINE_LABELS[meta.query_type]}</Row>
        <Row label="Interpreted as">{CONTEXT_LABELS[meta.context_source]}</Row>
        {isFollowUp ? (
          <Row label="Searched for">
            <code className="font-mono text-[0.7rem]">{meta.retrieval_seed}</code>
          </Row>
        ) : null}
        {meta.inherited_entity ? <Row label="Carried over">{meta.inherited_entity}</Row> : null}
        {meta.domains.map((domain) => (
          <Row
            key={domain}
            label={meta.domains.length > 1 ? `Scoped to · ${DOMAIN_LABELS[domain]}` : "Scoped to"}
          >
            {meta.scoped_to[domain] ?? "No single report - searched the domain"}
          </Row>
        ))}
        <Row label="Routing">
          {meta.classifier.method === "rule_based" ? "Rule-based" : "Model fallback"}
          {meta.classifier.confident ? "" : " (uncertain)"} · {meta.classifier.reason}
        </Row>
        {done ? (
          <Row label="Self-check">
            {cappedOut ? (
              <span className="text-warning">
                {done.attempts} attempts · stopped at the retry cap, so the model was not fully
                satisfied this answer was supported
              </span>
            ) : (
              `${done.attempts} attempt${done.attempts === 1 ? "" : "s"} · the model judged this answer supported by the sources`
            )}
          </Row>
        ) : null}
        <Row label="Model">
          {meta.model}
          {meta.high_accuracy ? " · high accuracy" : ""}
        </Row>
        {meta.history_turns > 0 ? (
          <Row label="Context">
            {meta.history_turns} earlier turn{meta.history_turns === 1 ? "" : "s"} in view
          </Row>
        ) : null}
      </dl>

      {/* The readout becomes a control: manual domain pinning is the remedy for an uncertain route. */}
      {onPinDomain && others.length > 0 ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 pl-3">
          <span className="text-muted">Wrong domain?</span>
          {others.map((domain) => (
            <button
              key={domain}
              type="button"
              onClick={() => onPinDomain(domain)}
              className="rounded border border-border px-1.5 py-0.5 text-muted transition-colors hover:bg-surface hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              Search {DOMAIN_LABELS[domain]} instead
            </button>
          ))}
        </div>
      ) : null}
    </details>
  );
}

/** This turn was auto-routed to `/compliance-check` after `/chat` refused it, see `useChat`. */
function ComplianceRoutingDetail({ result }: { result: ComplianceResult }) {
  return (
    <Shell summary={<span>Compliance check · {result.category} · {shortSource(result.report_name)}</span>}>
      <dl className="mt-2 grid gap-x-4 gap-y-1 border-l border-border pl-3 sm:grid-cols-[auto_1fr]">
        <Row label="Pipeline">Compliance check (auto-routed from /chat)</Row>
        <Row label="Report">{result.report_name}</Row>
        <Row label="Checklist">{result.category}</Row>
        <Row label="Requirements checked">{result.results.length}</Row>
      </dl>
    </Shell>
  );
}

/** This turn was auto-routed to `/rank` after `/chat` refused it, see `useChat`. */
function RankRoutingDetail({ result }: { result: RankResult }) {
  return (
    <Shell summary={<span>Corpus-wide ranking · {result.target ?? "unresolved"}</span>}>
      <dl className="mt-2 grid gap-x-4 gap-y-1 border-l border-border pl-3 sm:grid-cols-[auto_1fr]">
        <Row label="Pipeline">Corpus-wide ranking (auto-routed from /chat)</Row>
        <Row label="Outcome">{result.outcome === "ranked" ? "Ranked" : "Declined"}</Row>
        {result.target ? <Row label="Target">{result.target}</Row> : null}
        {result.bank_count ? <Row label="Banks covered">{result.bank_count}</Row> : null}
        {result.outcome === "refused" && result.route_to ? (
          <Row label="Routed to">{result.route_to}</Row>
        ) : null}
      </dl>
    </Shell>
  );
}

/** The shared collapsible shell `ComplianceRoutingDetail`/`RankRoutingDetail` render inside,
 *  below the answer. `ChatRoutingDetail` no longer uses this, see its own docstring above. */
function Shell({ summary, children }: { summary: React.ReactNode; children: React.ReactNode }) {
  return (
    <details className="group mt-3 text-xs">
      <summary className="inline-flex cursor-pointer list-none items-center gap-2 rounded text-muted transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent [&::-webkit-details-marker]:hidden">
        {summary}
        <span className="text-muted group-open:hidden">details</span>
        <span className="hidden text-muted group-open:inline">hide</span>
      </summary>
      {children}
    </details>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-muted">{label}</dt>
      <dd className="text-foreground/90">{children}</dd>
    </>
  );
}

/** Source names are long (e.g. "PT Bank Central Asia Tbk" followed by the report year and title). */
function shortSource(sourceName: string): string {
  return sourceName.split("—")[0].trim();
}
