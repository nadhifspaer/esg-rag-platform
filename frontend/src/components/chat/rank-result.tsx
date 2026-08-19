import type { RankResult } from "@/lib/types";

/** Reader-facing label for `ExcludedBank.reason`; falls back to the raw value if unmapped. */
const EXCLUSION_LABEL: Record<string, string> = {
  not_disclosed_in_report: "Not disclosed",
  not_disclosed_retrieval_gap: "No evidence retrieved (retrieval gap, not a stated absence)",
  unparseable: "Value stated but could not be parsed as a number",
};

/** Renders a `/rank` result: a real cross-bank ranking, or one of its three honest refusals. */
export function RankResultView({ result }: { result: RankResult }) {
  if (result.outcome === "refused") {
    return (
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Corpus-wide ranking · declined
        </p>
        <p className="text-sm leading-relaxed text-foreground">{result.message}</p>
        {result.route_to && result.route_to !== "none" ? (
          <p className="text-xs text-muted">Routed to: the {result.route_to} pipeline.</p>
        ) : null}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">
        Corpus-wide ranking · {result.target}
        {result.bank_count ? ` · ${result.bank_count} banks` : ""}
      </p>

      {result.ranked.length > 0 ? (
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="bg-surface">
                <th scope="col" className="border-b border-border px-2.5 py-1.5 text-left font-medium text-foreground">
                  Bank
                </th>
                <th scope="col" className="border-b border-border px-2.5 py-1.5 text-left font-medium text-foreground">
                  Value
                </th>
                <th scope="col" className="border-b border-border px-2.5 py-1.5 text-left font-medium text-foreground">
                  Citation
                </th>
              </tr>
            </thead>
            <tbody>
              {result.ranked.map((row) => (
                <tr key={row.bank} className="border-b border-border last:border-b-0">
                  <td className="px-2.5 py-1.5 align-top text-foreground/90">{row.bank}</td>
                  <td className="whitespace-nowrap px-2.5 py-1.5 align-top font-mono tabular-nums text-foreground">
                    {row.raw_value}
                    {row.scope_label ? (
                      <span className="ml-1.5 font-sans text-muted">({row.scope_label})</span>
                    ) : null}
                  </td>
                  <td className="px-2.5 py-1.5 align-top text-muted">{row.citation ?? "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {result.note ? <p className="text-xs text-muted">{result.note}</p> : null}

      {result.ambiguous.length > 0 ? (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted hover:text-foreground">
            {result.ambiguous.length} bank{result.ambiguous.length === 1 ? "" : "s"} held out:
            multiple scoped values disclosed
          </summary>
          <ul className="mt-1.5 space-y-1 border-l border-border pl-3">
            {result.ambiguous.map((bank) => (
              <li key={bank.bank} className="text-muted">
                <span className="text-foreground/90">{bank.bank}</span>: {bank.values.join("; ")}
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {result.excluded.length > 0 ? (
        <details className="text-xs">
          <summary className="cursor-pointer text-muted hover:text-foreground">
            {result.excluded.length} bank{result.excluded.length === 1 ? "" : "s"} excluded: no
            usable disclosed value
          </summary>
          <ul className="mt-1.5 space-y-1 border-l border-border pl-3">
            {result.excluded.map((bank) => (
              <li key={bank.bank} className="text-muted">
                <span className="text-foreground/90">{bank.bank}</span>:{" "}
                {EXCLUSION_LABEL[bank.reason] ?? bank.reason}
                {bank.detail ? ` (${bank.detail})` : ""}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}
