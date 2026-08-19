import { useState } from "react";

import type { ChatMessage, Domain } from "@/lib/types";

import { AnswerBody, citationId } from "./answer-body";
import { CategoryFrameworkPicker } from "./category-picker";
import { ComplianceTable } from "./compliance-table";
import { CopyButton } from "./copy-button";
import { Elapsed } from "./elapsed";
import { RankResultView } from "./rank-result";
import { RoutingDetail } from "./routing-detail";

const CONTENT_TYPE_LABEL: Record<string, string> = {
  text: "",
  table: "table",
  chart_caption: "figure",
};

export function Message({
  message,
  isStreaming,
  onRetry,
  onPinDomain,
  onPickCategory,
}: {
  message: ChatMessage;
  isStreaming: boolean;
  onRetry?: (message: ChatMessage) => void;
  onPinDomain?: (message: ChatMessage, domain: Domain) => void;
  onPickCategory?: (
    message: ChatMessage,
    category: "environmental" | "social" | "governance",
    framework: "gri" | "edgb",
  ) => void;
}) {
  // Which framework's checklist the picker below will run. Declared before the early
  // return because hooks can't follow one.
  const [selectedFramework, setSelectedFramework] = useState<"gri" | "edgb">("gri");

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        {/* Speaker attribution for screen readers; visually the alignment and colour carry it. */}
        <h2 className="sr-only">You asked</h2>
        <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-accent px-4 py-2.5 text-sm text-accent-foreground">
          {message.content}
        </p>
      </div>
    );
  }

  // Compliance/rank turns and a pending `needsCategory` must all hold off `waiting` too.
  const waiting =
    isStreaming &&
    !message.content &&
    !message.error &&
    !message.compliance &&
    !message.rank &&
    !message.needsCategory;
  const isChatAnswer = !message.compliance && !message.rank;

  return (
    <div className="max-w-[92%] space-y-2">
      <h2 className="sr-only">Answer</h2>
      <div className="rounded-2xl rounded-bl-sm border border-border bg-surface-raised px-4 py-3">
        {/* Chat-loop routing decision, shown as an always-visible badge before the answer streams in. */}
        {isChatAnswer && message.meta ? (
          <RoutingDetail
            meta={message.meta}
            done={message.done}
            onPinDomain={
              onPinDomain && message.sourceQuestion
                ? (domain) => onPinDomain(message, domain)
                : undefined
            }
          />
        ) : null}

        {waiting ? <Waiting message={message} /> : null}

        {message.compliance ? (
          <ComplianceTable result={message.compliance} messageId={message.id} />
        ) : null}

        {message.rank ? <RankResultView result={message.rank} /> : null}

        {isChatAnswer && message.content ? (
          <>
            <AnswerBody answer={message.content} messageId={message.id} />
            {isStreaming ? (
              <span
                className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-text-bottom"
                aria-hidden="true"
              />
            ) : null}
          </>
        ) : null}

        {message.needsCategory ? (
          <div className="mt-2 border-t border-border pt-2">
            <p className="text-sm text-muted">
              This reads as a compliance question about{" "}
              <span className="text-foreground">{message.needsCategory.company}</span>, but it
              does not clearly name an E/S/G area: none of the curated checklists&rsquo; topic
              keywords matched, so guessing one would risk running (and charging) the wrong
              sweep. Which checklist should it run against?
            </p>

            <CategoryFrameworkPicker
              framework={selectedFramework}
              onFrameworkChange={setSelectedFramework}
              onPick={(category) => onPickCategory?.(message, category, selectedFramework)}
            />
          </div>
        ) : null}

        {message.error ? (
          <div className="mt-2 border-t border-border pt-2">
            <p role="alert" className="text-sm text-danger">
              {message.error}
            </p>
            {onRetry && message.sourceQuestion ? (
              <button
                type="button"
                onClick={() => onRetry(message)}
                className="mt-2 rounded-md border border-border px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                Retry this question
              </button>
            ) : null}
          </div>
        ) : null}

        {message.note ? (
          // Not an error, so not red and not announced as an alert.
          <p className="mt-2 border-t border-border pt-2 text-sm text-muted">{message.note}</p>
        ) : null}

        {isChatAnswer && message.citations && message.citations.length > 0 ? (
          <div className="mt-3 border-t border-border pt-3">
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              {/* Not a heading: "Sources" repeated identically on every message made the
                  document outline useless, and it sat at h3 under an h1. */}
              <p id={`sources-${message.id}`} className="text-xs font-medium uppercase tracking-wide text-muted">
                Sources
              </p>
              <CopyButton
                text={answerWithSources(message)}
                label="Copy answer + sources"
                copiedLabel="Copied"
              />
            </div>
            <ol className="space-y-1 text-xs" aria-labelledby={`sources-${message.id}`}>
              {message.citations.map((citation) => (
                <li
                  key={citation.index}
                  id={citationId(message.id, citation.index)}
                  // `:target` highlight so following an inline marker visibly lands somewhere,
                  // and `scroll-mt` so the anchored row is not hidden under the thread's top edge.
                  className="group flex scroll-mt-24 flex-col gap-1 rounded px-1 py-0.5 text-muted target:bg-accent/10 target:text-foreground"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="font-mono text-accent">{citation.index}</span>
                    <span className="flex-1">
                      {citation.source_name}
                      {/* Full text-muted for contrast: the page number is the citation's most specific datum. */}
                      <span className="text-muted">, p. {citation.page_number}</span>
                      {CONTENT_TYPE_LABEL[citation.content_type] ? (
                        <span className="ml-1.5 rounded bg-surface px-1.5 py-0.5 text-[0.65rem] uppercase tracking-wide">
                          {CONTENT_TYPE_LABEL[citation.content_type]}
                        </span>
                      ) : null}
                    </span>
                    <CopyButton
                      // The app's canonical one-line citation string, built in `models/citation.py`.
                      text={citation.label}
                      label="Copy"
                      className="opacity-0 transition-opacity focus-visible:opacity-100 group-hover:opacity-100"
                    />
                  </div>
                  {citation.excerpt ? (
                    <p className="pl-5 leading-relaxed text-muted/90">{citation.excerpt}</p>
                  ) : null}
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </div>

      {message.compliance ? (
        <RoutingDetail pipeline="compliance" compliance={message.compliance} />
      ) : message.rank ? (
        <RoutingDetail pipeline="rank" rank={message.rank} />
      ) : null}
    </div>
  );
}

/** The wait state: headline copy per pipeline plus a state-driven (not CSS-animated) elapsed counter. */
function Waiting({ message }: { message: ChatMessage }) {
  // `waitingFor` covers the auto-routed case, where `message.meta` never arrives at all.
  if (message.waitingFor === "rank") {
    return (
      <Label
        message={message}
        headline="Running the corpus-wide ranking pipeline"
        detail={
          // An order of magnitude more retrieval work than an ordinary /chat turn.
          "This checks every bank's disclosure individually rather than retrieving from one " +
          "report, and can take up to about a minute."
        }
      />
    );
  }
  if (message.waitingFor === "compliance") {
    return (
      <Label
        message={message}
        headline="Running the compliance checklist"
        detail="Checking each requirement in the checklist against this bank's report."
      />
    );
  }

  const routed = Boolean(message.meta);
  return (
    <Label
      message={message}
      headline={routed ? "Drafting the answer and checking it against the sources" : "Routing your question"}
      detail={
        routed
          ? "This usually takes 30–60 seconds. The model drafts an answer, then re-reads the " +
            "retrieved pages to check every claim is supported before returning it."
          : undefined
      }
    />
  );
}

/** The wait state's shared shape: a pulsing dot, a headline, the elapsed time, and an optional
 *  explanatory line, factored out once a second and third wait-copy variant needed it. */
function Label({
  message,
  headline,
  detail,
}: {
  message: ChatMessage;
  headline: string;
  detail?: string;
}) {
  return (
    <div className="space-y-1.5">
      <p className="flex items-center gap-2 text-sm text-muted">
        <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-accent" aria-hidden="true" />
        <span>{headline}</span>
        {/* Full `text-muted`, not `/90`: the token is already 4.76:1 and has no headroom to be
            dimmed; at /90 it computed 3.90:1. */}
        {message.startedAt ? (
          <span className="text-muted">
            · <Elapsed startedAt={message.startedAt} />
          </span>
        ) : null}
      </p>
      {detail ? <p className="text-xs text-muted">{detail}</p> : null}
    </div>
  );
}

/** Answer plus its numbered sources, for pasting into a document. */
function answerWithSources(message: ChatMessage): string {
  const sources = (message.citations ?? [])
    .map((citation) => `[${citation.index}] ${citation.label}`)
    .join("\n");
  return `${message.content}\n\nSources:\n${sources}`;
}
