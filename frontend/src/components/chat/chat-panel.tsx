"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { MAX_HISTORY_TURNS } from "@/lib/conversation";
import type { ChatMessage, Domain } from "@/lib/types";
import { useChat } from "@/lib/use-chat";

import { ComplianceComposer } from "./compliance-composer";
import { Message } from "./message";
import { RankComposer } from "./rank-composer";

/** Examples carry their language explicitly (not inferred), since `lang` is an accessibility attribute. */
const EXAMPLES: { text: string; lang: "en" | "id" }[] = [
  { text: "What does BCA report for Scope 1 emissions?", lang: "en" },
  { text: "Berapa emisi Scope 1 yang dilaporkan BCA?", lang: "id" },
  { text: "What does GRI 2 require companies to disclose about board oversight?", lang: "en" },
];

/** Manual domain pin: a UX feature and the fallback when auto-routing picks the wrong domain. */
const DOMAIN_OPTIONS: { value: Domain | null; label: string }[] = [
  { value: null, label: "Auto" },
  { value: "company_documents", label: "Company reports" },
  { value: "standards", label: "Standards" },
];

/** The composer's mode toggle: Ask (default, via the classifier) or Compliance check / Rank banks (direct to their endpoints). */
const MODE_OPTIONS: { value: "ask" | "compliance" | "rank"; label: string }[] = [
  { value: "ask", label: "Ask" },
  { value: "compliance", label: "Compliance check" },
  { value: "rank", label: "Rank banks" },
];

export function ChatPanel() {
  const {
    messages,
    history,
    isStreaming,
    rateLimit,
    status,
    send,
    sendCompliance,
    sendRank,
    pickCategory,
    stop,
    reset,
  } = useChat();
  const [input, setInput] = useState("");
  const [highAccuracy, setHighAccuracy] = useState(false);
  const [domain, setDomain] = useState<Domain | null>(null);
  // Which composer is showing below the thread, pure UI state, not part of the conversation.
  const [mode, setMode] = useState<"ask" | "compliance" | "rank">("ask");
  const threadEnd = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);

  const throttledFor = useCountdown(rateLimit?.until ?? null);
  const throttled = throttledFor > 0;

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  /** Ask a question, restoring it to the composer if it did not produce an answer. */
  const ask = useCallback(
    async (query: string, options?: { domain?: Domain | null }) => {
      const trimmed = query.trim();
      if (!trimmed || isStreaming) return;

      setInput("");
      const ok = await send(trimmed, {
        highAccuracy,
        domain: options?.domain !== undefined ? options.domain : domain,
      });
      if (!ok) {
        setInput(trimmed);
        composer.current?.focus();
      }
    },
    [domain, highAccuracy, isStreaming, send],
  );

  function submit(event: React.FormEvent) {
    event.preventDefault();
    // Trim before doing anything. The old order cleared the textarea and *then* let `send` bail on
    // an empty string, so pressing Enter on whitespace wiped the box and did nothing.
    void ask(input);
  }

  /** Re-run a question against a domain the user picked, and leave the pin showing that choice. */
  const pinDomain = useCallback(
    (message: ChatMessage, next: Domain) => {
      setDomain(next);
      void ask(message.sourceQuestion ?? "", { domain: next });
    },
    [ask],
  );

  const retry = useCallback(
    (message: ChatMessage) => void ask(message.sourceQuestion ?? ""),
    [ask],
  );

  return (
    <div className="mx-auto flex h-full max-w-3xl flex-col gap-4 px-6 py-6">
      <div className="flex shrink-0 flex-wrap items-baseline justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold tracking-tight">Ask the corpus</h1>
          <p className="text-sm text-muted">
            21 bank disclosure filings and 4 ESG standards. Every answer is
            cited to a source page.
          </p>
        </div>
        {messages.length > 0 ? (
          <button
            type="button"
            onClick={reset}
            className="rounded-md border border-border px-3 py-1.5 text-xs text-muted transition-colors hover:bg-surface-raised hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            New conversation
          </button>
        ) : null}
      </div>

      {/* A separate live region, mounted unconditionally, holding one settled sentence at a time. */}
      <p role="status" className="sr-only">
        {status}
      </p>

      {messages.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-surface p-5">
          <p className="text-sm text-muted">
            Ask in English or Bahasa Indonesia. Follow-ups work: ask about one bank, then
            &ldquo;what about Mandiri?&rdquo;
          </p>
          <ul className="mt-3 flex flex-col gap-1.5">
            {EXAMPLES.map((example) => (
              <li key={example.text}>
                <button
                  type="button"
                  onClick={() => setInput(example.text)}
                  lang={example.lang}
                  className="text-left text-sm text-accent underline-offset-4 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  {example.text}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto pb-2">
          {messages.map((message, i) => (
            <Message
              key={message.id}
              message={message}
              isStreaming={isStreaming && i === messages.length - 1}
              onRetry={retry}
              onPinDomain={pinDomain}
              onPickCategory={pickCategory}
            />
          ))}
          <div ref={threadEnd} />
        </div>
      )}

      {rateLimit ? (
        // The only place a 429 appears; shows the backend's own explanation verbatim.
        <div
          role="alert"
          className="shrink-0 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-foreground"
        >
          <p className="font-medium text-warning">Query budget reached</p>
          <p className="mt-1 text-muted">{rateLimit.message}</p>
          <p className="mt-1 text-muted">
            {throttled ? (
              <>
                You can ask again in{" "}
                <span className="tabular-nums text-foreground">{throttledFor}s</span>. Your
                question is back in the box.
              </>
            ) : (
              "You can ask again now."
            )}
          </p>
        </div>
      ) : null}

      {/* Mode toggle: chooses which pipeline the composer below talks to; stays visible in every mode. */}
      <fieldset className="flex shrink-0 flex-wrap items-center gap-1.5 text-xs text-muted">
        <legend className="sr-only">Which pipeline to use</legend>
        <span aria-hidden="true">Mode</span>
        {MODE_OPTIONS.map((option) => {
          const selected = option.value === mode;
          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={selected}
              onClick={() => setMode(option.value)}
              className={`rounded-md border px-2 py-1 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background ${
                selected
                  ? "border-accent bg-accent/10 text-foreground"
                  : "border-border text-muted hover:bg-surface-raised hover:text-foreground"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </fieldset>

      {mode === "ask" ? (
        <form onSubmit={submit} className="shrink-0 space-y-2 bg-background pt-2">
          <div className="flex items-end gap-2">
            <label htmlFor="query" className="sr-only">
              Your question
            </label>
            <textarea
              id="query"
              ref={composer}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                // Enter sends, Shift+Enter breaks the line: the convention users expect here.
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void ask(input);
                }
              }}
              rows={2}
              placeholder="Ask about a bank's disclosures or a GRI standard…"
              className="min-h-[3rem] flex-1 resize-y rounded-lg border border-border bg-surface-raised px-3 py-2 text-sm text-foreground placeholder:text-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            />
            {isStreaming ? (
              <button
                type="button"
                onClick={stop}
                className="rounded-lg border border-border px-4 py-2.5 text-sm text-muted transition-colors hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                // Disabled while throttled: leaving it live invited a second press that could only
                // fail again, which makes a working cost guardrail read as a broken app.
                disabled={!input.trim() || throttled}
                className="rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-accent-foreground transition-opacity hover:opacity-90 disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              >
                Ask
              </button>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted">
            <fieldset className="flex flex-wrap items-center gap-1.5">
              <legend className="sr-only">Which documents to search</legend>
              <span aria-hidden="true">Search</span>
              {DOMAIN_OPTIONS.map((option) => {
                const selected = option.value === domain;
                return (
                  <button
                    key={option.label}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setDomain(option.value)}
                    className={`rounded-md border px-2 py-1 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background ${
                      selected
                        ? "border-accent bg-accent/10 text-foreground"
                        : "border-border text-muted hover:bg-surface-raised hover:text-foreground"
                    }`}
                  >
                    {option.label}
                  </button>
                );
              })}
            </fieldset>

            <label className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={highAccuracy}
                onChange={(event) => setHighAccuracy(event.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              <span>
                {/* Full `text-muted`: this is the cost-toggle explanation, meant to be shown, not buried. */}
                High accuracy <span className="text-muted">(gpt-4.1, slower, costs more)</span>
              </span>
            </label>

            <span className="ml-auto">
              {history.length}/{MAX_HISTORY_TURNS} turns remembered · this tab only
            </span>
          </div>
        </form>
      ) : mode === "compliance" ? (
        <ComplianceComposer
          onSubmit={(company, category, framework) =>
            void sendCompliance(company, category, framework)
          }
          onStop={stop}
          isStreaming={isStreaming}
          disabled={isStreaming || throttled}
        />
      ) : (
        <RankComposer
          onSubmit={(query) => void sendRank(query)}
          onStop={stop}
          isStreaming={isStreaming}
          disabled={isStreaming || throttled}
        />
      )}
    </div>
  );
}

/** Seconds remaining until `until`, ticking to zero. */
function useCountdown(until: number | null): number {
  const [remaining, setRemaining] = useState(() => secondsUntil(until));

  useEffect(() => {
    setRemaining(secondsUntil(until));
    if (until === null) return;
    const id = window.setInterval(() => {
      const next = secondsUntil(until);
      setRemaining(next);
      if (next <= 0) window.clearInterval(id);
    }, 1000);
    return () => window.clearInterval(id);
  }, [until]);

  return remaining;
}

function secondsUntil(until: number | null): number {
  if (until === null) return 0;
  return Math.max(0, Math.ceil((until - Date.now()) / 1000));
}
