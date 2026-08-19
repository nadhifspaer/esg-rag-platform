"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ChatRequestError, runComplianceCheck, runRank, streamChat } from "./api";
import {
  appendTurn,
  getConversationId,
  loadHistory,
  clearConversation as clearStored,
  toWire,
} from "./conversation";
import type {
  ChatMessage,
  ComplianceResult,
  ConversationTurn,
  Domain,
  RankResult,
  RoutingRefusal,
} from "./types";

/** Owns one conversation: the rendered thread, the client-owned history, and the request cycle. */

/** What a turn was sent with, so Retry and a domain re-run can reproduce it exactly. */
export interface SendOptions {
  highAccuracy: boolean;
  domain?: Domain | null;
}

/** A rate-limit refusal, held with an absolute deadline rather than the raw `Retry-After` count. */
export interface RateLimitState {
  message: string;
  until: number | null;
}

/** A short plain-text account of a compliance result, replayed to the model on a later `/chat` turn. */
function complianceSynopsis(result: ComplianceResult): string {
  return `[Compliance check — ${result.category}, ${result.report_name}] ${result.summary}`;
}

function rankSynopsis(result: RankResult): string {
  if (result.outcome === "refused") {
    return `[Corpus-wide ranking declined] ${result.message ?? ""}`;
  }
  const top = result.ranked
    .slice(0, 5)
    .map((row) => `${row.bank}: ${row.raw_value}${row.scope_label ? ` (${row.scope_label})` : ""}`)
    .join("; ");
  return `[Corpus-wide ranking — ${result.target ?? "unresolved"}] ${top || "(no banks ranked)"}`;
}

/** Display labels for the composer's explicit Compliance-check mode's user-bubble question text. */
const CATEGORY_LABEL: Record<"environmental" | "social" | "governance", string> = {
  environmental: "Environmental",
  social: "Social",
  governance: "Governance",
};

/** Runs one auto-routed pipeline call (`/rank` or `/compliance-check`) and patches the message with the outcome. */
async function runAutoRoutedPipeline<T>(
  call: (signal: AbortSignal) => Promise<T>,
  synopsis: (result: T) => string,
  pipeline: "rank" | "compliance",
  patchResult: (result: T) => Partial<ChatMessage>,
  ctx: {
    patch: (fields: Partial<ChatMessage>) => void;
    setHistory: (updater: (prev: ConversationTurn[]) => ConversationTurn[]) => void;
    setRateLimit: (state: RateLimitState | null) => void;
    setStatus: (status: string) => void;
    question: string;
    controller: AbortController;
  },
): Promise<boolean> {
  try {
    const result = await call(ctx.controller.signal);
    ctx.patch({ ...patchResult(result), waitingFor: undefined });
    ctx.setHistory((prev) =>
      appendTurn(prev, {
        question: ctx.question,
        answer: synopsis(result),
        retrievalQuery: null,
        pipeline,
      }),
    );
    ctx.setStatus("Answer ready.");
    return true;
  } catch (error) {
    if (ctx.controller.signal.aborted) {
      ctx.patch({ waitingFor: undefined, note: "Stopped." });
      ctx.setStatus("Stopped.");
      return false;
    }
    if (error instanceof ChatRequestError && error.status === 429) {
      // Same account as `/chat`'s own 429: the turn never really happened, so the banner
      // owns it rather than also marking the message an error.
      ctx.patch({ waitingFor: undefined });
      ctx.setRateLimit({
        message: error.message,
        until:
          error.retryAfterSeconds !== undefined
            ? Date.now() + error.retryAfterSeconds * 1000
            : null,
      });
      ctx.setStatus("Query budget reached.");
      return false;
    }
    const message =
      error instanceof ChatRequestError
        ? error.message
        : "Could not reach the API. Check your connection and try again.";
    ctx.patch({ waitingFor: undefined, error: message });
    ctx.setStatus("The question could not be answered.");
    return false;
  }
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<ConversationTurn[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [rateLimit, setRateLimit] = useState<RateLimitState | null>(null);
  /** A short, settled sentence describing the last transition, for the status live region. */
  const [status, setStatus] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const stored = loadHistory();
    setHistory(stored);
    // Rebuild a readable thread from stored turns; routing metadata and citations aren't stored.
    setMessages(
      stored.flatMap((turn, i) => [
        { id: `restored-u-${i}`, role: "user" as const, content: turn.question },
        { id: `restored-a-${i}`, role: "assistant" as const, content: turn.answer },
      ]),
    );
  }, []);

  /** Send one turn; resolves `true` when an answer was produced, so the composer can restore a failed question. */
  const send = useCallback(
    async (query: string, options: SendOptions): Promise<boolean> => {
      const trimmed = query.trim();
      if (!trimmed || isStreaming) return false;

      setRateLimit(null);
      // Announced immediately, so a screen reader isn't left in silence while routing happens.
      setStatus("Working. Routing your question.");
      const userId = `u-${Date.now()}`;
      const assistantId = `a-${Date.now()}`;

      setMessages((prev) => [
        ...prev,
        { id: userId, role: "user", content: trimmed },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          // Carried on the message so a Retry or a domain re-run can reproduce this turn.
          sourceQuestion: trimmed,
          startedAt: Date.now(),
        },
      ]);
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      let answer = "";
      let retrievalQuery: string | null = null;

      /** One "still working" announcement partway through the wait; cleared in `finally`. */
      const stillWorking = window.setTimeout(
        () => setStatus("Still working on your answer."),
        30_000,
      );

      const patch = (fields: Partial<ChatMessage>) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, ...fields } : m)),
        );

      /** Re-routes a `/chat` 422 refusal to the pipeline that actually answers it. */
      async function handleRoutingRefusal(
        routing: RoutingRefusal,
        question: string,
        patchFn: (fields: Partial<ChatMessage>) => void,
        signalController: AbortController,
      ): Promise<boolean> {
        if (routing.query_type === "aggregation_ranking") {
          patchFn({ waitingFor: "rank" });
          setStatus("Routed to the corpus-wide ranking pipeline.");
          return runAutoRoutedPipeline(
            (signal) => runRank(question, signal),
            rankSynopsis,
            "rank",
            (result) => ({ pipeline: "rank", rank: result }),
            {
              patch: patchFn,
              setHistory,
              setRateLimit,
              setStatus,
              question,
              controller: signalController,
            },
          );
        }

        // compliance_check
        if (!routing.company) {
          patchFn({
            waitingFor: undefined,
            error:
              "This reads as a compliance-check question, but no single bank was named (or " +
              "more than one was). Name one bank to run its checklist.",
          });
          setStatus("The question could not be answered.");
          return false;
        }
        if (!routing.category) {
          // Ambiguity, not a failure: hold on the message, since the next step is a button
          // click in the thread, not retyping the question.
          patchFn({ waitingFor: undefined, needsCategory: { company: routing.company } });
          setStatus("Which ESG checklist should this run against?");
          return true;
        }
        patchFn({ waitingFor: "compliance" });
        setStatus("Routed to the compliance-check pipeline.");
        const company = routing.company;
        const category = routing.category;
        // No framework selector exists on this path, so GRI stays the default.
        return runAutoRoutedPipeline(
          (signal) => runComplianceCheck(company, category, "gri", signal),
          complianceSynopsis,
          "compliance",
          (result) => ({ pipeline: "compliance", compliance: result }),
          {
            patch: patchFn,
            setHistory,
            setRateLimit,
            setStatus,
            question,
            controller: signalController,
          },
        );
      }

      try {
        await streamChat(
          {
            query: trimmed,
            history: toWire(history),
            conversationId: getConversationId(),
            highAccuracy: options.highAccuracy,
            domain: options.domain ?? null,
          },
          {
            onMeta: (meta) => {
              patch({ meta });
              // `meta` arrives once retrieval is genuinely finished; say so.
              setStatus(
                "Searching finished. Drafting the answer and checking it against the sources. " +
                  "This usually takes 30 to 60 seconds.",
              );
            },
            onCitations: (citations) => patch({ citations }),
            onToken: (text) => {
              answer += text;
              patch({ content: answer });
            },
            onDone: (done) => {
              retrievalQuery = done.retrieval_query;
              patch({ done });
            },
            onError: (message) => patch({ error: message }),
          },
          controller.signal,
        );

        // Only a completed turn joins the history; an errored or aborted turn is left out.
        if (answer) {
          setHistory((prev) => appendTurn(prev, { question: trimmed, answer, retrievalQuery }));
          setStatus("Answer ready.");
          return true;
        }
        setStatus("No answer was returned.");
        return false;
      } catch (error) {
        if (controller.signal.aborted) {
          // Stopping is something the user chose, so it is a note rather than an error.
          patch({ content: answer, note: "Stopped." });
          setStatus("Stopped.");
          return answer.length > 0;
        }

        // `/chat` refused this as aggregation_ranking or compliance_check; re-send the same
        // question to the pipeline that actually answers it, rather than showing this as an error.
        if (error instanceof ChatRequestError && error.routing) {
          return await handleRoutingRefusal(error.routing, trimmed, patch, controller);
        }

        if (error instanceof ChatRequestError && error.status === 429) {
          // The banner owns a 429 completely: the throttled turn never reached retrieval and
          // cost nothing, so the pair is dropped and the question goes back to the composer.
          setMessages((prev) => prev.filter((m) => m.id !== userId && m.id !== assistantId));
          setRateLimit({
            message: error.message,
            until:
              error.retryAfterSeconds !== undefined
                ? Date.now() + error.retryAfterSeconds * 1000
                : null,
          });
          setStatus("Query budget reached.");
          return false;
        }

        const message =
          error instanceof ChatRequestError
            ? error.message
            : "Could not reach the API. Check your connection and try again.";
        patch({ error: message });
        setStatus("The question could not be answered.");
        return false;
      } finally {
        window.clearTimeout(stillWorking);
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [history, isStreaming],
  );

  /** Explicit Compliance-check composer mode: calls `/compliance-check` directly, no `/chat` round trip. */
  const sendCompliance = useCallback(
    async (
      company: string,
      category: "environmental" | "social" | "governance",
      framework: "gri" | "edgb",
    ): Promise<boolean> => {
      if (!company || isStreaming) return false;

      setRateLimit(null);
      const question = `Compliance check — ${company} · ${CATEGORY_LABEL[category]} (${framework.toUpperCase()})`;
      const assistantId = `a-${Date.now()}`;

      setMessages((prev) => [
        ...prev,
        { id: `u-${Date.now()}`, role: "user", content: question },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          sourceQuestion: question,
          startedAt: Date.now(),
          waitingFor: "compliance",
        },
      ]);
      setStatus("Running the compliance checklist.");
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;
      const patch = (fields: Partial<ChatMessage>) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...fields } : m)));

      try {
        return await runAutoRoutedPipeline(
          (signal) => runComplianceCheck(company, category, framework, signal),
          complianceSynopsis,
          "compliance",
          (result) => ({ pipeline: "compliance", compliance: result }),
          { patch, setHistory, setRateLimit, setStatus, question, controller },
        );
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming],
  );

  /** Explicit Rank-banks composer mode: calls `/rank` directly with the typed query, no `/chat` round trip. */
  const sendRank = useCallback(
    async (query: string): Promise<boolean> => {
      const trimmed = query.trim();
      if (!trimmed || isStreaming) return false;

      setRateLimit(null);
      const assistantId = `a-${Date.now()}`;

      setMessages((prev) => [
        ...prev,
        { id: `u-${Date.now()}`, role: "user", content: trimmed },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          sourceQuestion: trimmed,
          startedAt: Date.now(),
          waitingFor: "rank",
        },
      ]);
      setStatus("Running the corpus-wide ranking pipeline.");
      setIsStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;
      const patch = (fields: Partial<ChatMessage>) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...fields } : m)));

      try {
        return await runAutoRoutedPipeline(
          (signal) => runRank(trimmed, signal),
          rankSynopsis,
          "rank",
          (result) => ({ pipeline: "rank", rank: result }),
          { patch, setHistory, setRateLimit, setStatus, question: trimmed, controller },
        );
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming],
  );

  /** Resumes a message paused on `needsCategory` once the user picks one of the three E/S/G buttons. */
  const pickCategory = useCallback(
    async (
      message: ChatMessage,
      category: "environmental" | "social" | "governance",
      framework: "gri" | "edgb",
    ) => {
      if (!message.needsCategory || isStreaming) return;
      const company = message.needsCategory.company;
      const assistantId = message.id;

      const patch = (fields: Partial<ChatMessage>) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...fields } : m)));

      patch({ needsCategory: undefined, waitingFor: "compliance", startedAt: Date.now() });
      setIsStreaming(true);
      setStatus("Routed to the compliance-check pipeline.");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        await runAutoRoutedPipeline(
          (signal) => runComplianceCheck(company, category, framework, signal),
          complianceSynopsis,
          "compliance",
          (result) => ({ pipeline: "compliance", compliance: result }),
          {
            patch,
            setHistory,
            setRateLimit,
            setStatus,
            question: message.sourceQuestion ?? "",
            controller,
          },
        );
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [isStreaming],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    clearStored();
    setMessages([]);
    setHistory([]);
    setRateLimit(null);
    setStatus("Conversation cleared.");
  }, []);

  return {
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
  };
}
