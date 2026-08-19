/** Client-owned conversation history, held in `sessionStorage` (the backend stores none). */

import type { ConversationTurn, ConversationTurnWire, Pipeline } from "./types";

const HISTORY_KEY = "esg.conversation.history";
const CONVERSATION_ID_KEY = "esg.conversation.id";

/** Keep at most this many turns, the same cap the backend enforces, trimmed client-side to avoid a 422. */
export const MAX_HISTORY_TURNS = 8;

/** Matches the backend's per-field cap so an over-long turn cannot be sent at all. */
const MAX_TURN_CHARS = 4000;

function hasSessionStorage(): boolean {
  // Guards two real cases: Server Component rendering (no `window`), and browsers where
  // storage access throws outright (Safari private mode, hardened privacy settings).
  try {
    return typeof window !== "undefined" && !!window.sessionStorage;
  } catch {
    return false;
  }
}

/** The conversation id for this tab, created on first use: opaque, client-generated, groups traces only. */
export function getConversationId(): string {
  if (!hasSessionStorage()) return "";
  const existing = window.sessionStorage.getItem(CONVERSATION_ID_KEY);
  if (existing) return existing;

  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `conv-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  window.sessionStorage.setItem(CONVERSATION_ID_KEY, id);
  return id;
}

export function loadHistory(): ConversationTurn[] {
  if (!hasSessionStorage()) return [];
  const raw = window.sessionStorage.getItem(HISTORY_KEY);
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Storage is user-writable, so validate rather than trust the shape. A corrupted entry
    // would otherwise reach the request body and come back as a confusing 422.
    return parsed.filter(isTurn).slice(-MAX_HISTORY_TURNS);
  } catch {
    return [];
  }
}

function isTurn(value: unknown): value is ConversationTurn {
  if (typeof value !== "object" || value === null) return false;
  const turn = value as Record<string, unknown>;
  return typeof turn.question === "string" && typeof turn.answer === "string";
}

export function saveHistory(turns: ConversationTurn[]): void {
  if (!hasSessionStorage()) return;
  try {
    window.sessionStorage.setItem(HISTORY_KEY, JSON.stringify(turns.slice(-MAX_HISTORY_TURNS)));
  } catch {
    // Quota exceeded, or storage disabled mid-session. Losing history degrades follow-ups
    // to first-turn behaviour, which still answers, so this must not break the app.
  }
}

/** Append a completed exchange, trimming to the cap. */
export function appendTurn(
  turns: ConversationTurn[],
  turn: {
    question: string;
    answer: string;
    retrievalQuery: string | null;
    pipeline?: Pipeline;
  },
): ConversationTurn[] {
  const next: ConversationTurn[] = [
    ...turns,
    {
      question: turn.question.slice(0, MAX_TURN_CHARS),
      answer: turn.answer.slice(0, MAX_TURN_CHARS),
      retrievalQuery: turn.retrievalQuery,
      pipeline: turn.pipeline ?? "chat",
    },
  ].slice(-MAX_HISTORY_TURNS);
  saveHistory(next);
  return next;
}

export function clearConversation(): void {
  if (!hasSessionStorage()) return;
  window.sessionStorage.removeItem(HISTORY_KEY);
  window.sessionStorage.removeItem(CONVERSATION_ID_KEY);
}

/** Convert to the snake_case shape the backend expects. */
export function toWire(turns: ConversationTurn[]): ConversationTurnWire[] {
  return turns.map((turn) => ({
    question: turn.question,
    answer: turn.answer,
    retrieval_query: turn.retrievalQuery,
    pipeline: turn.pipeline,
  }));
}
