/** Backend API client: streams `/chat` via `fetch` and manual Server-Sent Events parsing (not `EventSource`, which can't send a POST body or headers). */

import { getAccessToken } from "./supabase/client";
import type {
  ChatDone,
  ChatMeta,
  Citation,
  ComplianceResult,
  ConversationTurnWire,
  RankResult,
  RoutingRefusal,
} from "./types";

// Exported so other server-side callers (the /about document listing) share this one
// definition instead of re-declaring the same fallback.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export interface ChatRequest {
  query: string;
  history: ConversationTurnWire[];
  conversationId: string;
  highAccuracy: boolean;
  domain?: string | null;
}

export interface ChatHandlers {
  onMeta?: (meta: ChatMeta) => void;
  onCitations?: (citations: Citation[]) => void;
  onToken?: (text: string) => void;
  onDone?: (done: ChatDone) => void;
  onError?: (message: string) => void;
}

/** Raised for a non-streaming failure, so the UI can show the backend's own explanation. */
export class ChatRequestError extends Error {
  readonly status: number;
  readonly retryAfterSeconds?: number;
  /** Present only for the two routing-refusal 422s; lets `useChat` auto-retry against the right pipeline. */
  readonly routing?: RoutingRefusal;

  constructor(
    status: number,
    message: string,
    retryAfterSeconds?: number,
    routing?: RoutingRefusal,
  ) {
    super(message);
    this.name = "ChatRequestError";
    this.status = status;
    this.retryAfterSeconds = retryAfterSeconds;
    this.routing = routing;
  }
}

/** Send a chat turn and dispatch its stream to `handlers`; pre-flight failures throw, mid-stream failures fire `onError`. */
export async function streamChat(
  request: ChatRequest,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const token = await getAccessToken();
  if (!token) {
    throw new ChatRequestError(401, "You are not signed in. Use the demo login to continue.");
  }

  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      query: request.query,
      history: request.history,
      conversation_id: request.conversationId,
      high_accuracy: request.highAccuracy,
      ...(request.domain ? { domain: request.domain } : {}),
    }),
    signal,
  });

  if (!response.ok) {
    const { message, routing } = await parseFailure(response);
    throw new ChatRequestError(response.status, message, readRetryAfter(response), routing);
  }
  if (!response.body) {
    throw new ChatRequestError(502, "The server returned an empty response.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Dispatch only complete blocks; keep any partial tail in the buffer for the next chunk.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      dispatch(buffer.slice(0, boundary), handlers);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }

  // A final event with no trailing blank line still deserves delivery.
  if (buffer.trim()) dispatch(buffer, handlers);
}

/** POST /compliance-check: run the deterministic checklist sweep for one report; one JSON object, not a stream. */
export async function runComplianceCheck(
  company: string,
  category: "environmental" | "social" | "governance",
  framework: "gri" | "edgb" = "gri",
  signal?: AbortSignal,
): Promise<ComplianceResult> {
  const token = await getAccessToken();
  if (!token) {
    throw new ChatRequestError(401, "You are not signed in. Use the demo login to continue.");
  }

  const response = await fetch(`${API_BASE}/compliance-check`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ company, category, framework }),
    signal,
  });

  if (!response.ok) {
    const { message } = await parseFailure(response);
    throw new ChatRequestError(response.status, message, readRetryAfter(response));
  }
  return (await response.json()) as ComplianceResult;
}

/** POST /rank: corpus-wide aggregation/ranking, or one of its three honest refusals (not a thrown error). */
export async function runRank(query: string, signal?: AbortSignal): Promise<RankResult> {
  const token = await getAccessToken();
  if (!token) {
    throw new ChatRequestError(401, "You are not signed in. Use the demo login to continue.");
  }

  const response = await fetch(`${API_BASE}/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!response.ok) {
    const { message } = await parseFailure(response);
    throw new ChatRequestError(response.status, message, readRetryAfter(response));
  }
  return (await response.json()) as RankResult;
}

function dispatch(block: string, handlers: ChatHandlers): void {
  let event = "";
  const dataLines: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!event || dataLines.length === 0) return;

  let payload: unknown;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return; // A malformed frame is dropped rather than taking the stream down.
  }

  switch (event) {
    case "meta":
      handlers.onMeta?.(payload as ChatMeta);
      break;
    case "citations":
      handlers.onCitations?.((payload as { citations: Citation[] }).citations);
      break;
    case "token":
      handlers.onToken?.((payload as { text: string }).text);
      break;
    case "done":
      handlers.onDone?.(payload as ChatDone);
      break;
    case "error":
      handlers.onError?.((payload as { message: string }).message);
      break;
  }
}

/** Prefer the backend's own `detail`: its 422s and 429s explain what to do next, and may be structured. */
async function parseFailure(
  response: Response,
): Promise<{ message: string; routing?: RoutingRefusal }> {
  try {
    const body = await response.json();
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string") return { message: detail };
    if (Array.isArray(detail) && detail.length > 0) {
      // FastAPI validation errors are a list of objects.
      const first = detail[0] as { msg?: string };
      if (first?.msg) return { message: first.msg };
    }
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const structured = detail as Record<string, unknown>;
      if (typeof structured.message === "string") {
        const routing =
          structured.query_type === "aggregation_ranking" ||
          structured.query_type === "compliance_check"
            ? ({
                query_type: structured.query_type,
                company: typeof structured.company === "string" ? structured.company : null,
                category:
                  structured.category === "environmental" ||
                  structured.category === "social" ||
                  structured.category === "governance"
                    ? structured.category
                    : null,
              } satisfies RoutingRefusal)
            : undefined;
        return { message: structured.message, routing };
      }
    }
  } catch {
    // fall through
  }
  if (response.status === 401) {
    return { message: "Your session expired. Sign in again to continue." };
  }
  return { message: `Request failed (HTTP ${response.status}).` };
}

function readRetryAfter(response: Response): number | undefined {
  const header = response.headers.get("Retry-After");
  if (!header) return undefined;
  const seconds = Number.parseInt(header, 10);
  return Number.isFinite(seconds) ? seconds : undefined;
}
