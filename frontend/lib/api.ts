/**
 * API client for the DocuQA backend with a full error taxonomy.
 *
 * Every failure mode is classified into an AppError with a stable `code`
 * so the UI can render a specific message + recovery action:
 *   - NETWORK        fetch failed / offline
 *   - TIMEOUT        request exceeded the deadline (AbortController)
 *   - HTTP_4xx       backend validation / auth / rate limit / not found
 *   - HTTP_5xx       backend crash (transient — retry works)
 *   - BAD_RESPONSE   backend answered but with an unexpected shape
 */

export type AppErrorCode =
  | "NETWORK"
  | "TIMEOUT"
  | "HTTP_400"
  | "HTTP_401"
  | "HTTP_403"
  | "HTTP_404"
  | "HTTP_429"
  | "HTTP_5XX"
  | "BAD_RESPONSE";

export class AppError extends Error {
  code: AppErrorCode;
  status?: number;
  retryable: boolean;

  constructor(code: AppErrorCode, message: string, status?: number) {
    super(message);
    this.name = "AppError";
    this.code = code;
    this.status = status;
    this.retryable = code === "NETWORK" || code === "TIMEOUT" || code === "HTTP_5XX";
  }
}

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const DEFAULT_TIMEOUT_MS = 120_000; // LLM generation can take a while

export interface HealthResponse {
  status: string;
  indexed: boolean;
  providers: string[];
}

export interface IngestResponse {
  ok: boolean;
  message: string;
  error: string | null;
  pages: number;
  children: number;
  parents: number;
  sanitized: number;
}

export interface ChunkResult {
  text: string;
  page: number;
  chunk_id: string;
  parent_id?: string;
  similarity?: number;
  rerank_score?: number;
}

export interface QueryResponse {
  answer: string;
  provider: string | null;
  model: string | null;
  chunks: ChunkResult[];
  context_chunks: ChunkResult[];
  from_cache: boolean;
  grounding: { grounded: boolean; reason: string };
  faithfulness: number;
  relevance: number;
  blocked: string | null;
  errors: string;
  metrics: Record<string, unknown>;
}

export interface StatsResponse {
  ingests: number;
  queries: number;
  cache_hits: number;
  cache_misses: number;
  injections_blocked: number;
  llm_failures: number;
  redactions: number;
  indexed: boolean;
  documents: number;
  cache_entries: number;
  providers: string[];
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // FormData must NOT get a Content-Type header — the browser sets the
  // multipart boundary itself. JSON bodies set it explicitly below.
  const isFormData = init.body instanceof FormData;
  const headers = isFormData
    ? (init.headers as Record<string, string> | undefined)
    : { "Content-Type": "application/json", ...(init.headers || {}) };

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
  } catch (err) {
    // Distinguish timeout (AbortController fired) from real network failure
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new AppError(
        "TIMEOUT",
        `The request took longer than ${Math.round(timeoutMs / 1000)}s. The backend may be retrying a busy LLM provider — try again.`,
      );
    }
    throw new AppError(
      "NETWORK",
      "Cannot reach the backend. Check that the API server is running and your connection is online.",
    );
  } finally {
    clearTimeout(timer);
  }

  if (res.status === 429) {
    throw new AppError("HTTP_429", "Rate limit reached. Please wait a moment and try again.", 429);
  }
  if (res.status >= 500) {
    throw new AppError("HTTP_5XX", "The backend hit an unexpected error. Please retry.", res.status);
  }
  if (res.status === 404) {
    throw new AppError("HTTP_404", "The requested resource was not found.", 404);
  }
  if (res.status === 401) {
    throw new AppError("HTTP_401", "Authentication failed.", 401);
  }
  if (res.status === 403) {
    throw new AppError("HTTP_403", "You do not have permission to do this.", 403);
  }
  if (res.status >= 400) {
    // Try to surface the backend's detail message
    let detail = `Request failed with status ${res.status}.`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep default */
    }
    throw new AppError("HTTP_400", detail, res.status);
  }

  try {
    return (await res.json()) as T;
  } catch {
    throw new AppError("BAD_RESPONSE", "The backend returned an unreadable response.", res.status);
  }
}

export const api = {
  health: () => request<HealthResponse>("/health", {}, 10_000),
  stats: () => request<StatsResponse>("/stats", {}, 10_000),
  ingest: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<IngestResponse>("/ingest", { method: "POST", body: form }, 300_000);
  },
  query: (question: string, opts: { rerankOn?: boolean; cacheOn?: boolean } = {}) =>
    request<QueryResponse>(
      "/query",
      {
        method: "POST",
        body: JSON.stringify({
          question,
          rerank_on: opts.rerankOn,
          cache_on: opts.cacheOn,
        }),
      },
      // LLM queries can legitimately take minutes: Gemini 503 "high demand"
      // retries (with backoff) + generation time. The UI shows a live
      // "working" state, so a generous deadline beats a premature abort.
      300_000,
    ),
};