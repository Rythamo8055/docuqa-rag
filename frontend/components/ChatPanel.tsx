"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, AppError, type QueryResponse } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: {
    provider?: string | null;
    model?: string | null;
    fromCache?: boolean;
    grounding?: { grounded: boolean; reason: string };
    faithfulness?: number;
    relevance?: number;
    blocked?: string | null;
    citations?: { page: number; chunk_id: string; text: string; score?: number }[];
  };
}

interface ChatPanelProps {
  enabled: boolean;
  rerankOn: boolean;
  cacheOn: boolean;
  onBlocked: (reason: string | null) => void;
}

export default function ChatPanel({ enabled, rerankOn, cacheOn, onBlocked }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const ask = useCallback(async () => {
    const q = input.trim();
    if (!q || busy || !enabled) return;
    setInput("");
    setError(null);

    const userMsg: Message = { id: `u${nextId.current++}`, role: "user", content: q };
    setMessages((m) => [...m, userMsg]);
    setBusy(true);

    try {
      const res = await api.query(q, { rerankOn, cacheOn });
      const assistantMsg: Message = {
        id: `a${nextId.current++}`,
        role: "assistant",
        content: res.blocked
          ? res.answer
          : res.answer,
        meta: {
          provider: res.provider,
          model: res.model,
          fromCache: res.from_cache,
          grounding: res.grounding,
          faithfulness: res.faithfulness,
          relevance: res.relevance,
          blocked: res.blocked,
          citations: res.context_chunks?.slice(0, 3).map((c) => ({
            page: c.page,
            chunk_id: c.chunk_id,
            text: c.text,
            score: c.rerank_score ?? c.similarity,
          })),
        },
      };
      setMessages((m) => [...m, assistantMsg]);
      onBlocked(res.blocked);
    } catch (err) {
      const msg =
        err instanceof AppError
          ? friendly(err)
          : "Something went wrong. Please try again.";
      setError(msg);
      setMessages((m) => [
        ...m,
        { id: `a${nextId.current++}`, role: "assistant", content: `⚠️ ${msg}` },
      ]);
    } finally {
      setBusy(false);
    }
  }, [input, busy, enabled, rerankOn, cacheOn, onBlocked]);

  function friendly(err: AppError): string {
    switch (err.code) {
      case "NETWORK":
        return "Cannot reach the backend server. Make sure it is running, then retry.";
      case "TIMEOUT":
        return "The request timed out. The document may be large — try a more specific question.";
      case "HTTP_429":
        return "Rate limit reached. Please wait a moment before asking again.";
      case "HTTP_5XX":
        return "The backend hit a temporary error. Retrying usually works.";
      case "HTTP_400":
        return err.message;
      default:
        return err.message;
    }
  }

  return (
    <section className="panel" aria-label="Ask a question">
      <h2>Ask the document</h2>

      {!enabled ? (
        <div className="empty">
          <span className="empty__icon" aria-hidden="true">🗂️</span>
          <div className="empty__title">No document indexed yet</div>
          <div className="empty__hint">
            Upload a PDF on the left to start asking questions. Answers are grounded in the
            document with page-level citations.
          </div>
        </div>
      ) : (
        <div className="chat">
          <div className="thread" ref={threadRef} role="log" aria-live="polite">
            {messages.length === 0 && !busy && (
              <div className="empty">
                <span className="empty__icon" aria-hidden="true">💬</span>
                <div className="empty__title">Ask anything about the document</div>
                <div className="empty__hint">
                  Example: “What are the key findings?” — answers include [Page X, Chunk Y]
                  citations you can verify.
                </div>
              </div>
            )}

            {messages.map((m) => (
              <MessageView key={m.id} msg={m} />
            ))}

            {busy && (
              <div className="thinking" aria-label="Retrieving and generating answer">
                <span className="thinking__dots" aria-hidden="true">
                  <span /><span /><span />
                </span>
                Retrieving, reranking & generating…
              </div>
            )}

            {error && (
              <div className="alert alert--error" role="alert">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask();
                }
              }}
              placeholder="Ask a question… (Enter to send, Shift+Enter for newline)"
              disabled={busy}
              aria-label="Your question"
              rows={1}
            />
            <button
              className="composer__send"
              onClick={ask}
              disabled={busy || !input.trim()}
              aria-label="Send question"
            >
              {busy ? "…" : "Ask"}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}

function MessageView({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  const m = msg.meta;

  return (
    <div className={`msg ${isUser ? "msg--user" : "msg--assistant"}`}>
      <div className="msg__avatar" aria-hidden="true">
        {isUser ? "Y" : "AI"}
      </div>
      <div className="msg__body">
        <div className="msg__text">{msg.content}</div>

        {m && !isUser && (
          <>
            <div className="msg__meta">
              {m.blocked && <span className="badge badge--blocked">🚫 {m.blocked}</span>}
              {!m.blocked && m.grounding?.grounded && (
                <span className="badge badge--grounded">✓ {m.grounding.reason}</span>
              )}
              {!m.blocked && m.grounding && !m.grounding.grounded && (
                <span className="badge badge--ungrounded">⚠ {m.grounding.reason}</span>
              )}
              {m.fromCache && <span className="badge badge--cache">⚡ from cache</span>}
              {m.provider && m.provider !== "rule-based" && (
                <span>
                  {m.provider} · {m.model}
                </span>
              )}
              {typeof m.faithfulness === "number" && !m.blocked && (
                <span>faithfulness {m.faithfulness.toFixed(2)}</span>
              )}
              {typeof m.relevance === "number" && !m.blocked && (
                <span>relevance {m.relevance.toFixed(2)}</span>
              )}
            </div>

            {m.citations && m.citations.length > 0 && (
              <div className="citations">
                {m.citations.map((c, i) => (
                  <div key={i} className="citation">
                    <div className="citation__head">
                      [Page {c.page}, Chunk {c.chunk_id}]
                      {typeof c.score === "number" && (
                        <span style={{ fontWeight: 400, marginLeft: 8, color: "var(--ink-faint)" }}>
                          score {c.score.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <div className="citation__text">{c.text}</div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}