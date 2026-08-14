"use client";

import { useCallback, useEffect, useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import ChatPanel from "@/components/ChatPanel";
import { api, AppError, type HealthResponse, type StatsResponse } from "@/lib/api";

type BackendState = "checking" | "online" | "offline";

export default function Home() {
  const [backend, setBackend] = useState<BackendState>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [indexed, setIndexed] = useState(false);
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);
  const [rerankOn, setRerankOn] = useState(true);
  const [cacheOn, setCacheOn] = useState(true);
  const [blockedReason, setBlockedReason] = useState<string | null>(null);

  const checkBackend = useCallback(async () => {
    setBackend("checking");
    try {
      const h = await api.health();
      setHealth(h);
      setIndexed(h.indexed);
      setBackend("online");
      try {
        setStats(await api.stats());
      } catch {
        /* stats are non-critical */
      }
    } catch {
      setBackend("offline");
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    checkBackend();
  }, [checkBackend]);

  const handleIngested = useCallback(
    (res: { message: string; pages: number; children: number }) => {
      setIndexed(true);
      setBlockedReason(null);
      setIngestMsg(`${res.message}`);
      // refresh stats (cache entries, ingests)
      api.stats().then(setStats).catch(() => undefined);
    },
    [],
  );

  const handleUploadError = useCallback((msg: string) => {
    setIngestMsg(null);
    setBlockedReason(msg);
  }, []);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__inner">
          <div className="brand">
            <span className="brand__mark" aria-hidden="true">D</span>
            DocuQA
          </div>
          <div className="header-status">
            {backend === "checking" && (
              <>
                <span className="dot" /> checking backend…
              </>
            )}
            {backend === "online" && (
              <>
                <span className="dot dot--ok" /> backend online
                {health?.providers?.length
                  ? ` · ${health.providers.join(", ")}`
                  : " · no LLM provider (rule-based mode)"}
              </>
            )}
            {backend === "offline" && (
              <>
                <span className="dot dot--bad" /> backend offline
              </>
            )}
          </div>
        </div>
      </header>

      <main className="app-main">
        {backend === "offline" && (
          <div className="alert alert--error" role="alert">
            <span>🔌</span>
            <span>
              Cannot reach the API at {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}.
              Start it with <code>uvicorn api.main:app --port 8000</code>, then retry.
            </span>
            <button className="alert__retry" onClick={checkBackend}>
              Retry
            </button>
          </div>
        )}

        {backend === "online" && (
          <div className="grid">
            <aside className="grid__side">
              <UploadPanel onIngested={handleIngested} onError={handleUploadError} />

              {ingestMsg && !blockedReason && (
                <div className="alert alert--ok" role="status">
                  <span>✅</span>
                  <span>{ingestMsg}</span>
                </div>
              )}

              <section className="panel" aria-label="Retrieval settings">
                <h2>Retrieval</h2>
                <div className="toggle-row">
                  <div>
                    <div className="toggle-row__label">Cross-encoder rerank</div>
                    <div className="toggle-row__hint">Better precision on top-4</div>
                  </div>
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={rerankOn}
                      onChange={(e) => setRerankOn(e.target.checked)}
                    />
                    <span className="switch__track"><span className="switch__thumb" /></span>
                  </label>
                </div>
                <div className="toggle-row">
                  <div>
                    <div className="toggle-row__label">Semantic cache</div>
                    <div className="toggle-row__hint">Reuse answers for similar questions</div>
                  </div>
                  <label className="switch">
                    <input
                      type="checkbox"
                      checked={cacheOn}
                      onChange={(e) => setCacheOn(e.target.checked)}
                    />
                    <span className="switch__track"><span className="switch__thumb" /></span>
                  </label>
                </div>
              </section>

              {stats && (
                <section className="panel" aria-label="Service statistics">
                  <h2>Service</h2>
                  <div className="stat-grid">
                    <div className="stat">
                      <div className="stat__value">{stats.cache_entries}</div>
                      <div className="stat__label">cache entries</div>
                    </div>
                    <div className="stat">
                      <div className="stat__value">{stats.injections_blocked}</div>
                      <div className="stat__label">blocked attacks</div>
                    </div>
                    <div className="stat">
                      <div className="stat__value">{stats.queries}</div>
                      <div className="stat__label">queries</div>
                    </div>
                    <div className="stat">
                      <div className="stat__value">{stats.llm_failures}</div>
                      <div className="stat__label">LLM failures</div>
                    </div>
                  </div>
                  {stats.providers?.length > 0 && (
                    <div className="toggle-row__hint" style={{ marginTop: 10 }}>
                      Providers: {stats.providers.join(", ")}
                    </div>
                  )}
                </section>
              )}
            </aside>

            <div className="grid__main">
              <ChatPanel
                enabled={indexed}
                rerankOn={rerankOn}
                cacheOn={cacheOn}
                onBlocked={setBlockedReason}
              />
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        DocuQA · parent-child chunking · hybrid retrieval (BM25 + dense) · cross-encoder rerank ·
        guarded LLM generation · built for the AI Engineer Intern assessment
      </footer>
    </div>
  );
}