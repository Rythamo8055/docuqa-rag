"use client";

import { useCallback, useRef, useState } from "react";
import { api, AppError, type IngestResponse } from "@/lib/api";
import { validatePdf, UPLOAD_ERROR_COPY, type UploadError } from "@/lib/upload";

interface UploadPanelProps {
  onIngested: (res: IngestResponse) => void;
  onError: (err: string) => void;
}

type Phase = "idle" | "validating" | "uploading" | "done" | "error";

export default function UploadPanel({ onIngested, onError }: UploadPanelProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const busy = phase === "validating" || phase === "uploading";

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file || busy) return;
      setUploadError(null);
      setFileName(file.name);

      // 1) Client-side validation (mirrors backend)
      setPhase("validating");
      const check = await validatePdf(file);
      if (!check.ok) {
        setPhase("error");
        const copy = UPLOAD_ERROR_COPY[check.error as UploadError];
        setUploadError(
          check.message ??
            (copy ? `${copy.title} — ${copy.hint}` : "The file could not be validated."),
        );
        return;
      }

      // 2) Upload to backend (with pseudo-progress for feedback)
      setPhase("uploading");
      setProgress(15);
      const tick = setInterval(() => {
        setProgress((p) => Math.min(p + Math.random() * 9, 85));
      }, 350);
      try {
        const res = await api.ingest(file);
        clearInterval(tick);
        setProgress(100);
        setPhase("done");
        onIngested(res);
      } catch (err) {
        clearInterval(tick);
        setPhase("error");
        const msg =
          err instanceof AppError
            ? `${friendly(err)} (${err.code})`
            : "Upload failed. Please try again.";
        setUploadError(msg);
        onError(msg);
      }
    },
    [busy, onIngested, onError],
  );

  function friendly(err: AppError): string {
    if (err.code === "HTTP_429") return "Rate limit reached — wait a moment and retry";
    if (err.code === "HTTP_5XX") return "Backend error — this is usually transient, try again";
    if (err.code === "HTTP_400") return err.message;
    return err.message;
  }

  return (
    <section className="panel" aria-label="Upload a PDF">
      <h2>Document</h2>

      <div
        className={`dropzone${dragActive ? " dropzone--active" : ""}${busy ? " dropzone--disabled" : ""}`}
        role="button"
        tabIndex={0}
        aria-label="Upload a PDF document"
        onClick={() => !busy && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !busy) inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          disabled={busy}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <span className="dropzone__icon" aria-hidden="true">
          {busy ? "⏳" : "📄"}
        </span>
        <div className="dropzone__title">
          {busy ? "Processing…" : phase === "done" ? fileName : "Drop a PDF or click to choose"}
        </div>
        <div className="dropzone__hint">PDF only · up to 20 MB · processed locally</div>
      </div>

      {(phase === "uploading" || phase === "done") && (
        <div className="upload-progress">
          <div className="progress-track">
            <div className="progress-fill" style={{ transform: `scaleX(${progress / 100})` }} />
          </div>
          {phase === "uploading" && (
            <div className="upload-status">
              <span>Indexing: extracting, chunking, embedding…</span>
              <span>{Math.round(progress)}%</span>
            </div>
          )}
          {phase === "done" && (
            <div className="upload-status upload-status--ok">
              <span>✅ {fileName} indexed</span>
            </div>
          )}
        </div>
      )}

      {phase === "error" && uploadError && (
        <div className="alert alert--error" role="alert">
          <span>⚠️</span>
          <span>{uploadError}</span>
          <button className="alert__retry" onClick={() => inputRef.current?.click()}>
            Retry
          </button>
        </div>
      )}
    </section>
  );
}