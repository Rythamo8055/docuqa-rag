"""
FastAPI backend for the RAG system — the deployed API layer.

Serves the shared RAGService (src/rag_service.py) over REST so the
Next.js frontend (and any client) can:
  - POST /ingest   upload a PDF → validate → chunk → embed → index
  - POST /query    ask a question → guarded pipeline → answer + metrics
  - GET  /health   liveness probe (used by Render)
  - GET  /stats    service statistics

CORS is wide-open for the frontend origin (configure FRONTEND_ORIGIN).

Run locally:   uvicorn api.main:app --reload --port 8000
"""
from typing import List, Dict
import logging
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_service import RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuQA RAG API",
    version="1.0.0",
    description="Grounded document Q&A: hybrid retrieval + reranking + guarded LLM generation.",
)

# CORS: allow the Next.js frontend (Vercel) and local dev
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One service instance per process (models are cached in-memory)
service = RAGService()


class QueryRequest(BaseModel):
    question: str
    rerank_on: bool | None = None
    cache_on: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    provider: str | None
    model: str | None
    chunks: List[Dict]
    context_chunks: List[Dict]
    from_cache: bool
    grounding: Dict
    faithfulness: float
    relevance: float
    blocked: str | None
    errors: str  # comma-joined error codes (e.g. "llm-error, circuit-open")
    metrics: Dict


@app.get("/health")
def health():
    """Liveness probe for Render/uptime checks."""
    return {
        "status": "ok",
        "indexed": service.is_indexed(),
        "providers": service.router.available_providers(),
    }


@app.get("/stats")
def stats():
    """Service statistics (cache size, counters, providers)."""
    return service.stats_snapshot()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload + index a PDF. Returns ingest summary or a 400 with reason."""
    data = await file.read()
    result = service.ingest(data, file.filename or "upload.pdf")
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question against the indexed document(s)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    return service.query(
        req.question,
        rerank_on=req.rerank_on,
        cache_on=req.cache_on,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))