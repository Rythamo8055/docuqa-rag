"""
RAGService — the shared, framework-agnostic core of the RAG system.

One class that owns the full pipeline (ingest → index → retrieve →
rerank → generate) and is reused by three frontends:
  1. FastAPI backend (api/main.py)  — deployed on Render
  2. Streamlit app (app.py)         — local demo
  3. Eval harness (evals/run_eval.py) — offline metrics

All guardrails (input/output/upload/resilience) are enforced here, so
every frontend gets the same security posture by default.
"""
from typing import Dict, List, Optional
import json
import logging
import os
import tempfile

from src.pdf_utils import process_pdf
from src.embeddings import EmbeddingManager, ChromaVectorStore
from src.llm import (
    GroundedGenerator,
    check_grounding,
    extractive_answer,
    NOT_FOUND_PHRASE,
)
from src.llm_router import LLMRouter
from src.hybrid_search import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.cache import SemanticCache
from src.tracing import Tracer
from src.evaluation import compute_faithfulness, compute_relevance
from src.input_guardrails import (
    sanitize_input,
    detect_injection,
    validate_query,
    RateLimiter,
)
from src.output_guardrails import filter_output
from src.upload_security import (
    validate_upload,
    sanitize_document_text,
    create_secure_temp_file,
)
from src.resilience import (
    empty_answer_fallback,
    ErrorBucket,
    CircuitBreaker,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)

TOP_K = 4
RETRIEVE_N = TOP_K * 2  # fetch more, then rerank down
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


class RAGService:
    """Stateless-safe service: one instance per process."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        cache_path: str = ".cache/rag_cache.sqlite",
        rerank_on: bool = True,
        cache_on: bool = True,
        llm_enabled: bool = True,
    ):
        # Overridable via env (Render persistent disk, etc.)
        self.persist_dir = os.getenv("DATA_DIR", persist_dir)
        self.cache_path = os.getenv("CACHE_PATH", cache_path)
        self.rerank_on = rerank_on
        self.cache_on = cache_on
        # When False, query() skips the LLM entirely and answers with the
        # deterministic extractive fallback (offline/demo/CI mode).
        self.llm_enabled = llm_enabled

        self.embedding_manager = EmbeddingManager()
        self.vector_store: Optional[ChromaVectorStore] = None
        self.hybrid: Optional[HybridRetriever] = None
        self.parents: Dict[str, Dict] = {}

        # Rehydrate a persisted index so restarts don't lose documents.
        try:
            persisted = ChromaVectorStore(
                persist_dir=self.persist_dir,
                embedding_manager=self.embedding_manager,
            )
            stored_chunks = persisted.load_all()
            if stored_chunks:
                self.vector_store = persisted
                self.hybrid = HybridRetriever(persisted)
                self.hybrid.index_bm25(stored_chunks)
                logger.info(
                    f"Rehydrated {len(stored_chunks)} chunk(s) from {self.persist_dir}"
                )
            # Parent sections (small-to-big context) live in a sidecar JSON
            parents_file = os.path.join(self.persist_dir, "parents.json")
            if os.path.exists(parents_file):
                with open(parents_file, "r", encoding="utf-8") as f:
                    self.parents = json.load(f)
                logger.info(f"Rehydrated {len(self.parents)} parent section(s)")
        except Exception as e:
            logger.warning(f"Index rehydration skipped: {e}")

        self.reranker = CrossEncoderReranker()
        self.cache = SemanticCache(db_path=self.cache_path)
        self.tracer = Tracer()
        self.router = LLMRouter()
        self.generator = GroundedGenerator(self.router)
        self.rate_limiter = RateLimiter()
        self.llm_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)

        # Session-level counters (survive for the process lifetime)
        self.stats = {
            "ingests": 0,
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "injections_blocked": 0,
            "llm_failures": 0,
            "redactions": 0,
            "total_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        max_pages: int = 400,
    ) -> Dict:
        """
        Validate → sanitize → chunk → embed → index.

        Returns: {"ok": bool, "message": str, "error": Optional[str],
                  "pages": int, "children": int, "parents": int}
        """
        err = validate_upload(
            filename=filename,
            data=file_bytes,
            max_size_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
        )
        if not err.ok:
            return {"ok": False, "message": "", "error": err.reason,
                    "pages": 0, "children": 0, "parents": 0}

        tmp_path, tmp_err = create_secure_temp_file(file_bytes, suffix=".pdf")
        if tmp_path is None:
            return {"ok": False, "message": "", "error": tmp_err or "Failed to create secure temp file.",
                    "pages": 0, "children": 0, "parents": 0}
        try:
            result = process_pdf(tmp_path, max_pages=max_pages)
            children, parents = result["children"], result["parents"]
            if not children:
                return {"ok": False, "message": "", "error": "No extractable text found in this PDF.",
                        "pages": 0, "children": 0, "parents": 0}

            # Sanitize every chunk (hidden text / script / path injection)
            sanitized = 0
            for chunk in children:
                cleaned = sanitize_document_text(chunk["text"])
                if cleaned != chunk["text"]:
                    chunk["text"] = cleaned
                    sanitized += 1
            for parent in parents:
                parent["text"] = sanitize_document_text(parent["text"])

            self.parents = {p["parent_id"]: p for p in parents}

            # Persist parents for restart rehydration (small-to-big context)
            try:
                os.makedirs(self.persist_dir, exist_ok=True)
                with open(
                    os.path.join(self.persist_dir, "parents.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(self.parents, f)
            except Exception as e:
                logger.warning(f"Failed to persist parents.json: {e}")

            vs = ChromaVectorStore(
                persist_dir=self.persist_dir,
                embedding_manager=self.embedding_manager,
            )
            vs.add_chunks(children)

            hybrid = HybridRetriever(vs)
            hybrid.index_bm25(children)

            self.vector_store = vs
            self.hybrid = hybrid
            self.stats["ingests"] += 1

            self.tracer.trace(
                "ingest",
                filename=filename,
                pages=result["total_pages"],
                children=len(children),
                parents=len(parents),
                sanitized=sanitized,
            )
            return {
                "ok": True,
                "message": f"Indexed {result['total_pages']} pages → {len(children)} chunks",
                "error": None,
                "pages": result["total_pages"],
                "children": len(children),
                "parents": len(parents),
                "sanitized": sanitized,
            }
        except Exception as e:
            logger.exception("Ingest error")
            return {"ok": False, "message": "", "error": f"Error processing PDF: {e}",
                    "pages": 0, "children": 0, "parents": 0}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def is_indexed(self) -> bool:
        return self.hybrid is not None

    def query(
        self,
        question: str,
        rerank_on: Optional[bool] = None,
        cache_on: Optional[bool] = None,
    ) -> Dict:
        """
        Full guarded pipeline: sanitize → inject-scan → rate-limit →
        cache → hybrid retrieve → rerank → small-to-big → generate →
        output filter → metrics.

        Never raises: every failure path degrades to a safe response.
        """
        errors = ErrorBucket()
        if not self.is_indexed():
            return self._fail_answer(
                "Please upload and index a PDF first.",
                errors=errors, blocked="no-index",
            )

        rerank_on = self.rerank_on if rerank_on is None else rerank_on
        cache_on = self.cache_on if cache_on is None else cache_on

        # 1) Input guardrails
        clean = sanitize_input(question)
        ok, reason = validate_query(clean)
        if not ok:
            return self._fail_answer(
                "⚠️ Please ask a meaningful question (input contained no usable text).",
                errors=errors, blocked=reason or "empty-after-sanitize",
            )
        inj = detect_injection(clean)
        if inj.flagged:
            self.stats["injections_blocked"] += 1
            logger.warning("Injection blocked [%s]: %.80s", inj.reason, clean)
            return self._fail_answer(
                f"🚫 Question blocked by the input guardrail ({inj.reason}). "
                "If this was a mistake, rephrase it.",
                errors=errors, blocked=inj.reason,
            )
        if not self.rate_limiter.allow(self._session_id())[0]:
            return self._fail_answer(
                "⏳ Rate limit reached. Please wait a moment before asking again.",
                errors=errors, blocked="rate-limit",
            )

        self.stats["queries"] += 1

        # 2) Semantic cache
        emb = self.embedding_manager.embed_query(clean)
        if cache_on:
            hit = self.cache.get(clean, emb)
            if hit["hit"]:
                self.stats["cache_hits"] += 1
                self.tracer.trace("cache", query=clean, similarity=hit["similarity"])
                return {
                    "answer": hit["answer"],
                    "provider": "cache",
                    "model": None,
                    "chunks": [],
                    "context_chunks": [],
                    "from_cache": True,
                    "grounding": {"grounded": True, "reason": "Cached answer"},
                    "faithfulness": 1.0,
                    "relevance": 0.0,
                    "blocked": None,
                    "errors": "",
                    "metrics": {},
                }
            self.stats["cache_misses"] += 1

        # 3) Hybrid retrieve — depth scales with corpus size so larger
        # documents don't push relevant chunks out of the candidate pool.
        corpus_n = len(self.hybrid._corpus) if self.hybrid._corpus else 0
        retrieve_n = min(max(RETRIEVE_N, corpus_n), RETRIEVE_N * 3)
        candidates = self.hybrid.retrieve(clean, top_k=retrieve_n)

        # 4) Rerank
        if rerank_on:
            top = self.reranker.rerank(clean, candidates, top_k=TOP_K)
        else:
            top = candidates[:TOP_K]

        # 5) Small-to-big
        context_chunks = self._resolve_parents(top)

        # 6) Generate (circuit breaker + safe fallback)
        provider, model = None, None
        if not self.llm_enabled:
            # Offline/CI mode: deterministic extractive answers only.
            ext = extractive_answer(clean, top) if context_chunks else None
            answer = ext if ext else NOT_FOUND_PHRASE
            provider = "extractive" if ext else "rule-based"
        elif self.llm_breaker.state == "OPEN":
            errors.add("circuit-open", "LLM circuit open — using fallback response")
            answer = empty_answer_fallback(clean, len(context_chunks))
            provider = "rule-based"
        else:
            try:
                answer, provider, model = self.llm_breaker.call(
                    self.generator.generate_response,
                    clean, context_chunks, False,
                )
            except CircuitOpenError:
                errors.add("circuit-open", "LLM circuit open — using fallback response")
                answer = empty_answer_fallback(clean, len(context_chunks))
                provider = "rule-based"
            except Exception as e:
                self.stats["llm_failures"] += 1
                errors.add("llm-error", f"LLM call failed ({e})")
                answer = empty_answer_fallback(clean, len(context_chunks))
                provider = "rule-based"

        # 6.5) Refusal handling — the LLM returned the "not found" phrase
        # despite strong retrieval. This is usually a model refusal, not a
        # true miss, so try the deterministic extractive answer first
        # (grounded by construction). Only if the document genuinely has no
        # overlap with the question do we keep the refusal, normalized to
        # the exact mandated phrase.
        if NOT_FOUND_PHRASE.lower() in answer.lower():
            if context_chunks and provider != "extractive":
                ext = extractive_answer(clean, top)
                if ext:
                    answer = ext
                    provider = "extractive"
                    errors.add(
                        "extractive-fallback",
                        "LLM returned not-found despite retrieval; used extractive answer",
                    )
            if NOT_FOUND_PHRASE.lower() in answer.lower():
                answer = NOT_FOUND_PHRASE

        # 7) Output guardrails
        filtered = filter_output(answer)
        if filtered.pii_redacted:
            self.stats["redactions"] += 1
        final_answer = filtered.text

        # 8) Metrics
        faithfulness = compute_faithfulness(final_answer, context_chunks)
        relevance = compute_relevance(clean, top)
        grounding = check_grounding(final_answer, context_chunks)

        self.tracer.trace(
            "qa",
            query=clean,
            candidates=len(candidates),
            reranked=len(top),
            provider=provider,
            model=model,
            grounding=grounding,
            errors=errors.summary(),
        )

        # 9) Cache (only clean, real-LLM answers)
        if cache_on and provider and provider not in ("rule-based", "cache") \
                and not filtered.pii_redacted and not filtered.unsafe_detected:
            self.cache.put(clean, emb, final_answer, provider)

        return {
            "answer": final_answer,
            "provider": provider,
            "model": model,
            "chunks": top,
            "context_chunks": context_chunks,
            "from_cache": False,
            "grounding": grounding,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "blocked": None,
            "errors": errors.summary(),
            "metrics": {
                "candidates": len(candidates),
                "reranked": len(top),
                "cache_similarity": None,
            },
        }

    def stats_snapshot(self) -> Dict:
        """Public stats for health/status endpoints."""
        return {
            **self.stats,
            "indexed": self.is_indexed(),
            "documents": len(self.parents),
            "cache_entries": self.cache.size(),
            "providers": self.router.available_providers(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _session_id(self) -> str:
        """Single shared session for this service instance (API = 1 process)."""
        return "api"

    def _resolve_parents(self, children: List[Dict]) -> List[Dict]:
        """Small-to-big: map children back to parent sections, dedupe."""
        seen, context = set(), []
        for child in children:
            pid = child.get("parent_id")
            if pid in seen:
                continue
            seen.add(pid)
            parent = self.parents.get(pid, {})
            text = parent.get("text", child["text"])
            context.append({
                "text": text,
                "page": child["page"],
                "chunk_id": child["chunk_id"],
                "parent_id": pid,
            })
        return context

    def _fail_answer(self, message: str, errors: ErrorBucket, blocked: str) -> Dict:
        return {
            "answer": message,
            "provider": None,
            "model": None,
            "chunks": [],
            "context_chunks": [],
            "from_cache": False,
            "grounding": {"grounded": False, "reason": blocked},
            "faithfulness": 0.0,
            "relevance": 0.0,
            "blocked": blocked,
            "errors": errors.summary(),
            "metrics": {},
        }