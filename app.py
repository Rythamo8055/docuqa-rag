"""
Intelligent Document Q&A System with Retrieval-Augmented Generation (RAG).

Pipeline: PDF upload → secure validation → pypdf extraction → parent-child
chunking → MiniLM embeddings → ChromaDB → hybrid search (BM25+RRF) →
cross-encoder rerank → grounded LLM (router) → output guardrails → cited answer.

Security & reliability layers:
- Input guardrails: sanitize, injection detection, rate limiting
- Upload security: magic bytes, size caps, document sanitization
- Output guardrails: PII redaction, leakage detection, unsafe filter
- Resilience: retries, circuit breaker, safe fallbacks

Usage:
    streamlit run app.py
"""
from typing import List, Dict
import logging
import os

import streamlit as st

from src.pdf_utils import process_pdf
from src.embeddings import EmbeddingManager, ChromaVectorStore
from src.llm import GroundedGenerator, check_grounding
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
    safe_page_count,
    create_secure_temp_file,
)
from src.resilience import (
    safe_llm_call,
    empty_answer_fallback,
    ErrorBucket,
    CircuitBreaker,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocuQA - RAG System",
    page_icon="📄",
    layout="wide",
)

TOP_K = 4
RETRIEVE_N = TOP_K * 2  # fetch more, then rerank down


def init_state():
    """Initialize session state."""
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    if "router" not in st.session_state:
        st.session_state.router = LLMRouter()
    if "generator" not in st.session_state:
        st.session_state.generator = GroundedGenerator(st.session_state.router)
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "hybrid" not in st.session_state:
        st.session_state.hybrid = None
    if "reranker" not in st.session_state:
        st.session_state.reranker = CrossEncoderReranker()
    if "cache" not in st.session_state:
        st.session_state.cache = SemanticCache()
    if "tracer" not in st.session_state:
        st.session_state.tracer = Tracer()
    if "parents" not in st.session_state:
        st.session_state.parents = {}
    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rerank_on" not in st.session_state:
        st.session_state.rerank_on = True
    if "cache_on" not in st.session_state:
        st.session_state.cache_on = True
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()
    if "llm_breaker" not in st.session_state:
        st.session_state.llm_breaker = CircuitBreaker(
            failure_threshold=3, reset_timeout=60.0
        )
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "last_blocked" not in st.session_state:
        st.session_state.last_blocked = None


def process_upload(uploaded):
    """Validate → extract → chunk → embed → index. Returns summary strings."""
    # 1) Upload security: type/magic bytes/size validation
    data = uploaded.read()
    validation = validate_upload(
        filename=uploaded.name,
        data=data,
        max_size_mb=20,
    )
    if not validation.ok:
        return None, f"🚫 {validation.reason}"

    # 2) Write to a secure temp file (0600 perms, no shell interpolation)
    tmp_path, tmp_err = create_secure_temp_file(data, suffix=".pdf")
    if tmp_path is None:
        return None, f"🚫 {tmp_err or 'Failed to create secure temporary file.'}"

    try:
        # 3) Extract with page-count safety cap
        result = process_pdf(tmp_path, max_pages=400)
        children, parents = result["children"], result["parents"]
        if not children:
            return None, "No extractable text found in this PDF."

        # 4) Sanitize every chunk (hidden text / script / path injection)
        sanitized = 0
        for chunk in children:
            cleaned = sanitize_document_text(chunk["text"])
            if cleaned != chunk["text"]:
                chunk["text"] = cleaned
                sanitized += 1
        for parent in parents:
            parent["text"] = sanitize_document_text(parent["text"])
        logger.info(
            "Upload: %s sanitized=%d pages=%d children=%d",
            uploaded.name, sanitized, result["total_pages"], len(children),
        )

        st.session_state.parents = {p["parent_id"]: p for p in parents}

        vs = ChromaVectorStore(
            persist_dir="./chroma_db",
            embedding_manager=st.session_state.embedding_manager,
        )
        vs.add_chunks(children)

        hybrid = HybridRetriever(vs)
        hybrid.index_bm25(children)

        st.session_state.vector_store = vs
        st.session_state.hybrid = hybrid
        st.session_state.is_indexed = True

        st.session_state.tracer.trace(
            "ingest",
            pages=result["total_pages"],
            children=len(children),
            parents=len(parents),
        )

        n_pages = len({c["page"] for c in children})
        return f"✅ {n_pages} pages → {len(children)} chunks indexed", None
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        return None, f"Error processing PDF: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def resolve_parents(children: List[Dict]) -> List[Dict]:
    """
    Small-to-big: map retrieved children back to their parent sections.
    Context for the LLM uses parent text; citations keep the child's
    page + chunk id for precision. Dedupe by parent_id.
    """
    seen, context = set(), []
    for child in children:
        pid = child.get("parent_id")
        if pid in seen:
            continue
        seen.add(pid)
        parent = st.session_state.parents.get(pid, {})
        text = parent.get("text", child["text"])
        context.append({
            "text": text,
            "page": child["page"],
            "chunk_id": child["chunk_id"],
            "parent_id": pid,
        })
    return context


def run_query(question: str):
    """Full query pipeline. Returns dict with answer + diagnostics."""
    tracer = st.session_state.tracer
    emb = st.session_state.embedding_manager.embed_query(question)
    cache = st.session_state.cache
    errors = ErrorBucket()

    # 1) Input guardrails: sanitize → validate → injection scan → rate limit
    clean = sanitize_input(question)
    if not validate_query(clean)[0]:
        st.session_state.last_blocked = "empty-after-sanitize"
        return {
            "answer": "⚠️ Please ask a meaningful question (your input contained no usable text).",
            "provider": None, "model": None,
            "chunks": [], "context_chunks": [], "from_cache": False,
            "blocked": "empty-after-sanitize",
        }
    inj = detect_injection(clean)
    if inj.flagged:
        st.session_state.last_blocked = inj.reason
        logger.warning("Injection blocked [%s]: %.80s", inj.reason, clean)
        return {
            "answer": "🚫 That question was blocked by the input guardrail "
                      f"({inj.reason}). If this was a mistake, rephrase it.",
            "provider": None, "model": None,
            "chunks": [], "context_chunks": [], "from_cache": False,
            "blocked": inj.reason,
        }
    if not st.session_state.rate_limiter.allow():
        return {
            "answer": "⏳ Rate limit reached. Please wait a moment before asking again.",
            "provider": None, "model": None,
            "chunks": [], "context_chunks": [], "from_cache": False,
            "blocked": "rate-limit",
        }

    # 2) Semantic cache
    if st.session_state.cache_on:
        hit = cache.get(clean, emb)
        if hit["hit"]:
            tracer.trace("cache", query=clean, similarity=hit["similarity"])
            return {**hit, "chunks": [], "from_cache": True}

    # 3) Hybrid retrieve (BM25 + dense, RRF)
    candidates = st.session_state.hybrid.retrieve(clean, top_k=RETRIEVE_N)

    # 4) Rerank (cross-encoder)
    if st.session_state.rerank_on:
        top = st.session_state.reranker.rerank(clean, candidates, top_k=TOP_K)
    else:
        top = candidates[:TOP_K]

    # 5) Small-to-big: resolve parents for generation context
    context_chunks = resolve_parents(top)

    # 6) Grounded generation (with circuit breaker + safe fallback)
    breaker = st.session_state.llm_breaker
    if breaker.is_open():
        errors.add("LLM circuit open — using fallback response")
        answer, provider, model = empty_answer_fallback(
            clean, len(context_chunks), reason="circuit-open"
        )
    else:
        try:
            answer, provider, model = safe_llm_call(
                st.session_state.generator.generate_response,
                question=clean,
                chunks=context_chunks,
                stream=False,
            )
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            errors.add(f"LLM call failed ({e})")
            answer, provider, model = empty_answer_fallback(
                clean, len(context_chunks), reason="llm-error"
            )

    # 7) Output guardrails: filter the answer before it reaches the user
    filtered = filter_output(answer)
    if filtered.pii_redacted or filtered.leakage_detected or filtered.unsafe_detected:
        tracer.trace(
            "output_guardrail",
            pii=filtered.pii_redacted,
            leakage=filtered.leakage_detected,
            unsafe=filtered.unsafe_detected,
        )
    final_answer = filtered.text

    tracer.trace(
        "qa",
        query=clean,
        candidates=len(candidates),
        reranked=len(top),
        provider=provider,
        model=model,
        grounding=check_grounding(final_answer, context_chunks),
        errors=errors.summarize(),
    )

    # 8) Cache the answer (if enabled and a real LLM answered)
    if st.session_state.cache_on and provider != "rule-based" \
            and not filtered.pii_redacted and not filtered.unsafe_detected:
        cache.put(clean, emb, final_answer, provider)

    return {
        "answer": final_answer,
        "provider": provider,
        "model": model,
        "chunks": top,
        "context_chunks": context_chunks,
        "from_cache": False,
        "errors": errors.summarize(),
        "redaction": filtered.pii_redacted,
    }


def render_sidebar():
    """Sidebar: provider status, API keys, toggles."""
    with st.sidebar:
        st.title("⚙️ Settings")

        providers = st.session_state.router.available_providers()
        if providers:
            st.success(f"🟢 Active: {', '.join(providers)}")
        else:
            st.warning("🔴 No LLM provider. Add keys below or start Ollama.")

        with st.expander("🔑 API Keys", expanded=not providers):
            keys = {"GROQ_API_KEY": "Groq (free)",
                    "GEMINI_API_KEY": "Gemini (free)",
                    "OPENAI_API_KEY": "OpenAI",
                    "ANTHROPIC_API_KEY": "Anthropic"}
            for env, label in keys.items():
                current = os.getenv(env, "")
                val = st.text_input(label, value=current, type="password", key=env)
                if val and val != current:
                    os.environ[env] = val
                    st.rerun()
            if st.button("💾 Reload router"):
                st.session_state.router = LLMRouter()
                st.session_state.generator = GroundedGenerator(st.session_state.router)
                st.rerun()

        st.markdown("---")
        st.subheader("🧪 Retrieval Options")
        st.session_state.rerank_on = st.toggle(
            "Cross-encoder rerank", value=st.session_state.rerank_on,
            help="Better precision; downloads ~90MB model once")
        st.session_state.cache_on = st.toggle(
            "Semantic cache", value=st.session_state.cache_on,
            help="Reuse answers for similar questions (~80% cost cut)")

        with st.expander("🧊 Cache info"):
            st.write(f"Entries: {st.session_state.cache.size()}")
            if st.button("Clear cache"):
                st.session_state.cache.clear()
                st.rerun()

        st.markdown("---")
        if st.button("🔄 Reset session"):
            st.session_state.clear()
            st.rerun()


def render_upload():
    """PDF upload + ingest."""
    uploaded = st.file_uploader("📤 Upload a PDF", type=["pdf"])
    if uploaded is not None:
        msg, err = process_upload(uploaded)
        if err:
            st.error(err)
        elif msg:
            st.success(msg)


def render_chunks(chunks: List[Dict], title: str):
    """Collapsible view of retrieved chunks."""
    with st.expander(f"📚 {title} ({len(chunks)})", expanded=False):
        for c in chunks:
            score = c.get("rerank_score", c.get("similarity", 0))
            st.markdown(
                f"**Page {c['page']} · Chunk {c['chunk_id']}** "
                f"`score={score:.3f}`"
            )
            st.write(c["text"])
            st.divider()


def render_qa():
    """Question input + answer pipeline."""
    question = st.text_input(
        "❓ Ask a question about the document",
        placeholder="e.g., What are the key findings?",
    )
    if not question:
        return

    with st.spinner("Retrieving & generating..."):
        result = run_query(question)

    # Answer
    st.subheader("🤖 Answer")
    if result.get("from_cache"):
        st.caption("⚡ Answered from semantic cache (no LLM call)")
    if result.get("blocked"):
        st.warning(result["answer"])
        return
    st.write(result["answer"])

    # Redaction notice (output guardrail)
    if result.get("redaction"):
        st.info("🛡️ Sensitive content (e.g., emails/phones) was redacted from this answer.")

    # Resilience errors
    if result.get("errors"):
        st.caption("🔄 " + "; ".join(result["errors"]))

    # Grounding badge
    g = check_grounding(result["answer"], result["context_chunks"])
    if g["grounded"]:
        st.success(f"🛡️ {g['reason']}")
    else:
        st.warning(f"⚠️ {g['reason']}")

    # Provider badge
    if result["provider"] != "rule-based":
        st.caption(
            f"⚡ Provider: **{result['provider']}** · Model: `{result['model']}`"
        )

    # Metrics
    with st.expander("📊 Evaluation Metrics"):
        c1, c2 = st.columns(2)
        c1.metric("Faithfulness",
                  f"{compute_faithfulness(result['answer'], result['context_chunks']):.2f}")
        c2.metric("Relevance",
                  f"{compute_relevance(question, result['chunks']):.2f}")

    # Retrieved chunks (children used for retrieval)
    if result["chunks"]:
        render_chunks(result["chunks"], "Retrieved Context Chunks")
    if result["context_chunks"]:
        render_chunks(result["context_chunks"], "Context Sent to LLM (parents)")

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"][:300],
    })


def render_history():
    if not st.session_state.history:
        return
    with st.expander(f"🕒 History ({len(st.session_state.history)})", expanded=False):
        for i, h in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"**Q{i}:** {h['question']}")
            st.markdown(f"**A{i}:** {h['answer']}")
            st.divider()


def main():
    init_state()
    render_sidebar()

    st.title("📄 Intelligent Document Q&A")
    st.markdown(
        "Upload a PDF, ask anything — answers are **grounded strictly in the document** "
        "with `[Page X, Chunk Y]` citations. No LLM provider? Set a free Groq key in the sidebar."
    )

    render_upload()

    if st.session_state.is_indexed:
        st.markdown("---")
        render_qa()

    render_history()

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#888;font-size:12px'>"
        "pypdf · Parent-Child Chunking · MiniLM · ChromaDB · BM25+RRF · "
        "Cross-Encoder · Groq/OpenAI/Anthropic/Ollama · Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()