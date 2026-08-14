"""
Intelligent Document Q&A System with Retrieval-Augmented Generation (RAG).

Pipeline: PDF upload → pypdf extraction → parent-child chunking →
MiniLM embeddings → ChromaDB → hybrid search (BM25+RRF) →
cross-encoder rerank → grounded LLM (router) → cited answer.

Additive optimizations (all optional, all degrade gracefully):
- Semantic cache (SQLite, embedding-based)
- Cross-encoder reranker
- LLM provider router (Groq free → OpenAI → Anthropic → Ollama)
- Langfuse tracing

Usage:
    streamlit run app.py
"""
from typing import List, Dict
import logging
import os
import tempfile

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


def process_upload(uploaded):
    """Extract → chunk → embed → index. Returns summary strings."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        result = process_pdf(tmp_path)
        children, parents = result["children"], result["parents"]
        if not children:
            return None, "No extractable text found in this PDF."

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

    # 1) Semantic cache
    if st.session_state.cache_on:
        hit = cache.get(question, emb)
        if hit["hit"]:
            tracer.trace("cache", query=question, similarity=hit["similarity"])
            return {**hit, "chunks": [], "from_cache": True}

    # 2) Hybrid retrieve (BM25 + dense, RRF)
    candidates = st.session_state.hybrid.retrieve(question, top_k=RETRIEVE_N)

    # 3) Rerank (cross-encoder)
    if st.session_state.rerank_on:
        top = st.session_state.reranker.rerank(question, candidates, top_k=TOP_K)
    else:
        top = candidates[:TOP_K]

    # 4) Small-to-big: resolve parents for generation context
    context_chunks = resolve_parents(top)

    # 5) Grounded generation
    answer, provider, model = st.session_state.generator.generate_response(
        question, context_chunks, stream=False
    )

    tracer.trace(
        "qa",
        query=question,
        candidates=len(candidates),
        reranked=len(top),
        provider=provider,
        model=model,
        grounding=check_grounding(answer, context_chunks),
    )

    # 6) Cache the answer (if enabled and a real LLM answered)
    if st.session_state.cache_on and provider != "rule-based":
        cache.put(question, emb, answer, provider)

    return {
        "answer": answer,
        "provider": provider,
        "model": model,
        "chunks": top,
        "context_chunks": context_chunks,
        "from_cache": False,
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
            keys = {"GROQ_API_KEY": "Groq (free)", "OPENAI_API_KEY": "OpenAI",
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
    st.write(result["answer"])

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