"""
Intelligent Document Q&A System with Retrieval-Augmented Generation (RAG).

A Streamlit web application that:
1. Accepts PDF uploads
2. Processes and chunks documents
3. Generates semantic embeddings
4. Indexes into a vector database (ChromaDB)
5. Provides grounded answers with explicit citations

Usage:
    streamlit run app.py
"""
from typing import List, Dict
import logging
import os
import tempfile

import streamlit as st

from src.pdf_utils import process_pdf, count_tokens
from src.embeddings import (
    EmbeddingManager,
    ChromaVectorStore,
)
from src.llm import GroundedGenerator
from src.hybrid_search import HybridRetriever
from src.evaluation import compute_faithfulness, compute_relevance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App configuration
st.set_page_config(
    page_title="DocuQA - RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
TOP_K = 4
MAX_QUERY_HISTORY = 10


def initialize_session_state():
    """Initialize session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    if "generator" not in st.session_state:
        st.session_state.generator = GroundedGenerator(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    if "hybrid_retriever" not in st.session_state:
        st.session_state.hybrid_retriever = None
    if "is_indexed" not in st.session_state:
        st.session_state.is_indexed = False
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def display_citation_legend():
    """Display citation format legend."""
    with st.expander("📖 Citation Legend", expanded=False):
        st.markdown("""
        **Citation Format:** `[Page X, Chunk Y]`
        - **Page X**: The page number in the original PDF (1-indexed)
        - **Chunk Y**: The semantic chunk number within the entire document

        Answers are grounded only on the provided document. If information cannot be
        derived from the document, the system responds with:
        *"Information not found in the provided document."*
        """)


def handle_pdf_upload():
    """Handle PDF file upload and processing."""
    uploaded_file = st.file_uploader(
        "📤 Upload a PDF document",
        type=["pdf"],
        help="Supports multi-page PDFs, research papers, technical manuals, etc.",
    )

    if uploaded_file is None:
        return

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Processing PDF..."):
            # Extract text and chunk
            st.session_state.chunks = process_pdf(tmp_path)

            if not st.session_state.chunks:
                st.error("❌ No text could be extracted from the PDF.")
                return

            # Initialize vector store
            st.session_state.vector_store = ChromaVectorStore(
                persist_dir="./chroma_db",
                embedding_manager=st.session_state.embedding_manager,
            )

            # Index chunks
            with st.spinner("Generating embeddings and indexing..."):
                st.session_state.vector_store.add_chunks(st.session_state.chunks)
                st.session_state.is_indexed = True

            # Build hybrid retriever
            st.session_state.hybrid_retriever = HybridRetriever(
                vector_store=st.session_state.vector_store
            )
            st.session_state.hybrid_retriever.index_bm25(st.session_state.chunks)

            # Display summary
            total_tokens = sum(c["tokens"] for c in st.session_state.chunks)
            st.success(
                f"✅ **Document processed successfully!**\n\n"
                f"- Total pages: {len({c['page'] for c in st.session_state.chunks})}\n"
                f"- Total chunks: {len(st.session_state.chunks)}\n"
                f"- Total tokens: {total_tokens:,}"
            )

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def display_retrieved_context(chunks: List[Dict]):
    """Display retrieved context chunks in an expandable view."""
    with st.expander(f"📚 Retrieved Context Chunks ({len(chunks)})", expanded=False):
        for chunk in chunks:
            with st.container():
                st.markdown(f"**Page {chunk['page']}, Chunk {chunk['chunk_id']}**")
                st.markdown(f"*Relevance: {chunk.get('similarity', 0):.2f}*")
                # Show excerpt in expander
                with st.expander("Show text"):
                    st.write(chunk["text"])
            st.divider()


def handle_query():
    """Handle user query and generate response."""
    question = st.text_input(
        "❓ Ask a question about the document",
        placeholder="e.g., What are the key findings? What does the system architecture look like?",
        key="question_input",
    )

    if not question:
        return

    if not st.session_state.is_indexed:
        st.warning("Please upload a PDF document first.")
        return

    # Retrieve relevant chunks (using hybrid search if available)
    with st.spinner("Retrieving relevant content..."):
        if st.session_state.hybrid_retriever:
            retrieved_chunks = st.session_state.hybrid_retriever.retrieve(
                question, top_k=TOP_K
            )
        else:
            retrieved_chunks = st.session_state.vector_store.retrieve(
                question, top_k=TOP_K
            )

    # Display retrieved context
    display_retrieved_context(retrieved_chunks)

    # Generate grounded answer
    with st.spinner("Generating answer..."):
        response = st.session_state.generator.generate_response(
            question=question,
            chunks=retrieved_chunks,
            stream=True,
        )

    # Handle streaming or non-streaming response
    if hasattr(response, "__iter__"):
        # Streaming response
        answer_placeholder = st.empty()
        full_answer = ""
        for chunk in response:
            if hasattr(chunk, "choices"):
                # OpenAI streaming response
                content = chunk.choices[0].delta.content or ""
            else:
                # Other response formats
                content = str(chunk)
            full_answer += content
            answer_placeholder.markdown(full_answer)
        answer = full_answer
    else:
        answer = response

    # Display answer
    st.subheader("🤖 Answer")
    st.markdown(f"**{answer}**")

    # Compute and display evaluation metrics
    with st.expander("📊 Evaluation Metrics"):
        faithfulness = compute_faithfulness(answer, retrieved_chunks)
        relevance = compute_relevance(question, retrieved_chunks)
        st.metric("Faithfulness", f"{faithfulness:.2f}")
        st.metric("Relevance", f"{relevance:.2f}")

    # Save to query history
    st.session_state.query_history.append({
        "question": question,
        "answer": answer,
        "retrieved_chunks": len(retrieved_chunks),
    })


def display_query_history():
    """Display the query history."""
    history = st.session_state.get("query_history", [])
    if not history:
        return

    with st.expander(f"🕒 Query History ({len(history)})", expanded=False):
        for i, entry in enumerate(reversed(history[-5:]), 1):
            with st.container():
                st.markdown(f"**Q{i}:** {entry['question']}")
                st.markdown(f"**A{i}:** {entry['answer'][:200]}...")
                st.caption(f"Retrieved: {entry['retrieved_chunks']} chunks")
            st.divider()


def main():
    """Main application entry point."""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("📚 DocuQA Settings")
        st.markdown("---")

        # API Key configuration
        if not os.getenv("OPENAI_API_KEY"):
            with st.form("api_settings"):
                st.subheader("🔑 OpenAI API Key")
                api_key = st.text_input(
                    "Enter OpenAI API Key (optional - enables GPT models)",
                    type="password",
                )
                submit_key = st.form_submit_button("Set Key")
                if submit_key and api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.session_state.generator = GroundedGenerator(api_key=api_key)
                    st.success("✅ API key set!")

        st.markdown("---")
        display_citation_legend()

        # Reset button
        if st.button("🔄 Reset Session", type="secondary"):
            st.session_state.clear()
            st.rerun()

    # Main content
    st.title("📄 Intelligent Document Q&A")
    st.markdown("""
    Upload a PDF document and ask questions about it. The system uses **Retrieval-Augmented 
    Generation (RAG)** to provide answers grounded strictly in your document.

    **Features:**
    - ✅ PDF processing with semantic chunking
    - ✅ Vector embeddings with cosine similarity search
    - ✅ Hybrid search (BM25 + dense retrieval)
    - ✅ Grounded generation with explicit citations
    - ✅ Token streaming responses
    - ✅ Hallucination prevention
    """)

    # PDF upload section
    handle_pdf_upload()

    # Query section
    if st.session_state.is_indexed:
        st.markdown("---")
        handle_query()

    # Query history
    display_query_history()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Built with Streamlit • LangChain • ChromaDB • Sentence-Transformers"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
