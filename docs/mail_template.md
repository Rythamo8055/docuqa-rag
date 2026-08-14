# Mail Template — RAG Document Q&A Assignment Submission

---

**Subject:** AI Engineer Intern — RAG Document Q&A System Submission

---

Hi [Reviewer / Hiring Manager],

I've completed the **Intelligent Document Q&A System** assignment using Retrieval-Augmented Generation (RAG). Below is a summary of the project, test results, and live demo links.

---

## Project Overview

- **Repo:** [github.com/Rythamo8055/docuqa-rag](https://github.com/Rythamo8055/docuqa-rag)
- **Live Demo:** [nexaraium.streamlit.app](https://nexaraium.streamlit.app)
- **Stack:** Python 3.14 · Streamlit · MiniLM-L6-v2 · ChromaDB · BM25+RRF · Cross-Encoder Reranker · Multi-provider LLM Router

---

## What It Does

Upload a PDF → ask questions → get grounded answers with `[Page X, Chunk Y]` citations. The system includes:

- **Hybrid Retrieval:** BM25 + vector search with Reciprocal Rank Fusion
- **Cross-Encoder Reranking:** Precision boost on retrieved chunks
- **Extractive Fallback:** When the LLM refuses or hallucinates, a deterministic extraction provides verbatim answers with citations
- **Input Guardrails:** Prompt injection detection, rate limiting, input validation
- **Output Guardrails:** PII redaction, leakage detection, unsafe content filtering
- **Circuit Breaker:** Graceful degradation when LLM providers fail
- **Semantic Cache:** ~80-90% cost reduction on repeated questions

---

## Test Results

### Self-Tests (54/54 passed)

| Module | Tests | Result |
|--------|-------|--------|
| `resilience.py` | 13 (retry, circuit breaker, fallbacks, ErrorBucket) | 13/13 ✅ |
| `input_guardrails.py` | 15 (sanitize, injection detect, validate, rate-limit) | 15/15 ✅ |
| `output_guardrails.py` | 14 (PII redact, leakage detect, unsafe filter) | 14/14 ✅ |
| `upload_security.py` | 12 (PDF validate, sanitize, temp file) | 12/12 ✅ |

### Eval Harness (10/10 passed)

| Metric | Result |
|--------|--------|
| Pass rate | 10/10 (100%) |
| Retrieval hit-rate | 1.0 |
| Context precision (Ragas) | 0.783 |
| Faithfulness (avg) | 0.791 |
| Relevance (avg) | 0.638 |
| Injection blocked | ✅ |

### Stress Test (27/28 passed)

| Metric | Result |
|--------|--------|
| Pass rate | 27/28 (96%) |
| Retrieval hit-rate | 1.0 |
| Context precision (Ragas) | 0.982 |
| Grounded answers | 93% |
| Latency p95 | 12.4s |
| Fabrication errors | 0 |

---

## Rubric Compliance

| Criterion | Weight | Status |
|-----------|--------|--------|
| RAG quality & grounding | 30% | ✅ hybrid+rerank, grounding check, extractive fallback, 10/10 eval |
| Deployment & live demo | 25% | ✅ Streamlit Cloud live at nexaraium.streamlit.app |
| Code architecture & git | 25% | ✅ modular src/, typed, clean commit history |
| Documentation & README | 20% | ✅ mermaid diagrams, test results, eval reports |

---

## How to Run Locally

```bash
git clone https://github.com/Rythamo8055/docuqa-rag.git
cd docuqa-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add one API key (GROQ/GEMINI/OPENAI/ANTHROPIC)
streamlit run app.py        # → http://localhost:8501
```

---

## Key Design Decisions

1. **Parent-child chunking** (800/300 tokens, 15% overlap) — precise retrieval + rich LLM context
2. **Hybrid BM25 + vector with RRF** — catches both semantic and keyword matches
3. **Extractive fallback** — deterministic, hallucination-proof when LLM refuses
4. **Circuit breaker + rate limiter** — graceful degradation under load
5. **Zero-cost first** — free-tier LLM providers (Groq, Gemini), local embeddings, no paid dependencies

---

Please let me know if you have any questions or need additional details.

Best regards,
Rahul

---

*Generated: August 2026*
