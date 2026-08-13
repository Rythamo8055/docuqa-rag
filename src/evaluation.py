"""
RAG Evaluation Utilities.

Provides lightweight evaluation metrics for RAG systems including
faithfulness scoring.
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_faithfulness(answer: str, context_chunks: List[Dict]) -> float:
    """
    Compute a simple faithfulness score for an answer.

    Faithfulness measures whether the answer can be justified by the
    provided context. A higher score indicates the answer aligns better
    with the source documents.

    This is a lightweight implementation that checks:
    - If answer contains phrases not found in context (lower score)
    - If answer is "Information not found" (perfect if context insufficient)

    Args:
        answer: The generated answer
        context_chunks: List of retrieved context chunks

    Returns:
        Faithfulness score between 0.0 and 1.0
    """
    if not context_chunks:
        if "not found" in answer.lower():
            return 1.0
        return 0.0

    # Combine all context text
    context_text = " ".join(chunk["text"] for chunk in context_chunks)

    # Check for the "not found" response
    not_found_responses = [
        "information not found in the provided document",
        "not found in the provided document",
    ]
    if any(resp in answer.lower() for resp in not_found_responses):
        # If the answer says "not found", check if context is truly insufficient
        # Simple heuristic: if context is empty or answer is short
        return 1.0 if not context_text.strip() else 0.8

    # Simple n-gram overlap heuristic
    context_words = set(context_text.lower().split())
    answer_words = set(answer.lower().split())

    # Calculate word overlap ratio
    if not answer_words:
        return 0.0

    overlap = len(answer_words & context_words)
    coverage = overlap / len(answer_words)

    # Weight based on context length (longer context should allow more words)
    length_factor = min(len(context_words) / 100, 1.0)

    score = coverage * (0.5 + 0.5 * length_factor)
    return round(score, 4)


def compute_relevance(query: str, context_chunks: List[Dict]) -> float:
    """
    Compute a simple relevance score between query and context.

    Uses term frequency-inverse document frequency (TF-IDF) style weighting
    to estimate how relevant the retrieved chunks are to the query.

    Args:
        query: The user's query
        context_chunks: List of retrieved context chunks

    Returns:
        Relevance score between 0.0 and 1.0
    """
    query_words = set(query.lower().split())
    if not query_words or not context_chunks:
        return 0.0

    max_relevance = 0.0
    for chunk in context_chunks:
        context_words = set(chunk["text"].lower().split())
        overlap = len(query_words & context_words)
        relevance = overlap / len(query_words)
        max_relevance = max(max_relevance, relevance)

    return round(max_relevance, 4)


def generate_ragas_style_metrics(
    questions: List[str],
    answers: List[str],
    contexts: List[List[Dict]],
) -> Dict[str, float]:
    """
    Generate lightweight RAG evaluation metrics similar to Ragas.

    Args:
        questions: List of user queries
        answers: List of generated answers
        contexts: List of context chunks lists (one per question)

    Returns:
        Dictionary with average faithfulness and relevance scores
    """
    if not questions:
        return {"faithfulness": 0.0, "relevance": 0.0}

    faithfulness_scores = []
    relevance_scores = []

    for q, a, ctx in zip(questions, answers, contexts):
        faithfulness_scores.append(compute_faithfulness(a, ctx))
        relevance_scores.append(compute_relevance(q, ctx))

    return {
        "faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 4),
        "relevance": round(sum(relevance_scores) / len(relevance_scores), 4),
    }
