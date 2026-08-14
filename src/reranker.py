"""
Cross-Encoder Reranker.

Reranks retrieved chunks using a cross-encoder model — the single biggest
retrieval-accuracy boost per dollar. Cross-encoders jointly attend to
(query, chunk) pairs, giving far more precise relevance scores than
bi-encoder cosine similarity alone.
"""
from typing import List, Dict, Optional
import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Small, fast, free, local — ~90MB. Swap for a stronger model if needed.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class CrossEncoderReranker:
    """Reranks candidate chunks against the query with a cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder (only downloaded when first used)."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Score each (query, chunk) pair and return the top_k chunks.

        Args:
            query: User query
            chunks: Candidate chunks from hybrid retrieval
            top_k: Number of chunks to keep

        Returns:
            Chunks sorted by rerank score (desc), each with 'rerank_score'.
        """
        if not chunks:
            return []

        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = [
            {**chunk, "rerank_score": round(float(score), 4)}
            for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)

        logger.info(
            f"Reranked {len(chunks)} chunks → kept top {min(top_k, len(scored))}"
        )
        return scored[:top_k]