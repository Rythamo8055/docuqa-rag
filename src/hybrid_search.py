"""
Bonus: Hybrid Search with BM25 + Vector (Reciprocal Rank Fusion).

Combines dense vector search with sparse keyword matching (BM25) using
Reciprocal Rank Fusion (RRF) to improve retrieval accuracy.
"""
from typing import List, Dict, Tuple
import logging

from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60  # RRF constant (standard value)


class HybridRetriever:
    """
    Combines vector similarity search with BM25 keyword search using
    Reciprocal Rank Fusion for improved retrieval accuracy.
    """

    def __init__(self, vector_store, rrf_k: int = DEFAULT_RRF_K):
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self._bm25 = None
        self._corpus = None

    def index_bm25(self, chunks: List[Dict]) -> None:
        """
        Build a BM25 index from the chunk texts.

        Args:
            chunks: List of chunk dictionaries with 'text' key
        """
        corpus = [chunk["text"] for chunk in chunks]
        # Simple tokenization - can be enhanced with better NLP tokenization
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self._corpus = corpus
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 index for {len(corpus)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> List[Dict]:
        """
        Retrieve chunks using hybrid search (BM25 + vector) with RRF fusion.

        Args:
            query: User's search query
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with fused similarity scores
        """
        # Get vector search results
        vector_results = self.vector_store.retrieve(
            query, top_k=top_k * 2  # Get more for fusion
        )

        # Get BM25 results
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)

            # Create BM25 ranked list
            bm25_ranked = sorted(
                enumerate(bm25_scores), key=lambda x: x[1], reverse=True
            )[:top_k * 2]
        else:
            bm25_ranked = []

        # Apply RRF fusion
        fused_scores = self._reciprocal_rank_fusion(
            vector_results, bm25_ranked
        )

        # Sort by fused score and return top_k
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [vector_results[i] for i, _ in fused_scores[:top_k]]

        return top_chunks

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_ranked: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """
        Combine vector and BM25 results using Reciprocal Rank Fusion.

        RRF score = sum( k / (rank_position + k) ) for each matching result

        Args:
            vector_results: List of chunks from vector search
            bm25_ranked: List of (corpus_index, score) tuples from BM25

        Returns:
            List of (corpus_index, fused_score) tuples sorted by fused score
        """
        scores = {}

        # Vector search rankings
        for rank, chunk in enumerate(vector_results):
            text = chunk["text"]
            if text in self._corpus:
                idx = self._corpus.index(text)
                scores[idx] = scores.get(idx, 0) + self.rrf_k / (rank + 1 + self.rrf_k)

        # BM25 rankings
        for rank, (idx, _) in enumerate(bm25_ranked):
            scores[idx] = scores.get(idx, 0) + self.rrf_k / (rank + 1 + self.rrf_k)

        # Sort by score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
