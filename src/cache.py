"""
Semantic Cache.

Embedding-based caching: identical or near-identical questions hit the cache
instead of the LLM API — cutting repeat-query cost by ~80-90% and making
the UI feel instant. Persisted in SQLite (zero extra dependencies).

Mechanism: store query embedding + answer. On new query, compare embeddings
via cosine similarity; if >= threshold, return the cached answer.
"""
from typing import Dict, Optional
import sqlite3
import time
import os
import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = ".cache/rag_cache.sqlite"
DEFAULT_THRESHOLD = 0.92  # cosine similarity for "same question"


class SemanticCache:
    """SQLite-backed semantic cache keyed by query embeddings."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, threshold: float = DEFAULT_THRESHOLD):
        self.db_path = db_path
        self.threshold = threshold
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                query    TEXT PRIMARY KEY,
                qvec     BLOB NOT NULL,
                answer   TEXT NOT NULL,
                provider TEXT,
                ts       REAL
            )"""
        )
        self.conn.commit()

    def get(self, query: str, query_vec: np.ndarray) -> Dict:
        """
        Look up a cached answer.

        Returns:
            {"hit": True, "answer": str, "provider": str, "similarity": float}
            or {"hit": False}
        """
        rows = self.conn.execute(
            "SELECT qvec, answer, provider FROM cache"
        ).fetchall()
        if not rows:
            return {"hit": False}

        qv = np.asarray(query_vec, dtype=np.float32)
        best_sim = 0.0
        best: Optional[tuple] = None

        for blob, answer, provider in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            denom = np.linalg.norm(qv) * np.linalg.norm(vec) + 1e-9
            sim = float(np.dot(qv, vec) / denom)
            if sim > best_sim:
                best_sim = sim
                best = (answer, provider)

        if best is not None and best_sim >= self.threshold:
            logger.info(f"Semantic cache HIT (sim={best_sim:.3f})")
            return {
                "hit": True,
                "answer": best[0],
                "provider": best[1] or "cache",
                "similarity": round(best_sim, 4),
            }
        return {"hit": False}

    def put(self, query: str, query_vec: np.ndarray, answer: str, provider: str) -> None:
        """Store a query-answer pair in the cache."""
        blob = np.asarray(query_vec, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (query, qvec, answer, provider, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (query, blob, answer, provider, time.time()),
        )
        self.conn.commit()

    def size(self) -> int:
        """Number of cached entries."""
        return self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def clear(self) -> None:
        """Clear the cache."""
        self.conn.execute("DELETE FROM cache")
        self.conn.commit()