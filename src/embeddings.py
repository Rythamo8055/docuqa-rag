"""
Embeddings and Vector Storage Module.

Handles dense vector embeddings using sentence-transformers and storage
in ChromaDB with cosine similarity retrieval.
"""
from typing import List, Dict, Tuple
import logging

from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

logger = logging.getLogger(__name__)

# Using all-MiniLM-L6-v2 as specified in the assessment
# (alternative: OpenAI's text-embedding-3-small)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "doc_qa"
DEFAULT_TOP_K = 4
CHROMA_PERSIST_DIR = "./chroma_db"


class EmbeddingManager:
    """
    Manages sentence-transformer embeddings with lazy loading.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a list of texts.
        Normalization enables efficient cosine similarity via dot product.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of shape (len(texts), embedding_dim), L2-normalized
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([text])[0]


class ChromaVectorStore:
    """
    Wrapper around ChromaDB for storing and querying document embeddings.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_manager: EmbeddingManager = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Add text chunks to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'text', 'page', 'chunk_id' keys
        """
        # Clean collection if it already exists with data
        # (for simplicity in this assessment app)
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                self.collection.delete(where={})
        except Exception:
            pass  # Collection might not have data yet

        # Prepare data for ChromaDB
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk.get("parent_id", 0),
                "tokens": chunk.get("tokens", 0),
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB")

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query using cosine similarity.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            List of retrieved chunk dictionaries with similarity scores
        """
        query_embedding = self.embedding_manager.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert cosine distance to similarity score
            similarity = 1 - dist if dist <= 2 else 0
            retrieved_chunks.append({
                "text": doc,
                "page": meta["page"],
                "chunk_id": meta["chunk_id"],
                "parent_id": meta.get("parent_id", 0),
                "tokens": meta.get("tokens", 0),
                "similarity": round(similarity, 4),
                "distance": round(dist, 4),
            })

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
        return retrieved_chunks

    def clear(self) -> None:
        """Remove all documents from the collection."""
        try:
            self.collection.delete(where={})
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.debug(f"Nothing to clear or error: {e}")

    def load_all(self) -> List[Dict]:
        """
        Rehydrate every stored chunk (text + metadata) from disk.

        Used to rebuild in-memory structures (BM25 index) after a process
        restart, so a persisted index survives server restarts/deploys.
        Returns an empty list when nothing is stored.
        """
        try:
            count = self.collection.count()
            if count == 0:
                return []
            data = self.collection.get(
                include=["documents", "metadatas"],
                limit=count,
            )
        except Exception as e:
            logger.warning(f"load_all failed: {e}")
            return []

        chunks = []
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            if meta is None:
                continue
            chunks.append({
                "text": doc,
                "page": meta.get("page", 1),
                "chunk_id": meta.get("chunk_id", 0),
                "parent_id": meta.get("parent_id", 0),
                "tokens": meta.get("tokens", 0),
            })
        return chunks
