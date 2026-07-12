"""Vector store implementations for RAG retrieval."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from knowprobe.core.models import RAGChunk, RetrievalResult
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search."""

    @abstractmethod
    def add_documents(
        self, chunks: list[RAGChunk], embeddings: FloatArray
    ) -> None:
        """Add document chunks with their embeddings to the store."""
        ...

    @abstractmethod
    def search(
        self, query_embedding: FloatArray, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Search for the top-k most similar chunks."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored documents."""
        ...

    @property
    @abstractmethod
    def num_documents(self) -> int:
        """Return the number of stored documents."""
        ...


class InMemoryVectorStore(VectorStore):
    """In-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        self._chunks: list[RAGChunk] = []
        self._embeddings: FloatArray | None = None
        self._dimension: int | None = None
        logger.info("vector_store.init", type="in_memory")

    def add_documents(
        self, chunks: list[RAGChunk], embeddings: FloatArray
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({embeddings.shape[0]})"
            )
        if embeddings.shape[0] == 0:
            logger.warning("vector_store.empty_add")
            return

        self._dimension = embeddings.shape[1]

        if self._embeddings is None:
            self._chunks = list(chunks)
            self._embeddings = embeddings.copy()
        else:
            if embeddings.shape[1] != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected {self._dimension}, got {embeddings.shape[1]}"
                )
            self._chunks.extend(chunks)
            self._embeddings = np.vstack([self._embeddings, embeddings])

        logger.info(
            "vector_store.added",
            num_added=len(chunks),
            total_docs=self.num_documents,
        )

    def search(
        self, query_embedding: FloatArray, top_k: int = 5
    ) -> list[RetrievalResult]:
        if self._embeddings is None or self.num_documents == 0:
            logger.warning("vector_store.empty_search")
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self._dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: "
                f"expected {self._dimension}, got {query_embedding.shape[1]}"
            )

        # Cosine similarity = dot product for normalized vectors
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(
                RetrievalResult(
                    chunk=self._chunks[int(idx)],
                    score=float(similarities[idx]),
                    rank=rank,
                )
            )

        logger.debug(
            "vector_store.search",
            top_k=top_k,
            best_score=float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0,
        )
        return results

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None
        self._dimension = None
        logger.info("vector_store.cleared")

    @property
    def num_documents(self) -> int:
        return len(self._chunks)
