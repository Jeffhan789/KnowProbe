"""Retrieval strategies for RAG pipeline."""

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from numpy.typing import NDArray

from knowprobe.core.models import RAGChunk, RAGDocument, RetrievalResult
from knowprobe.rag.embeddings import EmbeddingProvider
from knowprobe.rag.vector_store import InMemoryVectorStore, VectorStore
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


class BaseRetriever(ABC):
    """Abstract base class for document retrievers."""

    @abstractmethod
    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a query."""
        ...

    @abstractmethod
    def index_documents(self, documents: list[RAGDocument]) -> None:
        """Index a list of documents for retrieval."""
        ...


class DenseRetriever(BaseRetriever):
    """Dense retriever using vector embeddings and cosine similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        from knowprobe.rag.document_processor import DocumentProcessor, FixedSizeChunking

        if embedding_provider is None:
            from knowprobe.rag.embeddings import SentenceTransformerEmbeddings

            embedding_provider = SentenceTransformerEmbeddings()
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store or InMemoryVectorStore()
        self.processor = DocumentProcessor(
            chunking_strategy=FixedSizeChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        logger.info("retriever.dense_init")

    def index_documents(self, documents: list[RAGDocument]) -> None:
        if not documents:
            logger.warning("retriever.empty_index")
            return

        chunks = self.processor.process(documents)
        if not chunks:
            logger.warning("retriever.no_chunks")
            return

        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.encode(texts)
        self.vector_store.add_documents(chunks, embeddings)

        logger.info(
            "retriever.indexed",
            num_docs=len(documents),
            num_chunks=len(chunks),
        )

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        query_embedding = self.embedding_provider.encode_single(query_text)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.info(
            "retriever.dense_search",
            query=query_text[:50],
            top_k=top_k,
            results_found=len(results),
        )
        return results


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse (BM25) retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        if abs(dense_weight + sparse_weight - 1.0) > 1e-6:
            raise ValueError("dense_weight + sparse_weight must equal 1.0")

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        from knowprobe.rag.document_processor import DocumentProcessor, FixedSizeChunking
        from knowprobe.rag.embeddings import SentenceTransformerEmbeddings

        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddings()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.processor = DocumentProcessor(
            chunking_strategy=FixedSizeChunking(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        self._chunks: list[RAGChunk] = []
        self._doc_freq: Counter[str] = Counter()
        self._total_docs: int = 0
        logger.info(
            "retriever.hybrid_init",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def index_documents(self, documents: list[RAGDocument]) -> None:
        if not documents:
            return

        chunks = self.processor.process(documents)
        if not chunks:
            return

        self._chunks = chunks
        self._total_docs = len(documents)

        # Update document frequency for BM25
        for chunk in chunks:
            tokens = set(self._tokenize(chunk.content))
            for token in tokens:
                self._doc_freq[token] += 1

        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.encode(texts)
        self.vector_store.add_documents(chunks, embeddings)

        logger.info(
            "retriever.hybrid_indexed",
            num_docs=len(documents),
            num_chunks=len(chunks),
        )

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        # Dense retrieval
        query_embedding = self.embedding_provider.encode_single(query_text)
        dense_results = self.vector_store.search(query_embedding, top_k=top_k * 2)

        # Sparse retrieval (BM25-like)
        sparse_scores = self._sparse_score(query_text)

        # Combine scores
        combined_scores: dict[str, float] = {}
        for r in dense_results:
            combined_scores[r.chunk.chunk_id] = self.dense_weight * r.score

        for chunk_id, score in sparse_scores.items():
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += self.sparse_weight * score
            else:
                combined_scores[chunk_id] = self.sparse_weight * score

        # Sort and return top-k
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(sorted_scores))

        chunk_map = {c.chunk_id: c for c in self._chunks}
        results: list[RetrievalResult] = []
        for rank, (chunk_id, score) in enumerate(sorted_scores[:top_k], 1):
            if chunk_id in chunk_map:
                results.append(RetrievalResult(chunk=chunk_map[chunk_id], score=score, rank=rank))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for sparse retrieval."""
        import re

        return re.findall(r"\b\w+\b", text.lower())

    def _sparse_score(self, query_text: str) -> dict[str, float]:
        """Calculate BM25-like scores for all chunks."""
        query_tokens = self._tokenize(query_text)
        if not query_tokens or self._total_docs == 0:
            return {}

        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        avgdl = sum(len(c.content) for c in self._chunks) / len(self._chunks) if self._chunks else 0

        for chunk in self._chunks:
            tokens = self._tokenize(chunk.content)
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for token in query_tokens:
                df = self._doc_freq.get(token, 0)
                if df == 0:
                    continue
                idf = np.log((self._total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf_score = tf.get(token, 0)
                denom = tf_score + k1 * (1 - b + b * (dl / avgdl)) if avgdl > 0 else tf_score + k1
                score += idf * (tf_score * (k1 + 1)) / denom
            scores[chunk.chunk_id] = score

        return scores
