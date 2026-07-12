"""Document processing and chunking for RAG pipeline."""

import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

from knowprobe.core.models import RAGChunk, RAGDocument
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base for document chunking strategies."""

    @abstractmethod
    def chunk(self, document: RAGDocument, **kwargs: Any) -> list[RAGChunk]:
        """Split a document into chunks."""
        ...


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with configurable overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap}"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "chunking.init",
            strategy="fixed_size",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: RAGDocument, **kwargs: Any) -> list[RAGChunk]:
        text = document.content
        if not text.strip():
            logger.warning("chunking.empty_document", doc_id=document.doc_id)
            return []

        chunks: list[RAGChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Try to break at sentence boundary if possible
            if end < len(text):
                chunk_text = self._break_at_boundary(chunk_text)

            chunk_id = f"{document.doc_id}_chunk_{idx}_{uuid.uuid4().hex[:8]}"
            chunks.append(
                RAGChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=chunk_text.strip(),
                    chunk_index=idx,
                    metadata={
                        **document.metadata,
                        "char_start": start,
                        "char_end": start + len(chunk_text),
                    },
                )
            )
            start += step
            idx += 1

        logger.info(
            "chunking.complete",
            doc_id=document.doc_id,
            num_chunks=len(chunks),
        )
        return chunks

    def _break_at_boundary(self, text: str) -> str:
        """Attempt to break at sentence or word boundary."""
        # Try sentence boundary
        sentence_match = re.search(r"[.!?。！？]\s+", text[self.chunk_size // 2 :])
        if sentence_match:
            return text[: self.chunk_size // 2 + sentence_match.end()]

        # Fallback to word boundary
        word_match = re.search(r"\s+", text[self.chunk_size // 2 :])
        if word_match:
            return text[: self.chunk_size // 2 + word_match.start()]

        return text


class DocumentProcessor:
    """Process RAG documents into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        self.chunking_strategy = chunking_strategy or FixedSizeChunking(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.info("processor.init")

    def process(self, documents: list[RAGDocument]) -> list[RAGChunk]:
        """Process a list of documents into chunks."""
        all_chunks: list[RAGChunk] = []
        for doc in documents:
            try:
                chunks = self.chunking_strategy.chunk(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(
                    "processor.chunk_error",
                    doc_id=doc.doc_id,
                    error=str(e),
                    exc_info=True,
                )
        logger.info("processor.complete", total_chunks=len(all_chunks))
        return all_chunks
