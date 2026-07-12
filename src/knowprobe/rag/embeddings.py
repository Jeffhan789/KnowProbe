"""Embedding providers for RAG pipeline."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def encode(self, texts: list[str], **kwargs: Any) -> FloatArray:
        """Encode a list of texts into dense vectors."""
        ...

    def encode_single(self, text: str, **kwargs: Any) -> FloatArray:
        """Encode a single text into a dense vector."""
        result = self.encode([text], **kwargs)
        return result[0]


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Sentence-Transformer based embedding provider."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._dimension: int | None = None
        logger.info("embeddings.init", model=model_name)

    def _load_model(self) -> Any:
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(
                    "embeddings.model_loaded",
                    model=self.model_name,
                    dimension=self._dimension,
                )
            except ImportError as e:
                logger.error("embeddings.import_error", error=str(e))
                raise RuntimeError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                ) from e
            except Exception as e:
                logger.error("embeddings.load_error", error=str(e))
                raise
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def encode(self, texts: list[str], **kwargs: Any) -> FloatArray:
        model = self._load_model()
        if not texts:
            return np.array([], dtype=np.float64).reshape(0, self.dimension)

        try:
            # Filter out empty strings
            valid_texts = [t if t.strip() else " " for t in texts]
            embeddings = model.encode(
                valid_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float64)
        except Exception as e:
            logger.error("embeddings.encode_error", num_texts=len(texts), error=str(e))
            raise


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, seed: int = 42) -> None:
        self._dimension = dimension
        self._rng = np.random.default_rng(seed)
        logger.info("embeddings.mock_init", dimension=dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str], **kwargs: Any) -> FloatArray:
        if not texts:
            return np.array([], dtype=np.float64).reshape(0, self._dimension)
        embeddings = self._rng.standard_normal((len(texts), self._dimension))
        # Normalize to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
