"""RAG Pipeline evaluation package for KnowProbe."""

from knowprobe.rag.document_processor import (
    ChunkingStrategy,
    DocumentProcessor,
    FixedSizeChunking,
)
from knowprobe.rag.embeddings import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    SentenceTransformerEmbeddings,
)
from knowprobe.rag.pipeline import PipelineBuilder, RAGPipeline
from knowprobe.rag.rag_evaluator import RAGEvaluator, RAGMetricCalculator
from knowprobe.rag.rag_generator import (
    GenerationBackend,
    OllamaBackend,
    RAGGenerator,
    RAGPromptBuilder,
    TransformersBackend,
)
from knowprobe.rag.retriever import BaseRetriever, DenseRetriever, HybridRetriever
from knowprobe.rag.vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    # Document processing
    "ChunkingStrategy",
    "FixedSizeChunking",
    "DocumentProcessor",
    # Embeddings
    "EmbeddingProvider",
    "SentenceTransformerEmbeddings",
    "MockEmbeddingProvider",
    # Vector store
    "VectorStore",
    "InMemoryVectorStore",
    # Retriever
    "BaseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    # Generator
    "GenerationBackend",
    "OllamaBackend",
    "TransformersBackend",
    "RAGPromptBuilder",
    "RAGGenerator",
    # Evaluator
    "RAGMetricCalculator",
    "RAGEvaluator",
    # Pipeline
    "RAGPipeline",
    "PipelineBuilder",
]
