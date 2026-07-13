"""RAG Pipeline orchestration and execution."""

import time
import uuid
from typing import Any

from knowprobe.core.models import (
    RAGBenchmarkResult,
    RAGDocument,
    RAGPipelineResult,
    RAGQuery,
)
from knowprobe.rag.embeddings import SentenceTransformerEmbeddings
from knowprobe.rag.rag_evaluator import RAGEvaluator
from knowprobe.rag.rag_generator import RAGGenerator
from knowprobe.rag.retriever import BaseRetriever, DenseRetriever, HybridRetriever
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Orchestrate the full RAG pipeline: retrieval + generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: RAGGenerator,
        evaluator: RAGEvaluator | None = None,
        name: str = "default",
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator or RAGEvaluator()
        self.name = name
        logger.info("pipeline.init", name=name)

    def index(self, documents: list[RAGDocument]) -> None:
        """Index documents for retrieval."""
        if not documents:
            logger.warning("pipeline.empty_index")
            return
        self.retriever.index_documents(documents)
        logger.info("pipeline.indexed", num_docs=len(documents))

    def run(self, query: RAGQuery, top_k: int = 5) -> RAGPipelineResult:
        """Run the full RAG pipeline on a single query."""
        start_time = time.perf_counter()

        try:
            # Retrieval
            retrieval_results = self.retriever.retrieve(query.query_text, top_k=top_k)

            # Generation
            answer, raw_prompt, gen_latency = self.generator.generate(
                query, [r.chunk for r in retrieval_results]
            )

            total_latency = (time.perf_counter() - start_time) * 1000

            result = RAGPipelineResult(
                query=query,
                retrieval_results=retrieval_results,
                generated_answer=answer,
                raw_prompt=raw_prompt,
                latency_ms=total_latency,
            )

            logger.info(
                "pipeline.run_complete",
                query_id=query.query_id,
                latency_ms=total_latency,
                retrieved=len(retrieval_results),
            )
            return result

        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                "pipeline.run_error",
                query_id=query.query_id,
                error=str(e),
                latency_ms=latency,
            )
            raise

    def run_and_evaluate(self, query: RAGQuery, top_k: int = 5) -> RAGPipelineResult:
        """Run pipeline and evaluate the result."""
        result = self.run(query, top_k=top_k)
        metrics = self.evaluator.evaluate(result, query=query)
        result.metrics = metrics
        return result

    def run_batch(self, queries: list[RAGQuery], top_k: int = 5) -> list[RAGPipelineResult]:
        """Run the pipeline on a batch of queries."""
        results: list[RAGPipelineResult] = []
        for query in queries:
            try:
                result = self.run(query, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error(
                    "pipeline.batch_error",
                    query_id=query.query_id,
                    error=str(e),
                )
        return results

    def benchmark(
        self,
        queries: list[RAGQuery],
        top_k: int = 5,
        benchmark_id: str | None = None,
    ) -> RAGBenchmarkResult:
        """Run a full benchmark evaluation on a set of queries."""
        benchmark_id = benchmark_id or f"bench_{uuid.uuid4().hex[:8]}"
        logger.info("pipeline.benchmark_start", benchmark_id=benchmark_id, num_queries=len(queries))

        # Run all queries
        results = self.run_batch(queries, top_k=top_k)

        # Evaluate
        eval_result = self.evaluator.evaluate_batch(results, queries=queries)
        metrics = eval_result["metrics"]
        aggregate = eval_result["aggregate"]

        # Build benchmark result
        benchmark = RAGBenchmarkResult(
            benchmark_id=benchmark_id,
            pipeline_name=self.name,
            num_queries=len(queries),
            metrics=metrics,
            aggregate_scores=aggregate,
            per_query_results=results,
        )

        logger.info(
            "pipeline.benchmark_complete",
            benchmark_id=benchmark_id,
            avg_latency=aggregate.get("avg_latency_ms", 0.0),
        )
        return benchmark


class PipelineBuilder:
    """Builder for constructing RAG pipelines with sensible defaults."""

    @staticmethod
    def from_config(config: dict[str, Any]) -> RAGPipeline:
        """Build a RAG pipeline from configuration dictionary."""
        rag_config = config.get("rag", {})
        embedding_model = rag_config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        chunk_size = rag_config.get("chunk_size", 512)
        chunk_overlap = rag_config.get("chunk_overlap", 50)
        retriever_type = rag_config.get("retriever", "dense")

        # Create embedding provider
        embedding_provider = SentenceTransformerEmbeddings(model_name=embedding_model)

        # Create retriever
        retriever: BaseRetriever
        if retriever_type == "hybrid":
            retriever = HybridRetriever(
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            retriever = DenseRetriever(
                embedding_provider=embedding_provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        # Create generator (no backend by default - must be configured)
        generator = RAGGenerator()

        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            name=retriever_type,
        )

    @staticmethod
    def dense_pipeline(
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> RAGPipeline:
        """Create a dense retrieval pipeline with sensible defaults."""
        embedding_provider = SentenceTransformerEmbeddings(model_name=embedding_model)
        retriever = DenseRetriever(
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        generator = RAGGenerator()
        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            name="dense",
        )

    @staticmethod
    def hybrid_pipeline(
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        dense_weight: float = 0.7,
    ) -> RAGPipeline:
        """Create a hybrid retrieval pipeline with sensible defaults."""
        embedding_provider = SentenceTransformerEmbeddings(model_name=embedding_model)
        retriever = HybridRetriever(
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            dense_weight=dense_weight,
            sparse_weight=1.0 - dense_weight,
        )
        generator = RAGGenerator()
        return RAGPipeline(
            retriever=retriever,
            generator=generator,
            name="hybrid",
        )
