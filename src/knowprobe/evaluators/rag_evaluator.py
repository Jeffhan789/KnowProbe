"""RAG pipeline evaluator for KnowProbe.

Evaluates RAG (Retrieval-Augmented Generation) pipelines across:
- Retrieval metrics: Precision@K, Recall@K, MRR, NDCG, HitRate@K
- Generation metrics: Faithfulness, Answer Relevance, Context Precision
- End-to-end metrics: Latency, Token efficiency

Designed for comparing RAG configurations on knowledge-base QA tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from knowprobe.core.models import RAGDocument, RAGQuery, RAGResult
from knowprobe.utils.logging import get_logger

from .metrics import GrammarMetric, MetricRegistry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RAG evaluation data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalMetrics:
    """Retrieval quality metrics for a single query."""

    query_id: str
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    num_retrieved: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "latency_ms": self.latency_ms,
            "num_retrieved": self.num_retrieved,
        }


@dataclass(frozen=True)
class GenerationMetrics:
    """Generation quality metrics for a single query."""

    query_id: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    answer_bleu: float = 0.0
    answer_rouge_l: float = 0.0
    grammar_score: float = 0.0
    latency_ms: float = 0.0
    answer_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "answer_bleu": self.answer_bleu,
            "answer_rouge_l": self.answer_rouge_l,
            "grammar_score": self.grammar_score,
            "latency_ms": self.latency_ms,
            "answer_length": self.answer_length,
        }


@dataclass(frozen=True)
class RAGEvaluationReport:
    """Complete evaluation report for a RAG pipeline run."""

    run_id: str
    retrieval_metrics: list[RetrievalMetrics]
    generation_metrics: list[GenerationMetrics]
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "retrieval_metrics": [r.to_dict() for r in self.retrieval_metrics],
            "generation_metrics": [g.to_dict() for g in self.generation_metrics],
            "aggregate": self.aggregate,
        }


# ---------------------------------------------------------------------------
# Retrieval evaluators
# ---------------------------------------------------------------------------


class RetrievalEvaluator:
    """Evaluate retrieval quality of a RAG pipeline.

    Computes standard IR metrics: Precision@K, Recall@K, MRR, NDCG, HitRate.
    """

    def __init__(self, k_values: list[int] | None = None) -> None:
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate_single(
        self,
        query: RAGQuery,
        result: RAGResult,
    ) -> RetrievalMetrics:
        """Evaluate retrieval for a single query-result pair."""
        retrieved_ids = [doc.doc_id for doc in result.retrieved_docs]
        relevant_ids = set(query.relevant_doc_ids)
        num_retrieved = len(retrieved_ids)

        if not relevant_ids:
            logger.warning("no_relevant_docs_for_query", query_id=query.query_id)
            return RetrievalMetrics(
                query_id=query.query_id,
                latency_ms=result.latency_ms,
                num_retrieved=num_retrieved,
            )

        # Build relevance array (1 if retrieved doc is relevant, 0 otherwise)
        relevance = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids]

        # Precision@K
        precision_at_k: dict[int, float] = {}
        for k in self.k_values:
            top_k = relevance[:k]
            precision_at_k[k] = sum(top_k) / len(top_k) if top_k else 0.0

        # Recall@K
        recall_at_k: dict[int, float] = {}
        for k in self.k_values:
            top_k = relevance[:k]
            recall_at_k[k] = sum(top_k) / len(relevant_ids) if relevant_ids else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel:
                mrr = 1.0 / rank
                break

        # NDCG@K
        ndcg_at_k: dict[int, float] = {}
        for k in self.k_values:
            ndcg_at_k[k] = self._compute_ndcg(relevance[:k], len(relevant_ids))

        # HitRate@K (binary: any relevant doc in top K)
        hit_rate_at_k: dict[int, float] = {}
        for k in self.k_values:
            hit_rate_at_k[k] = 1.0 if any(relevance[:k]) else 0.0

        return RetrievalMetrics(
            query_id=query.query_id,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            hit_rate_at_k=hit_rate_at_k,
            latency_ms=result.latency_ms,
            num_retrieved=num_retrieved,
        )

    def evaluate_batch(
        self,
        queries: list[RAGQuery],
        results: list[RAGResult],
    ) -> list[RetrievalMetrics]:
        """Evaluate retrieval for a batch of queries."""
        if len(queries) != len(results):
            raise ValueError(f"Query and result count mismatch: {len(queries)} vs {len(results)}")

        metrics: list[RetrievalMetrics] = []
        for query, result in zip(queries, results, strict=False):
            try:
                metric = self.evaluate_single(query, result)
                metrics.append(metric)
            except Exception as e:
                logger.error(
                    "retrieval_evaluation_failed",
                    query_id=query.query_id,
                    error=str(e),
                )
                metrics.append(
                    RetrievalMetrics(
                        query_id=query.query_id,
                        latency_ms=result.latency_ms,
                        num_retrieved=len(result.retrieved_docs),
                    )
                )
        return metrics

    @staticmethod
    def _compute_ndcg(relevance: list[int], num_relevant: int) -> float:
        """Compute NDCG for a single ranked list."""
        if not relevance or num_relevant == 0:
            return 0.0

        # DCG
        dcg = sum(
            rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1)=0
            for i, rel in enumerate(relevance)
        )

        # Ideal DCG: all relevant items at top
        ideal_relevance = [1] * min(num_relevant, len(relevance)) + [0] * max(
            0, len(relevance) - num_relevant
        )
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Generation evaluators
# ---------------------------------------------------------------------------


class GenerationEvaluator:
    """Evaluate generation quality of a RAG pipeline.

    Computes faithfulness, answer relevance, and similarity to expected answer.
    """

    def __init__(self, embedding_model: str | None = None) -> None:
        self.embedding_model = embedding_model
        self._embedding_fn = None
        self._grammar_metric = GrammarMetric()

    def _get_embedding(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for texts using sentence-transformers."""
        if self._embedding_fn is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
                self._embedding_fn = SentenceTransformer(model_name)
                logger.info("rag_embedding_model_loaded", model=model_name)
            except Exception as e:
                logger.error("rag_embedding_model_load_failed", error=str(e))
                raise RuntimeError(f"Failed to load embedding model: {e}") from e
        embedding_fn = self._embedding_fn
        if embedding_fn is None:
            raise RuntimeError("Embedding model failed to initialize")
        return embedding_fn.encode(texts)

    def evaluate_single(
        self,
        query: RAGQuery,
        result: RAGResult,
    ) -> GenerationMetrics:
        """Evaluate generation for a single query-result pair."""
        # Faithfulness: how much of the answer is supported by retrieved docs
        faithfulness = self._compute_faithfulness(result.generated_answer, result.retrieved_docs)

        # Answer relevance: similarity between query and answer
        answer_relevance = self._compute_answer_relevance(query.query_text, result.generated_answer)

        # Context precision: what fraction of retrieved docs are relevant
        context_precision = self._compute_context_precision(query, result.retrieved_docs)

        # BLEU and ROUGE against expected answer
        answer_bleu = 0.0
        answer_rouge_l = 0.0
        if query.expected_answer:
            answer_bleu = self._compute_answer_bleu(result.generated_answer, query.expected_answer)
            answer_rouge_l = self._compute_answer_rouge_l(
                result.generated_answer, query.expected_answer
            )

        # Grammar score
        grammar_result = self._grammar_metric.compute([result.generated_answer])
        grammar_score = grammar_result[0].value if grammar_result else 1.0

        return GenerationMetrics(
            query_id=query.query_id,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            answer_bleu=answer_bleu,
            answer_rouge_l=answer_rouge_l,
            grammar_score=grammar_score,
            latency_ms=result.latency_ms,
            answer_length=len(result.generated_answer.split()),
        )

    def evaluate_batch(
        self,
        queries: list[RAGQuery],
        results: list[RAGResult],
    ) -> list[GenerationMetrics]:
        """Evaluate generation for a batch of queries."""
        if len(queries) != len(results):
            raise ValueError(f"Query and result count mismatch: {len(queries)} vs {len(results)}")

        metrics: list[GenerationMetrics] = []
        for query, result in zip(queries, results, strict=False):
            try:
                metric = self.evaluate_single(query, result)
                metrics.append(metric)
            except Exception as e:
                logger.error(
                    "generation_evaluation_failed",
                    query_id=query.query_id,
                    error=str(e),
                )
                metrics.append(
                    GenerationMetrics(
                        query_id=query.query_id,
                        latency_ms=result.latency_ms,
                        answer_length=len(result.generated_answer.split()),
                    )
                )
        return metrics

    def _compute_faithfulness(
        self,
        answer: str,
        retrieved_docs: list[RAGDocument],
    ) -> float:
        """Compute faithfulness: answer tokens that appear in retrieved docs."""
        if not retrieved_docs or not answer:
            return 0.0

        context = " ".join(doc.content for doc in retrieved_docs).lower()
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.split())

        if not answer_tokens:
            return 0.0

        supported = len(answer_tokens & context_tokens)
        return supported / len(answer_tokens)

    def _compute_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> float:
        """Compute semantic relevance between query and answer."""
        try:
            embeddings = self._get_embedding([query, answer])
            q_emb, a_emb = embeddings[0], embeddings[1]
            similarity = float(
                np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
            )
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.error("answer_relevance_failed", error=str(e))
            return 0.0

    def _compute_context_precision(
        self,
        query: RAGQuery,
        retrieved_docs: list[RAGDocument],
    ) -> float:
        """Compute context precision: fraction of retrieved docs that are relevant."""
        if not retrieved_docs or not query.relevant_doc_ids:
            return 0.0

        relevant_ids = set(query.relevant_doc_ids)
        relevant_count = sum(1 for doc in retrieved_docs if doc.doc_id in relevant_ids)
        return relevant_count / len(retrieved_docs)

    def _compute_answer_bleu(self, answer: str, expected: str) -> float:
        """Compute BLEU score between answer and expected answer."""
        try:
            bleu_metric = MetricRegistry.get("bleu")
            scores = bleu_metric.compute([answer], [expected])
            if scores:
                return scores[0].value
        except Exception as e:
            logger.error("answer_bleu_failed", error=str(e))
        return 0.0

    def _compute_answer_rouge_l(self, answer: str, expected: str) -> float:
        """Compute ROUGE-L score between answer and expected answer."""
        try:
            rouge_metric = MetricRegistry.get("rouge")
            scores = rouge_metric.compute([answer], [expected])
            for score in scores:
                if score.name == "rougeL":
                    return score.value
        except Exception as e:
            logger.error("answer_rouge_l_failed", error=str(e))
        return 0.0


# ---------------------------------------------------------------------------
# Main RAG evaluator
# ---------------------------------------------------------------------------


class RAGEvaluator:
    """Main evaluator for RAG pipeline evaluation.

    Combines retrieval and generation evaluation into a comprehensive report.
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.retrieval_evaluator = RetrievalEvaluator(k_values=k_values)
        self.generation_evaluator = GenerationEvaluator(embedding_model=embedding_model)

    def evaluate(
        self,
        queries: list[RAGQuery],
        results: list[RAGResult],
        run_id: str = "rag_eval",
    ) -> RAGEvaluationReport:
        """Evaluate a complete RAG pipeline run."""
        logger.info(
            "rag_evaluation_started",
            run_id=run_id,
            num_queries=len(queries),
        )

        retrieval_metrics = self.retrieval_evaluator.evaluate_batch(queries, results)
        generation_metrics = self.generation_evaluator.evaluate_batch(queries, results)

        aggregate = self._compute_aggregate(retrieval_metrics, generation_metrics)

        report = RAGEvaluationReport(
            run_id=run_id,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            aggregate=aggregate,
        )

        logger.info(
            "rag_evaluation_completed",
            run_id=run_id,
            avg_mrr=aggregate.get("avg_mrr", 0.0),
            avg_faithfulness=aggregate.get("avg_faithfulness", 0.0),
        )
        return report

    def _compute_aggregate(
        self,
        retrieval_metrics: list[RetrievalMetrics],
        generation_metrics: list[GenerationMetrics],
    ) -> dict[str, Any]:
        """Compute aggregate statistics across all queries."""
        aggregate: dict[str, Any] = {}

        if retrieval_metrics:
            # Average retrieval metrics
            for k in self.retrieval_evaluator.k_values:
                p_values = [m.precision_at_k.get(k, 0.0) for m in retrieval_metrics]
                r_values = [m.recall_at_k.get(k, 0.0) for m in retrieval_metrics]
                h_values = [m.hit_rate_at_k.get(k, 0.0) for m in retrieval_metrics]
                n_values = [m.ndcg_at_k.get(k, 0.0) for m in retrieval_metrics]
                aggregate[f"avg_precision_at_{k}"] = float(np.mean(p_values))
                aggregate[f"avg_recall_at_{k}"] = float(np.mean(r_values))
                aggregate[f"avg_hit_rate_at_{k}"] = float(np.mean(h_values))
                aggregate[f"avg_ndcg_at_{k}"] = float(np.mean(n_values))

            mrr_values = [m.mrr for m in retrieval_metrics]
            aggregate["avg_mrr"] = float(np.mean(mrr_values))
            aggregate["median_latency_retrieval_ms"] = float(
                np.median([m.latency_ms for m in retrieval_metrics])
            )

        if generation_metrics:
            aggregate["avg_faithfulness"] = float(
                np.mean([m.faithfulness for m in generation_metrics])
            )
            aggregate["avg_answer_relevance"] = float(
                np.mean([m.answer_relevance for m in generation_metrics])
            )
            aggregate["avg_context_precision"] = float(
                np.mean([m.context_precision for m in generation_metrics])
            )
            aggregate["avg_answer_bleu"] = float(
                np.mean([m.answer_bleu for m in generation_metrics])
            )
            aggregate["avg_answer_rouge_l"] = float(
                np.mean([m.answer_rouge_l for m in generation_metrics])
            )
            aggregate["avg_grammar_score"] = float(
                np.mean([m.grammar_score for m in generation_metrics])
            )
            aggregate["median_latency_generation_ms"] = float(
                np.median([m.latency_ms for m in generation_metrics])
            )

        return aggregate
