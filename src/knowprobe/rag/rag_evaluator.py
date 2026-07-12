"""RAG evaluation metrics and assessment."""

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from knowprobe.core.models import RAGChunk, RAGMetrics, RAGPipelineResult, RAGQuery
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]


class RAGMetricCalculator:
    """Calculate individual RAG evaluation metrics."""

    @staticmethod
    def recall_at_k(
        retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
    ) -> float:
        """Calculate Recall@K: proportion of relevant docs retrieved in top-k."""
        if not relevant_ids:
            return 0.0
        retrieved_top_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        return len(retrieved_top_k & relevant) / len(relevant)

    @staticmethod
    def precision_at_k(
        retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
    ) -> float:
        """Calculate Precision@K: proportion of retrieved docs that are relevant."""
        if k == 0:
            return 0.0
        retrieved_top_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        return len(retrieved_top_k & relevant) / k

    @staticmethod
    def mrr(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
        """Calculate Mean Reciprocal Rank of first relevant document."""
        relevant = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
    ) -> float:
        """Calculate NDCG@K: normalized discounted cumulative gain."""
        if not relevant_ids or k == 0:
            return 0.0

        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 1)

        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant_ids)) + 1))

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def bleu_score(reference: str, hypothesis: str) -> float:
        """Calculate BLEU-4 score between reference and hypothesis."""
        try:
            from sacrebleu import sentence_bleu

            score = sentence_bleu(hypothesis, [reference])
            return score.score / 100.0
        except ImportError:
            logger.warning("metrics.bleu_import_error")
            return 0.0
        except Exception as e:
            logger.error("metrics.bleu_error", error=str(e))
            return 0.0

    @staticmethod
    def rouge_l_score(reference: str, hypothesis: str) -> float:
        """Calculate ROUGE-L score between reference and hypothesis."""
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            return float(scores["rougeL"].fmeasure)
        except ImportError:
            logger.warning("metrics.rouge_import_error")
            return 0.0
        except Exception as e:
            logger.error("metrics.rouge_error", error=str(e))
            return 0.0

    @staticmethod
    def bertscore(reference: str, hypothesis: str) -> float:
        """Calculate BERTScore F1 between reference and hypothesis."""
        try:
            from bert_score import score as bert_score_fn

            _, _, f1 = bert_score_fn(
                [hypothesis], [reference], lang="en", verbose=False, device="cpu"
            )
            return float(f1[0])
        except ImportError:
            logger.warning("metrics.bertscore_import_error")
            return 0.0
        except Exception as e:
            logger.error("metrics.bertscore_error", error=str(e))
            return 0.0

    @staticmethod
    def exact_match(reference: str, hypothesis: str) -> float:
        """Calculate exact match score (1.0 if exact match, 0.0 otherwise)."""
        return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0

    @staticmethod
    def contains_answer(reference: str, hypothesis: str) -> float:
        """Check if hypothesis contains the reference answer (or vice versa)."""
        ref_lower = reference.strip().lower()
        hyp_lower = hypothesis.strip().lower()
        if ref_lower in hyp_lower or hyp_lower in ref_lower:
            return 1.0
        return 0.0

    @staticmethod
    def semantic_similarity(reference: str, hypothesis: str) -> float:
        """Calculate semantic similarity using cosine similarity of embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode([reference, hypothesis], normalize_embeddings=True)
            return float(np.dot(embeddings[0], embeddings[1]))
        except ImportError:
            logger.warning("metrics.semantic_similarity_import_error")
            return 0.0
        except Exception as e:
            logger.error("metrics.semantic_similarity_error", error=str(e))
            return 0.0

    @staticmethod
    def context_relevance(
        query_text: str, retrieved_chunks: Sequence[RAGChunk]
    ) -> float:
        """Calculate average semantic similarity between query and retrieved chunks."""
        if not retrieved_chunks:
            return 0.0
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            texts = [query_text] + [c.content for c in retrieved_chunks]
            embeddings = model.encode(texts, normalize_embeddings=True)
            similarities = [
                float(np.dot(embeddings[0], emb)) for emb in embeddings[1:]
            ]
            return float(np.mean(similarities))
        except ImportError:
            return 0.0
        except Exception as e:
            logger.error("metrics.context_relevance_error", error=str(e))
            return 0.0

    @staticmethod
    def answer_faithfulness(
        generated_answer: str, retrieved_chunks: Sequence[RAGChunk]
    ) -> float:
        """Calculate faithfulness of answer to retrieved context."""
        if not retrieved_chunks or not generated_answer.strip():
            return 0.0
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            context = " ".join(c.content for c in retrieved_chunks)
            embeddings = model.encode([generated_answer, context], normalize_embeddings=True)
            return float(np.dot(embeddings[0], embeddings[1]))
        except ImportError:
            return 0.0
        except Exception as e:
            logger.error("metrics.faithfulness_error", error=str(e))
            return 0.0


class RAGEvaluator:
    """Evaluate RAG pipeline results against expected answers."""

    def __init__(self) -> None:
        self.calculator = RAGMetricCalculator()
        logger.info("evaluator.init")

    def evaluate(
        self,
        pipeline_result: RAGPipelineResult,
        query: RAGQuery | None = None,
        k_values: list[int] | None = None,
    ) -> RAGMetrics:
        """Evaluate a single RAG pipeline result."""
        if k_values is None:
            k_values = [1, 3, 5]

        retrieval_scores: dict[str, float] = {}
        generation_scores: dict[str, float] = {}
        end_to_end_scores: dict[str, float] = {}

        # Determine relevant docs
        relevant_doc_ids = query.relevant_doc_ids if query else []
        retrieved_doc_ids = [
            r.chunk.doc_id for r in pipeline_result.retrieval_results
        ]

        # Retrieval metrics
        if relevant_doc_ids:
            for k in k_values:
                retrieval_scores[f"recall@{k}"] = self.calculator.recall_at_k(
                    retrieved_doc_ids, relevant_doc_ids, k
                )
                retrieval_scores[f"precision@{k}"] = self.calculator.precision_at_k(
                    retrieved_doc_ids, relevant_doc_ids, k
                )
                retrieval_scores[f"ndcg@{k}"] = self.calculator.ndcg_at_k(
                    retrieved_doc_ids, relevant_doc_ids, k
                )
            retrieval_scores["mrr"] = self.calculator.mrr(
                retrieved_doc_ids, relevant_doc_ids
            )

        # Context relevance
        if query and pipeline_result.retrieval_results:
            retrieval_scores["context_relevance"] = self.calculator.context_relevance(
                query.query_text,
                [r.chunk for r in pipeline_result.retrieval_results],
            )

        # Generation metrics
        expected = query.expected_answer if query else ""
        generated = pipeline_result.generated_answer

        if expected and generated:
            generation_scores["bleu_4"] = self.calculator.bleu_score(expected, generated)
            generation_scores["rouge_l"] = self.calculator.rouge_l_score(expected, generated)
            generation_scores["bertscore_f1"] = self.calculator.bertscore(expected, generated)
            generation_scores["exact_match"] = self.calculator.exact_match(expected, generated)
            generation_scores["contains_answer"] = self.calculator.contains_answer(
                expected, generated
            )
            generation_scores["semantic_similarity"] = self.calculator.semantic_similarity(
                expected, generated
            )

        # End-to-end metrics
        if expected and generated:
            end_to_end_scores["answer_accuracy"] = generation_scores.get(
                "semantic_similarity", 0.0
            )

        if pipeline_result.retrieval_results and generated:
            end_to_end_scores["faithfulness"] = self.calculator.answer_faithfulness(
                generated, [r.chunk for r in pipeline_result.retrieval_results]
            )

        metrics = RAGMetrics(
            query_id=pipeline_result.query.query_id,
            retrieval=retrieval_scores,
            generation=generation_scores,
            end_to_end=end_to_end_scores,
            latency_ms=pipeline_result.latency_ms,
        )

        logger.info(
            "evaluator.complete",
            query_id=pipeline_result.query.query_id,
            num_metrics=len(retrieval_scores) + len(generation_scores) + len(end_to_end_scores),
        )
        return metrics

    def evaluate_batch(
        self,
        pipeline_results: list[RAGPipelineResult],
        queries: list[RAGQuery] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a batch of RAG pipeline results and aggregate scores."""
        query_map = {q.query_id: q for q in queries} if queries else {}

        all_metrics: list[RAGMetrics] = []
        for result in pipeline_results:
            query = query_map.get(result.query.query_id)
            metrics = self.evaluate(result, query=query)
            all_metrics.append(metrics)

        # Aggregate scores
        aggregate: dict[str, Any] = {
            "num_queries": len(pipeline_results),
            "retrieval": self._aggregate_scores([m.retrieval for m in all_metrics]),
            "generation": self._aggregate_scores([m.generation for m in all_metrics]),
            "end_to_end": self._aggregate_scores([m.end_to_end for m in all_metrics]),
            "avg_latency_ms": float(
                np.mean([m.latency_ms for m in all_metrics])
            ) if all_metrics else 0.0,
        }

        logger.info("evaluator.batch_complete", num_results=len(pipeline_results))
        return {"metrics": all_metrics, "aggregate": aggregate}

    @staticmethod
    def _aggregate_scores(scores_list: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate a list of score dictionaries by averaging."""
        if not scores_list:
            return {}

        all_keys = set()
        for scores in scores_list:
            all_keys.update(scores.keys())

        aggregated: dict[str, float] = {}
        for key in all_keys:
            values = [s[key] for s in scores_list if key in s]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        return aggregated
