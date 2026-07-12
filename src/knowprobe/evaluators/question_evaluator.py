"""Question quality evaluator for KnowProbe.

Evaluates generated questions across multiple quality dimensions:
- Relevance: how well the question relates to the knowledge source
- Type Consistency: whether the question matches the intended question type
- Answerability: whether the question can be answered from the knowledge
- Structural Grounding: schema-based questions should reference KB structure
- Fluency: grammatical and lexical quality
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from knowprobe.core.models import (
    EvaluationResult,
    GeneratedQuestion,
    KnowledgeInput,
    QuestionType,
)
from knowprobe.utils.logging import get_logger

from .metrics import GrammarMetric, MetricRegistry, MetricScore

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Quality dimension scores
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityDimension:
    """Score for a single quality dimension."""

    name: str
    score: float
    weight: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionQualityReport:
    """Complete quality report for a generated question."""

    question_id: str
    overall_score: float
    dimensions: list[QualityDimension]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "overall_score": self.overall_score,
            "dimensions": [
                {
                    "name": d.name,
                    "score": d.score,
                    "weight": d.weight,
                    "details": d.details,
                }
                for d in self.dimensions
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Dimension evaluators
# ---------------------------------------------------------------------------

class DimensionEvaluator(ABC):
    """Base class for a single quality dimension evaluator."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the quality dimension."""

    @property
    @abstractmethod
    def default_weight(self) -> float:
        """Default weight for aggregation."""

    @abstractmethod
    def evaluate(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QualityDimension:
        """Evaluate a single question."""


class RelevanceEvaluator(DimensionEvaluator):
    """Evaluate semantic relevance between question and knowledge source."""

    name = "relevance"
    default_weight = 0.25

    _model: SentenceTransformer | None = None
    _model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        if model_name:
            self._model_name = model_name
        self._embedding_model = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(self._model_name)
                logger.info(
                    "relevance_model_loaded",
                    model=self._model_name,
                )
            except Exception as e:
                logger.error("relevance_model_load_failed", error=str(e))
                raise RuntimeError(f"Failed to load embedding model: {e}") from e
        return self._embedding_model

    def evaluate(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QualityDimension:
        question_text = question.question_text
        knowledge_content = question.knowledge_input.content

        try:
            model = self._get_model()
            embeddings = model.encode([question_text, knowledge_content])
            q_emb, k_emb = embeddings[0], embeddings[1]
            # Cosine similarity
            similarity = float(
                np.dot(q_emb, k_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(k_emb))
            )
            # Normalize to 0-1 (similarity already in [-1, 1], but MiniLM is positive)
            score = max(0.0, min(1.0, similarity))
            return QualityDimension(
                name=self.name,
                score=score,
                weight=self.default_weight,
                details={
                    "cosine_similarity": similarity,
                    "knowledge_preview": knowledge_content[:200],
                },
            )
        except Exception as e:
            logger.error("relevance_evaluation_failed", error=str(e), question_id=question.id)
            return QualityDimension(
                name=self.name,
                score=0.0,
                weight=self.default_weight,
                details={"error": str(e)},
            )


class TypeConsistencyEvaluator(DimensionEvaluator):
    """Evaluate whether the question matches the intended question type.

    For factual questions: should contain specific entities, dates, or ask
    for concrete facts.
    For schema questions: should reference structural elements (classes,
    properties, relations, types).
    """

    name = "type_consistency"
    default_weight = 0.20

    # Schema-related keywords
    SCHEMA_KEYWORDS: set[str] = {
        "type", "class", "property", "relation", "attribute",
        "schema", "structure", "hierarchy", "category", "domain",
        "range", "subclass", "superclass", "ontology", "predicate",
    }

    # Factual question patterns
    FACTUAL_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(what|who|when|where|how many|how much)\b", re.IGNORECASE),
        re.compile(r"\b(is|are|was|were|did|does|do|has|have|had)\b", re.IGNORECASE),
    ]

    def evaluate(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QualityDimension:
        question_text = question.question_text.lower()
        question_type = question.question_type

        score = 0.0
        details: dict[str, Any] = {}

        if question_type == QuestionType.FACTUAL:
            score = self._evaluate_factual(question_text, details)
        elif question_type == QuestionType.SCHEMA:
            score = self._evaluate_schema(question_text, details)
        elif question_type == QuestionType.COMPOSITE:
            # Composite should show both characteristics
            factual_score = self._evaluate_factual(question_text, {})
            schema_score = self._evaluate_schema(question_text, {})
            score = (factual_score + schema_score) / 2
            details["factual_score"] = factual_score
            details["schema_score"] = schema_score
        else:
            score = 0.5
            details["reason"] = "unknown_question_type"

        details["expected_type"] = question_type.value
        return QualityDimension(
            name=self.name,
            score=score,
            weight=self.default_weight,
            details=details,
        )

    def _evaluate_factual(self, text: str, details: dict[str, Any]) -> float:
        """Score how well a question looks like a factual question."""
        pattern_matches = sum(1 for p in self.FACTUAL_PATTERNS if p.search(text))
        # Check for entity mentions (capitalized words, numbers, dates)
        entity_indicators = len(re.findall(r"\b[A-Z][a-z]+\b", text))
        entity_indicators += len(re.findall(r"\b\d{4}\b", text))  # Years
        entity_indicators += len(re.findall(r"\b\d+\b", text))  # Numbers

        score = 0.3 * min(1.0, pattern_matches / 1.0)
        score += 0.3 * min(1.0, entity_indicators / 3.0)
        # Penalize if it looks like schema
        schema_words = sum(1 for kw in self.SCHEMA_KEYWORDS if kw in text)
        score -= 0.2 * min(1.0, schema_words / 2.0)
        score = max(0.0, min(1.0, score))

        details["pattern_matches"] = pattern_matches
        details["entity_indicators"] = entity_indicators
        details["schema_word_count"] = schema_words
        return score

    def _evaluate_schema(self, text: str, details: dict[str, Any]) -> float:
        """Score how well a question looks like a schema question."""
        schema_words = sum(1 for kw in self.SCHEMA_KEYWORDS if kw in text)
        # Check for structural patterns
        structural_patterns = [
            r"what (type|kind|class) of",
            r"what (is|are) the (relation|property|attribute)",
            r"how (is|are) .* (related|connected|linked)",
            r"what (does|do) .* (belong|classify)",
        ]
        pattern_matches = sum(1 for p in structural_patterns if re.search(p, text, re.IGNORECASE))

        score = 0.4 * min(1.0, schema_words / 2.0)
        score += 0.3 * min(1.0, pattern_matches / 1.0)
        # Check if the question references specific structural elements from knowledge
        score = max(0.0, min(1.0, score))

        details["schema_word_count"] = schema_words
        details["structural_pattern_matches"] = pattern_matches
        return score


class AnswerabilityEvaluator(DimensionEvaluator):
    """Evaluate whether the question can be answered from the knowledge.

    Uses embedding-based similarity between question+knowledge vs the
    expected answerability. Also checks for answerability heuristics
    (no vague questions, no questions requiring external knowledge).
    """

    name = "answerability"
    default_weight = 0.20

    # Vague/unanswerable indicators
    VAGUE_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(why|how come|what if)\b", re.IGNORECASE),
        re.compile(r"\b(opinion|think|feel|believe)\b", re.IGNORECASE),
    ]

    # Knowledge coverage keywords (extracted from knowledge content)
    def evaluate(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QualityDimension:
        question_text = question.question_text.lower()
        knowledge = question.knowledge_input.content.lower()

        details: dict[str, Any] = {}

        # Check for vague indicators
        vague_count = sum(1 for p in self.VAGUE_PATTERNS if p.search(question_text))
        details["vague_indicators"] = vague_count

        # Check if question terms appear in knowledge
        question_terms = set(re.findall(r"\b\w{3,}\b", question_text))
        knowledge_terms = set(re.findall(r"\b\w{3,}\b", knowledge))
        overlap = len(question_terms & knowledge_terms)
        coverage_ratio = overlap / len(question_terms) if question_terms else 0.0
        details["term_coverage_ratio"] = coverage_ratio
        details["question_terms"] = len(question_terms)
        details["overlapping_terms"] = overlap

        # Score computation
        score = coverage_ratio * 0.7  # Knowledge overlap is major factor
        score -= vague_count * 0.15  # Penalize vague questions
        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name=self.name,
            score=score,
            weight=self.default_weight,
            details=details,
        )


class FluencyEvaluator(DimensionEvaluator):
    """Evaluate grammatical and lexical fluency of the question."""

    name = "fluency"
    default_weight = 0.15

    # Simple heuristic patterns for fluency issues
    FLUENCY_ISSUES: list[tuple[re.Pattern, float, str]] = [
        (re.compile(r"\b\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+\s+\w+"), 0.3, "excessively_long_sentence"),
        (re.compile(r"\b(the|a|an)\s+\1\b", re.IGNORECASE), 0.4, "repeated_article"),
        (re.compile(r"\b\w+\b\s*\(\s*\b\w+\b\s*\)\s*\b\w+\b\s*\(\s*\b\w+\b\s*\)"), 0.2, "nested_parens"),
        (re.compile(r"[^a-zA-Z0-9\s\u4e00-\u9fff.,;:!?()'\"-]"), 0.2, "special_chars"),
    ]

    def evaluate(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QualityDimension:
        text = question.question_text
        details: dict[str, Any] = {}

        # Grammar check using GrammarMetric
        grammar_metric = GrammarMetric()
        grammar_result = grammar_metric.compute([text])
        grammar_score = grammar_result[0].value if grammar_result else 1.0
        details["grammar_score"] = grammar_score

        # Fluency heuristics
        penalty = 0.0
        issues: list[str] = []
        for pattern, weight, issue_name in self.FLUENCY_ISSUES:
            if pattern.search(text):
                penalty += weight
                issues.append(issue_name)

        details["fluency_penalty"] = penalty
        details["fluency_issues"] = issues

        # Sentence length score (prefer moderate length)
        word_count = len(text.split())
        if 5 <= word_count <= 25:
            length_score = 1.0
        elif word_count < 5:
            length_score = word_count / 5.0
        else:
            length_score = max(0.3, 1.0 - (word_count - 25) / 50.0)
        details["word_count"] = word_count
        details["length_score"] = length_score

        score = 0.4 * grammar_score + 0.4 * max(0.0, 1.0 - penalty) + 0.2 * length_score
        score = max(0.0, min(1.0, score))

        return QualityDimension(
            name=self.name,
            score=score,
            weight=self.default_weight,
            details=details,
        )


class StructuralGroundingEvaluator(DimensionEvaluator):
    """Evaluate structural grounding for schema questions.

    Schema questions should explicitly reference elements from the knowledge
    graph schema (classes, properties, relations). This evaluator extracts
    schema elements from the knowledge input and checks for their presence.
    """

    name = "structural_grounding"
    default_weight = 0.20

    def evaluate(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QualityDimension:
        question_text = question.question_text.lower()
        knowledge = question.knowledge_input
        details: dict[str, Any] = {}

        # Extract potential schema elements from structured data
        schema_elements: set[str] = set()
        if knowledge.structured:
            self._extract_schema_elements(knowledge.structured, schema_elements)
        # Also extract from content using patterns
        schema_elements.update(self._extract_from_text(knowledge.content))

        details["schema_elements_found"] = len(schema_elements)

        # Check how many schema elements appear in the question
        if schema_elements:
            matched = sum(1 for elem in schema_elements if elem.lower() in question_text)
            grounding_ratio = matched / len(schema_elements)
            details["matched_elements"] = matched
            details["grounding_ratio"] = grounding_ratio
        else:
            grounding_ratio = 0.5  # Neutral if no schema elements detected
            details["reason"] = "no_schema_elements_in_knowledge"

        score = grounding_ratio

        # For non-schema questions, this dimension is less critical
        if question.question_type != QuestionType.SCHEMA:
            score = score * 0.5 + 0.5  # Scale to 0.5-1.0
            details["scaled_for_non_schema"] = True

        return QualityDimension(
            name=self.name,
            score=score,
            weight=self.default_weight,
            details=details,
        )

    def _extract_schema_elements(self, data: dict[str, Any], result: set[str]) -> None:
        """Recursively extract schema element names from structured data."""
        if isinstance(data, dict):
            for key, value in data.items():
                result.add(key)
                if isinstance(value, (dict, list)):
                    self._extract_schema_elements(value, result)
                elif isinstance(value, str):
                    result.add(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._extract_schema_elements(item, result)
                elif isinstance(item, str):
                    result.add(item)

    def _extract_from_text(self, text: str) -> set[str]:
        """Extract candidate schema elements from raw text using patterns."""
        elements: set[str] = set()
        # Match patterns like "Class: X", "Property: Y", "Relation: Z"
        patterns = [
            r"(?:class|type|category)[\s:]+([A-Za-z_][A-Za-z0-9_]*)",
            r"(?:property|attribute|field)[\s:]+([A-Za-z_][A-Za-z0-9_]*)",
            r"(?:relation|relationship|predicate)[\s:]+([A-Za-z_][A-Za-z0-9_]*)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                elements.add(match.group(1))
        return elements


# ---------------------------------------------------------------------------
# Main question evaluator
# ---------------------------------------------------------------------------

class QuestionEvaluator:
    """Main evaluator for generated question quality.

    Combines multiple dimension evaluators into an overall quality score.
    """

    def __init__(
        self,
        dimensions: list[DimensionEvaluator] | None = None,
        custom_weights: dict[str, float] | None = None,
    ) -> None:
        self.dimensions = dimensions or [
            RelevanceEvaluator(),
            TypeConsistencyEvaluator(),
            AnswerabilityEvaluator(),
            FluencyEvaluator(),
            StructuralGroundingEvaluator(),
        ]
        self.custom_weights = custom_weights or {}
        self._metric_registry = MetricRegistry()

    def evaluate_single(
        self,
        question: GeneratedQuestion,
        reference: str | None = None,
    ) -> QuestionQualityReport:
        """Evaluate a single generated question."""
        dimensions: list[QualityDimension] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for evaluator in self.dimensions:
            try:
                dim = evaluator.evaluate(question, reference)
                weight = self.custom_weights.get(dim.name, dim.weight)
                dimensions.append(
                    QualityDimension(
                        name=dim.name,
                        score=dim.score,
                        weight=weight,
                        details=dim.details,
                    )
                )
                weighted_sum += dim.score * weight
                total_weight += weight
            except Exception as e:
                logger.error(
                    "dimension_evaluation_failed",
                    dimension=evaluator.name,
                    question_id=question.id,
                    error=str(e),
                )
                # Include a failed dimension with 0 score
                weight = self.custom_weights.get(evaluator.name, evaluator.default_weight)
                dimensions.append(
                    QualityDimension(
                        name=evaluator.name,
                        score=0.0,
                        weight=weight,
                        details={"error": str(e)},
                    )
                )
                total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return QuestionQualityReport(
            question_id=question.id or "unknown",
            overall_score=overall_score,
            dimensions=dimensions,
            metadata={
                "model": question.model_name,
                "strategy": question.prompt_strategy.value,
                "question_type": question.question_type.value,
            },
        )

    def evaluate_batch(
        self,
        questions: list[GeneratedQuestion],
        references: list[str] | None = None,
    ) -> list[QuestionQualityReport]:
        """Evaluate a batch of generated questions."""
        reports: list[QuestionQualityReport] = []
        for i, question in enumerate(questions):
            ref = references[i] if references and i < len(references) else None
            try:
                report = self.evaluate_single(question, ref)
                reports.append(report)
            except Exception as e:
                logger.error(
                    "batch_evaluation_failed",
                    question_id=question.id,
                    error=str(e),
                )
                # Add a failed report
                reports.append(
                    QuestionQualityReport(
                        question_id=question.id or f"q_{i}",
                        overall_score=0.0,
                        dimensions=[],
                        metadata={"error": str(e)},
                    )
                )
        return reports

    def compute_auto_metrics(
        self,
        questions: list[GeneratedQuestion],
        references: list[str],
        metric_names: list[str] | None = None,
    ) -> list[EvaluationResult]:
        """Compute automatic metrics (BLEU, ROUGE, etc.) against references.

        Args:
            questions: Generated questions.
            references: Reference (ground-truth) questions.
            metric_names: List of metric names to compute. Defaults to all.

        Returns:
            Evaluation results per metric.
        """
        if not questions or not references:
            return []

        predictions = [q.question_text for q in questions]
        metric_names = metric_names or self._metric_registry.list_metrics()
        results: list[EvaluationResult] = []

        for name in metric_names:
            try:
                metric = self._metric_registry.get(name)
                scores = metric.compute(predictions, references)
                for score in scores:
                    results.append(
                        EvaluationResult(
                            question_id="batch",
                            metric_name=score.name,
                            score=score.value,
                            details=score.details,
                        )
                    )
            except Exception as e:
                logger.error(
                    "auto_metric_failed",
                    metric=name,
                    error=str(e),
                )
                results.append(
                    EvaluationResult(
                        question_id="batch",
                        metric_name=name,
                        score=0.0,
                        details={"error": str(e)},
                    )
                )

        return results

    def get_dimension_scores(self, reports: list[QuestionQualityReport]) -> dict[str, list[float]]:
        """Extract dimension scores across reports for statistical analysis."""
        scores: dict[str, list[float]] = {}
        for report in reports:
            for dim in report.dimensions:
                if dim.name not in scores:
                    scores[dim.name] = []
                scores[dim.name].append(dim.score)
        return scores
