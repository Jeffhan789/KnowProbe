"""Experiment runner for KnowProbe.

Manages the execution of controlled experiments comparing:
- Multiple models (Llama-3.1-8B, Qwen-2.5-7B, Flan-T5-Large, etc.)
- Multiple prompt strategies (zero_shot, few_shot, cot, self_consistency)
- Multiple question types (factual, schema, composite)
- Multiple evaluation metrics

Produces structured ExperimentResult objects with full provenance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from knowprobe.core.models import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    GeneratedQuestion,
    PromptStrategy,
    QuestionType,
)
from knowprobe.utils.logging import get_logger

from .metrics import MetricRegistry
from .question_evaluator import QuestionEvaluator, QuestionQualityReport

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Experiment result structures
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    """Results for a single experimental condition (model + strategy + type)."""

    model_name: str
    strategy: PromptStrategy
    question_type: QuestionType
    questions: list[GeneratedQuestion] = field(default_factory=list)
    auto_metrics: list[EvaluationResult] = field(default_factory=list)
    quality_reports: list[QuestionQualityReport] = field(default_factory=list)

    def get_metric_scores(self, metric_name: str) -> list[float]:
        """Extract scores for a specific auto metric."""
        scores = []
        for m in self.auto_metrics:
            if m.metric_name == metric_name and "raw_scores" in m.details:
                scores.extend(m.details["raw_scores"])
        return scores

    def get_dimension_scores(self, dimension_name: str) -> list[float]:
        """Extract scores for a specific quality dimension."""
        scores = []
        for report in self.quality_reports:
            for dim in report.dimensions:
                if dim.name == dimension_name:
                    scores.append(dim.score)
        return scores

    def get_overall_quality_scores(self) -> list[float]:
        """Extract overall quality scores."""
        return [r.overall_score for r in self.quality_reports]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "strategy": self.strategy.value,
            "question_type": self.question_type.value,
            "num_questions": len(self.questions),
            "auto_metrics": [
                {
                    "metric_name": m.metric_name,
                    "score": m.score,
                    "details": m.details,
                }
                for m in self.auto_metrics
            ],
            "quality_summary": {
                "overall_mean": float(np.mean(self.get_overall_quality_scores()))
                if self.quality_reports
                else 0.0,
                "overall_std": float(np.std(self.get_overall_quality_scores()))
                if self.quality_reports
                else 0.0,
            },
        }


@dataclass
class ComparativeAnalysis:
    """Statistical comparison between experimental conditions."""

    comparison_name: str
    baseline_condition: str
    comparison_condition: str
    metric_name: str
    baseline_mean: float
    comparison_mean: float
    difference: float
    percent_change: float
    p_value: float | None = None
    effect_size: float | None = None
    is_significant: bool = False
    significance_level: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_name": self.comparison_name,
            "baseline_condition": self.baseline_condition,
            "comparison_condition": self.comparison_condition,
            "metric_name": self.metric_name,
            "baseline_mean": self.baseline_mean,
            "comparison_mean": self.comparison_mean,
            "difference": self.difference,
            "percent_change": self.percent_change,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "is_significant": self.is_significant,
            "significance_level": self.significance_level,
        }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Orchestrates controlled experiments for question generation evaluation.

    Usage:
        runner = ExperimentRunner(config, question_generator)
        result = runner.run(knowledge_inputs, reference_questions)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        question_generator: Any | None = None,
    ) -> None:
        self.config = config
        self.question_generator = question_generator
        self.evaluator = QuestionEvaluator()
        self.metric_registry = MetricRegistry()
        self.condition_results: list[ConditionResult] = []

    def run(
        self,
        knowledge_inputs: list[Any],
        reference_questions: dict[str, list[str]] | None = None,
    ) -> ExperimentResult:
        """Run the full experiment.

        Args:
            knowledge_inputs: List of knowledge inputs for question generation.
            reference_questions: Optional mapping from knowledge source ID to
                reference questions for automatic metric computation.

        Returns:
            Structured experiment result with all questions and evaluations.
        """
        logger.info(
            "experiment_started",
            experiment_id=self.config.experiment_id,
            name=self.config.name,
            models=self.config.models,
            strategies=[s.value for s in self.config.prompt_strategies],
            types=[t.value for t in self.config.question_types],
        )

        all_questions: list[GeneratedQuestion] = []
        all_evaluations: list[EvaluationResult] = []
        self.condition_results = []

        # Generate all condition combinations
        for model_name in self.config.models:
            for strategy in self.config.prompt_strategies:
                for q_type in self.config.question_types:
                    condition = self._run_condition(
                        model_name=model_name,
                        strategy=strategy,
                        question_type=q_type,
                        knowledge_inputs=knowledge_inputs,
                        reference_questions=reference_questions,
                    )
                    self.condition_results.append(condition)
                    all_questions.extend(condition.questions)
                    all_evaluations.extend(condition.auto_metrics)

        # Build summary statistics
        summary = self._build_summary()

        experiment_result = ExperimentResult(
            experiment_id=self.config.experiment_id,
            config=self.config,
            questions=all_questions,
            evaluations=all_evaluations,
            summary=summary,
        )

        logger.info(
            "experiment_completed",
            experiment_id=self.config.experiment_id,
            total_questions=len(all_questions),
            total_evaluations=len(all_evaluations),
        )
        return experiment_result

    def _run_condition(
        self,
        model_name: str,
        strategy: PromptStrategy,
        question_type: QuestionType,
        knowledge_inputs: list[Any],
        reference_questions: dict[str, list[str]] | None,
    ) -> ConditionResult:
        """Run a single experimental condition."""
        logger.info(
            "condition_started",
            model=model_name,
            strategy=strategy.value,
            question_type=question_type.value,
        )

        # Generate questions (if generator provided)
        questions: list[GeneratedQuestion] = []
        if self.question_generator is not None:
            questions = self._generate_questions(
                model_name=model_name,
                strategy=strategy,
                question_type=question_type,
                knowledge_inputs=knowledge_inputs,
            )
        else:
            logger.warning("no_generator_provided", condition=f"{model_name}_{strategy.value}")
            # Create placeholder questions for evaluation
            questions = self._create_placeholder_questions(
                knowledge_inputs, model_name, strategy, question_type
            )

        # Quality evaluation
        quality_reports = self.evaluator.evaluate_batch(questions)

        # Automatic metrics (if references provided)
        auto_metrics: list[EvaluationResult] = []
        if reference_questions:
            refs = self._collect_references(questions, reference_questions)
            if refs:
                auto_metrics = self.evaluator.compute_auto_metrics(
                    questions=questions,
                    references=refs,
                    metric_names=self.config.evaluation_metrics,
                )

        condition = ConditionResult(
            model_name=model_name,
            strategy=strategy,
            question_type=question_type,
            questions=questions,
            auto_metrics=auto_metrics,
            quality_reports=quality_reports,
        )

        logger.info(
            "condition_completed",
            model=model_name,
            strategy=strategy.value,
            question_type=question_type.value,
            num_questions=len(questions),
        )
        return condition

    def _generate_questions(
        self,
        model_name: str,
        strategy: PromptStrategy,
        question_type: QuestionType,
        knowledge_inputs: list[Any],
    ) -> list[GeneratedQuestion]:
        """Generate questions using the configured generator."""
        if self.question_generator is None:
            raise RuntimeError("A question_generator is required for live generation")
        questions: list[GeneratedQuestion] = []
        for ki in knowledge_inputs:
            try:
                # Assume question_generator has a generate method
                q = self.question_generator.generate(
                    knowledge_input=ki,
                    model_name=model_name,
                    strategy=strategy,
                    question_type=question_type,
                )
                questions.append(q)
            except Exception as e:
                logger.error(
                    "question_generation_failed",
                    model=model_name,
                    strategy=strategy.value,
                    error=str(e),
                )
        return questions

    def _create_placeholder_questions(
        self,
        knowledge_inputs: list[Any],
        model_name: str,
        strategy: PromptStrategy,
        question_type: QuestionType,
    ) -> list[GeneratedQuestion]:
        """Create placeholder questions when no generator is available."""
        from knowprobe.core.models import KnowledgeInput, ModelProvider

        questions: list[GeneratedQuestion] = []
        for i, ki in enumerate(knowledge_inputs):
            if isinstance(ki, KnowledgeInput):
                questions.append(
                    GeneratedQuestion(
                        id=f"{model_name}_{strategy.value}_{question_type.value}_{i}",
                        question_text=f"Placeholder question for {ki.source_id}",
                        knowledge_input=ki,
                        question_type=question_type,
                        prompt_strategy=strategy,
                        model_name=model_name,
                        model_provider=ModelProvider.OLLAMA,
                    )
                )
        return questions

    def _collect_references(
        self,
        questions: list[GeneratedQuestion],
        reference_questions: dict[str, list[str]],
    ) -> list[str]:
        """Collect reference questions aligned with generated questions."""
        refs: list[str] = []
        for q in questions:
            source_refs = reference_questions.get(q.knowledge_input.source_id, [])
            if source_refs:
                refs.append(source_refs[0])  # Use first reference
            else:
                refs.append("")  # No reference available
        return refs

    def _build_summary(self) -> dict[str, Any]:
        """Build summary statistics across all conditions."""
        summary: dict[str, Any] = {
            "num_conditions": len(self.condition_results),
            "models_tested": self.config.models,
            "strategies_tested": [s.value for s in self.config.prompt_strategies],
            "types_tested": [t.value for t in self.config.question_types],
        }

        # Aggregate by model
        model_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # Aggregate by strategy
        strategy_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # Aggregate by question type
        type_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for condition in self.condition_results:
            overall_scores = condition.get_overall_quality_scores()
            if overall_scores:
                model_stats[condition.model_name]["overall_quality"].extend(overall_scores)
                strategy_stats[condition.strategy.value]["overall_quality"].extend(overall_scores)
                type_stats[condition.question_type.value]["overall_quality"].extend(overall_scores)

            # Dimension scores
            for dim_name in [
                "relevance",
                "type_consistency",
                "answerability",
                "fluency",
                "structural_grounding",
            ]:
                scores = condition.get_dimension_scores(dim_name)
                if scores:
                    model_stats[condition.model_name][dim_name].extend(scores)
                    strategy_stats[condition.strategy.value][dim_name].extend(scores)
                    type_stats[condition.question_type.value][dim_name].extend(scores)

        # Compute means
        summary["by_model"] = self._compute_means(model_stats)
        summary["by_strategy"] = self._compute_means(strategy_stats)
        summary["by_question_type"] = self._compute_means(type_stats)

        return summary

    @staticmethod
    def _compute_means(
        stats: dict[str, dict[str, list[float]]],
    ) -> dict[str, dict[str, float]]:
        """Compute mean scores for each category and dimension."""
        result: dict[str, dict[str, float]] = {}
        for key, dimensions in stats.items():
            result[key] = {}
            for dim_name, scores in dimensions.items():
                result[key][dim_name] = float(np.mean(scores)) if scores else 0.0
        return result

    # -----------------------------------------------------------------------
    # Comparative analysis
    # -----------------------------------------------------------------------

    def compare_strategies(
        self,
        baseline: PromptStrategy,
        comparison: PromptStrategy,
        metric_name: str = "overall_quality",
    ) -> ComparativeAnalysis | None:
        """Compare two prompt strategies statistically."""
        baseline_scores: list[float] = []
        comparison_scores: list[float] = []

        for condition in self.condition_results:
            if metric_name == "overall_quality":
                scores = condition.get_overall_quality_scores()
            else:
                scores = condition.get_dimension_scores(metric_name)

            if condition.strategy == baseline:
                baseline_scores.extend(scores)
            elif condition.strategy == comparison:
                comparison_scores.extend(scores)

        if not baseline_scores or not comparison_scores:
            return None

        return self._perform_comparison(
            baseline_name=baseline.value,
            comparison_name=comparison.value,
            metric_name=metric_name,
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
        )

    def compare_models(
        self,
        baseline_model: str,
        comparison_model: str,
        metric_name: str = "overall_quality",
    ) -> ComparativeAnalysis | None:
        """Compare two models statistically."""
        baseline_scores: list[float] = []
        comparison_scores: list[float] = []

        for condition in self.condition_results:
            if metric_name == "overall_quality":
                scores = condition.get_overall_quality_scores()
            else:
                scores = condition.get_dimension_scores(metric_name)

            if condition.model_name == baseline_model:
                baseline_scores.extend(scores)
            elif condition.model_name == comparison_model:
                comparison_scores.extend(scores)

        if not baseline_scores or not comparison_scores:
            return None

        return self._perform_comparison(
            baseline_name=baseline_model,
            comparison_name=comparison_model,
            metric_name=metric_name,
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
        )

    def compare_question_types(
        self,
        baseline_type: QuestionType,
        comparison_type: QuestionType,
        metric_name: str = "overall_quality",
    ) -> ComparativeAnalysis | None:
        """Compare two question types statistically."""
        baseline_scores: list[float] = []
        comparison_scores: list[float] = []

        for condition in self.condition_results:
            if metric_name == "overall_quality":
                scores = condition.get_overall_quality_scores()
            else:
                scores = condition.get_dimension_scores(metric_name)

            if condition.question_type == baseline_type:
                baseline_scores.extend(scores)
            elif condition.question_type == comparison_type:
                comparison_scores.extend(scores)

        if not baseline_scores or not comparison_scores:
            return None

        return self._perform_comparison(
            baseline_name=baseline_type.value,
            comparison_name=comparison_type.value,
            metric_name=metric_name,
            baseline_scores=baseline_scores,
            comparison_scores=comparison_scores,
        )

    def _perform_comparison(
        self,
        baseline_name: str,
        comparison_name: str,
        metric_name: str,
        baseline_scores: list[float],
        comparison_scores: list[float],
    ) -> ComparativeAnalysis:
        """Perform statistical comparison between two sets of scores."""
        baseline_mean = float(np.mean(baseline_scores))
        comparison_mean = float(np.mean(comparison_scores))
        difference = comparison_mean - baseline_mean
        percent_change = (difference / baseline_mean * 100) if baseline_mean != 0 else 0.0

        # Paired t-test if same number of samples, otherwise Welch's t-test
        if len(baseline_scores) == len(comparison_scores) and len(baseline_scores) > 1:
            t_stat, p_value = stats.ttest_rel(baseline_scores, comparison_scores)
        else:
            t_stat, p_value = stats.ttest_ind(baseline_scores, comparison_scores, equal_var=False)

        # Cohen's d effect size
        pooled_std = (
            np.sqrt((np.var(baseline_scores) + np.var(comparison_scores)) / 2)
            if len(baseline_scores) > 1 and len(comparison_scores) > 1
            else 0.0
        )
        effect_size = difference / pooled_std if pooled_std > 0 else 0.0

        significance_level = 0.05
        is_significant = p_value < significance_level if p_value is not None else False

        return ComparativeAnalysis(
            comparison_name=f"{baseline_name}_vs_{comparison_name}",
            baseline_condition=baseline_name,
            comparison_condition=comparison_name,
            metric_name=metric_name,
            baseline_mean=baseline_mean,
            comparison_mean=comparison_mean,
            difference=difference,
            percent_change=percent_change,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=is_significant,
            significance_level=significance_level,
        )

    def get_all_comparisons(
        self, metric_name: str = "overall_quality"
    ) -> list[ComparativeAnalysis]:
        """Get all pairwise comparisons for strategies, models, and types."""
        comparisons: list[ComparativeAnalysis] = []

        # Strategy comparisons
        strategies = self.config.prompt_strategies
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                comp = self.compare_strategies(strategies[i], strategies[j], metric_name)
                if comp:
                    comparisons.append(comp)

        # Model comparisons
        models = self.config.models
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                comp = self.compare_models(models[i], models[j], metric_name)
                if comp:
                    comparisons.append(comp)

        # Type comparisons
        types = self.config.question_types
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                comp = self.compare_question_types(types[i], types[j], metric_name)
                if comp:
                    comparisons.append(comp)

        return comparisons

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_results(self, experiment_result: ExperimentResult, output_dir: str | Path) -> Path:
        """Save experiment results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{experiment_result.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        filepath = output_path / filename

        # Convert to serializable dict
        data = {
            "experiment_id": experiment_result.experiment_id,
            "config": experiment_result.config.model_dump(),
            "questions_count": len(experiment_result.questions),
            "evaluations_count": len(experiment_result.evaluations),
            "summary": experiment_result.summary,
            "conditions": [c.to_dict() for c in self.condition_results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info("experiment_results_saved", filepath=str(filepath))
        return filepath

    def load_results(self, filepath: str | Path) -> dict[str, Any]:
        """Load experiment results from JSON."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return data
