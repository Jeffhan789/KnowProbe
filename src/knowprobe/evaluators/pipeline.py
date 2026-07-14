"""Evaluation pipeline for KnowProbe.

Orchestrates the complete evaluation workflow:
1. Load data and configuration
2. Run automatic metrics (BLEU, ROUGE, BERTScore, etc.)
3. Run quality dimension evaluation (relevance, consistency, answerability, etc.)
4. Run RAG evaluation (if applicable)
5. Aggregate and compare results
6. Generate reports (CSV, JSON, Markdown, LaTeX)

Designed for both batch evaluation and interactive use.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from knowprobe.core.models import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    GeneratedQuestion,
    RAGQuery,
    RAGResult,
)
from knowprobe.utils.logging import get_logger

from .experiment_runner import ComparativeAnalysis, ExperimentRunner
from .question_evaluator import QuestionEvaluator, QuestionQualityReport
from .rag_evaluator import RAGEvaluationReport, RAGEvaluator
from .reporter import EvaluationReport, EvaluationReporter

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


class PipelineConfig:
    """Configuration for the evaluation pipeline."""

    def __init__(
        self,
        auto_metrics: list[str] | None = None,
        quality_dimensions: list[str] | None = None,
        enable_rag_eval: bool = False,
        k_values: list[int] | None = None,
        output_dir: str = "outputs/evaluation",
        confidence_level: float = 0.95,
        export_formats: list[str] | None = None,
    ) -> None:
        self.auto_metrics = auto_metrics or ["bleu", "rouge", "bert_score"]
        self.quality_dimensions = quality_dimensions or [
            "relevance",
            "type_consistency",
            "answerability",
            "fluency",
            "structural_grounding",
        ]
        self.enable_rag_eval = enable_rag_eval
        self.k_values = k_values or [1, 3, 5, 10]
        self.output_dir = Path(output_dir)
        self.confidence_level = confidence_level
        self.export_formats = export_formats or ["json", "csv", "markdown"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_metrics": self.auto_metrics,
            "quality_dimensions": self.quality_dimensions,
            "enable_rag_eval": self.enable_rag_eval,
            "k_values": self.k_values,
            "output_dir": str(self.output_dir),
            "confidence_level": self.confidence_level,
            "export_formats": self.export_formats,
        }


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


class PipelineStage:
    """Base class for a pipeline stage."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the stage and return updated context."""
        raise NotImplementedError


class PipelineContext:
    """Shared context passed through pipeline stages."""

    def __init__(self) -> None:
        self.questions: list[GeneratedQuestion] = []
        self.references: list[str] = []
        self.quality_reports: list[QuestionQualityReport] = []
        self.auto_metric_results: list[EvaluationResult] = []
        self.rag_queries: list[RAGQuery] = []
        self.rag_results: list[RAGResult] = []
        self.rag_report: RAGEvaluationReport | None = None
        self.experiment_result: ExperimentResult | None = None
        self.comparisons: list[ComparativeAnalysis] = []
        self.report: EvaluationReport | None = None
        self.metadata: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_questions": len(self.questions),
            "num_references": len(self.references),
            "num_quality_reports": len(self.quality_reports),
            "num_auto_metrics": len(self.auto_metric_results),
            "num_rag_queries": len(self.rag_queries),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


class LoadDataStage(PipelineStage):
    """Stage 1: Load generated questions and references."""

    def __init__(self) -> None:
        super().__init__("load_data")

    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("pipeline_stage_load_data")
        # Data is expected to be pre-loaded in context
        if not context.questions:
            logger.warning("no_questions_loaded")
        return context


class QualityEvaluationStage(PipelineStage):
    """Stage 2: Evaluate question quality dimensions."""

    def __init__(self, evaluator: QuestionEvaluator | None = None) -> None:
        super().__init__("quality_evaluation")
        self.evaluator = evaluator or QuestionEvaluator()

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.questions:
            logger.warning("quality_eval_skipped_no_questions")
            return context

        logger.info(
            "pipeline_stage_quality_eval",
            num_questions=len(context.questions),
        )

        context.quality_reports = self.evaluator.evaluate_batch(
            context.questions,
            context.references if context.references else None,
        )

        logger.info(
            "quality_eval_completed",
            num_reports=len(context.quality_reports),
        )
        return context


class AutoMetricStage(PipelineStage):
    """Stage 3: Compute automatic metrics (BLEU, ROUGE, etc.)."""

    def __init__(
        self,
        metric_names: list[str] | None = None,
        evaluator: QuestionEvaluator | None = None,
    ) -> None:
        super().__init__("auto_metrics")
        self.metric_names = metric_names
        self.evaluator = evaluator or QuestionEvaluator()

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.questions or not context.references:
            logger.warning("auto_metrics_skipped_no_data")
            return context

        logger.info(
            "pipeline_stage_auto_metrics",
            metrics=self.metric_names,
            num_questions=len(context.questions),
        )

        context.auto_metric_results = self.evaluator.compute_auto_metrics(
            questions=context.questions,
            references=context.references,
            metric_names=self.metric_names,
        )

        logger.info(
            "auto_metrics_completed",
            num_results=len(context.auto_metric_results),
        )
        return context


class RAGEvaluationStage(PipelineStage):
    """Stage 4: Evaluate RAG pipeline (optional)."""

    def __init__(
        self,
        k_values: list[int] | None = None,
        evaluator: RAGEvaluator | None = None,
    ) -> None:
        super().__init__("rag_evaluation")
        self.k_values = k_values
        self.evaluator = evaluator

    def run(self, context: PipelineContext) -> PipelineContext:
        if not context.rag_queries or not context.rag_results:
            logger.warning("rag_eval_skipped_no_data")
            return context

        logger.info(
            "pipeline_stage_rag_eval",
            num_queries=len(context.rag_queries),
        )

        if self.evaluator is None:
            self.evaluator = RAGEvaluator(k_values=self.k_values)

        context.rag_report = self.evaluator.evaluate(
            queries=context.rag_queries,
            results=context.rag_results,
            run_id=context.metadata.get("run_id", "rag_eval"),
        )

        logger.info(
            "rag_eval_completed",
            avg_mrr=context.rag_report.aggregate.get("avg_mrr", 0.0),
        )
        return context


class ExperimentComparisonStage(PipelineStage):
    """Stage 5: Run experiment comparisons (if experiment config provided)."""

    def __init__(self, runner: ExperimentRunner | None = None) -> None:
        super().__init__("experiment_comparison")
        self.runner = runner

    def run(self, context: PipelineContext) -> PipelineContext:
        if self.runner is None or not self.runner.condition_results:
            logger.warning("experiment_comparison_skipped")
            return context

        logger.info("pipeline_stage_experiment_comparison")
        context.comparisons = self.runner.get_all_comparisons()

        logger.info(
            "experiment_comparison_completed",
            num_comparisons=len(context.comparisons),
        )
        return context


class ReportGenerationStage(PipelineStage):
    """Stage 6: Generate evaluation report."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__("report_generation")
        self.config = config
        self.reporter = EvaluationReporter(confidence_level=config.confidence_level)

    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info("pipeline_stage_report_generation")

        # Build raw data for statistics
        raw_data = self._build_raw_data(context)

        # Build comparison data
        comparisons = self._build_comparison_data(context)

        # Build report
        context.report = self.reporter.build_report(
            title="KnowProbe Evaluation Report",
            raw_data=raw_data,
            comparisons=comparisons,
            metadata=context.metadata,
        )

        logger.info("report_generation_completed")
        return context

    def _build_raw_data(
        self,
        context: PipelineContext,
    ) -> dict[str, dict[str, list[float]]]:
        """Build raw data dictionary from context for statistics."""
        raw_data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        # Quality dimension scores
        for report in context.quality_reports:
            group_key = f"{report.metadata.get('model', 'unknown')}_{report.metadata.get('strategy', 'unknown')}"
            raw_data[group_key]["overall_quality"].append(report.overall_score)
            for dim in report.dimensions:
                raw_data[group_key][dim.name].append(dim.score)

        # Auto metric scores (grouped by metric name)
        for result in context.auto_metric_results:
            # Auto metrics are typically aggregated; store under "all" group
            if result.metric_name not in raw_data["all"]:
                raw_data["all"][result.metric_name] = []
            raw_data["all"][result.metric_name].append(result.score)

        # RAG metrics
        if context.rag_report:
            for r_metric in context.rag_report.retrieval_metrics:
                raw_data["rag_retrieval"]["mrr"].append(r_metric.mrr)
                for k in r_metric.precision_at_k:
                    raw_data["rag_retrieval"][f"p@{k}"].append(r_metric.precision_at_k[k])
                for k in r_metric.recall_at_k:
                    raw_data["rag_retrieval"][f"r@{k}"].append(r_metric.recall_at_k[k])

            for g_metric in context.rag_report.generation_metrics:
                raw_data["rag_generation"]["faithfulness"].append(g_metric.faithfulness)
                raw_data["rag_generation"]["answer_relevance"].append(g_metric.answer_relevance)
                raw_data["rag_generation"]["grammar"].append(g_metric.grammar_score)

        return dict(raw_data)

    def _build_comparison_data(
        self,
        context: PipelineContext,
    ) -> list[dict[str, Any]]:
        """Build comparison data from comparative analyses."""
        comparisons: list[dict[str, Any]] = []
        for comp in context.comparisons:
            comparisons.append(comp.to_dict())
        return comparisons


class ExportStage(PipelineStage):
    """Stage 7: Export results to files."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__("export")
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.report is None:
            logger.warning("export_skipped_no_report")
            return context

        logger.info(
            "pipeline_stage_export",
            formats=self.config.export_formats,
            output_dir=str(self.config.output_dir),
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        reporter = EvaluationReporter(confidence_level=self.config.confidence_level)

        if "json" in self.config.export_formats:
            reporter.export_json(context.report, self.config.output_dir)
        if "csv" in self.config.export_formats:
            reporter.export_csv(context.report, self.config.output_dir)
        if "markdown" in self.config.export_formats:
            reporter.export_markdown(context.report, self.config.output_dir)

        # Also save raw data
        raw_data_path = self.config.output_dir / "pipeline_context.json"
        with open(raw_data_path, "w", encoding="utf-8") as f:
            json.dump(context.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        logger.info("export_completed", output_dir=str(self.config.output_dir))
        return context


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class EvaluationPipeline:
    """Orchestrates the complete evaluation workflow.

    Usage:
        pipeline = EvaluationPipeline(config)
        context = pipeline.run(questions, references)
        # Access results
        print(context.report.to_markdown())
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        stages: list[PipelineStage] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.stages = stages or self._default_stages()

    def _default_stages(self) -> list[PipelineStage]:
        """Build default pipeline stages."""
        return [
            LoadDataStage(),
            QualityEvaluationStage(),
            AutoMetricStage(
                metric_names=self.config.auto_metrics,
            ),
            RAGEvaluationStage(
                k_values=self.config.k_values,
            )
            if self.config.enable_rag_eval
            else PipelineStage("rag_evaluation_skip"),
            ReportGenerationStage(self.config),
            ExportStage(self.config),
        ]

    def run(
        self,
        questions: list[GeneratedQuestion],
        references: list[str] | None = None,
        rag_queries: list[RAGQuery] | None = None,
        rag_results: list[RAGResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineContext:
        """Run the full evaluation pipeline.

        Args:
            questions: Generated questions to evaluate.
            references: Optional reference questions for automatic metrics.
            rag_queries: Optional RAG queries for RAG evaluation.
            rag_results: Optional RAG results for RAG evaluation.
            metadata: Additional metadata to include in the report.

        Returns:
            PipelineContext with all evaluation results.
        """
        context = PipelineContext()
        context.questions = questions
        context.references = references or []
        context.rag_queries = rag_queries or []
        context.rag_results = rag_results or []
        context.metadata = metadata or {}

        logger.info(
            "evaluation_pipeline_started",
            num_questions=len(questions),
            num_rag_queries=len(rag_queries or []),
        )

        for stage in self.stages:
            try:
                logger.info("pipeline_stage_running", stage=stage.name)
                context = stage.run(context)
            except Exception as e:
                logger.error(
                    "pipeline_stage_failed",
                    stage=stage.name,
                    error=str(e),
                )
                # Continue with remaining stages

        logger.info("evaluation_pipeline_completed")
        return context

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        knowledge_inputs: list[Any],
        reference_questions: dict[str, list[str]] | None = None,
        question_generator: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineContext:
        """Run an experiment and evaluate results.

        Args:
            experiment_config: Configuration for the experiment.
            knowledge_inputs: Knowledge inputs for question generation.
            reference_questions: Optional reference questions.
            question_generator: Optional question generator.
            metadata: Additional metadata.

        Returns:
            PipelineContext with experiment results and evaluation.
        """
        # Run experiment
        runner = ExperimentRunner(
            config=experiment_config,
            question_generator=question_generator,
        )
        experiment_result = runner.run(
            knowledge_inputs=knowledge_inputs,
            reference_questions=reference_questions,
        )

        # Collect all questions and references
        all_questions = experiment_result.questions
        all_refs: list[str] = []
        if reference_questions:
            all_refs = runner._collect_references(all_questions, reference_questions)

        # Build context with experiment data
        context = PipelineContext()
        context.questions = all_questions
        context.references = all_refs
        context.experiment_result = experiment_result
        context.metadata = metadata or {}
        context.metadata["experiment_id"] = experiment_config.experiment_id

        # Add quality reports from experiment conditions
        for condition in runner.condition_results:
            context.quality_reports.extend(condition.quality_reports)
            context.auto_metric_results.extend(condition.auto_metrics)

        # Add comparisons
        context.comparisons = runner.get_all_comparisons()

        # Run report and export stages
        report_stage = ReportGenerationStage(self.config)
        context = report_stage.run(context)

        export_stage = ExportStage(self.config)
        context = export_stage.run(context)

        # Save experiment results
        runner.save_results(experiment_result, self.config.output_dir)

        return context

    # -----------------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------------

    def evaluate_questions_only(
        self,
        questions: list[GeneratedQuestion],
        references: list[str] | None = None,
    ) -> EvaluationReport:
        """Quick evaluation of questions without RAG or experiment."""
        context = self.run(questions, references)
        if context.report is None:
            raise RuntimeError("Report generation failed")
        return context.report

    def evaluate_rag_only(
        self,
        queries: list[RAGQuery],
        results: list[RAGResult],
    ) -> RAGEvaluationReport:
        """Quick evaluation of RAG pipeline only."""
        config = PipelineConfig(
            enable_rag_eval=True,
            k_values=self.config.k_values,
            output_dir=str(self.config.output_dir),
        )
        pipeline = EvaluationPipeline(config)
        context = pipeline.run(
            questions=[],
            rag_queries=queries,
            rag_results=results,
        )
        if context.rag_report is None:
            raise RuntimeError("RAG evaluation failed")
        return context.rag_report
