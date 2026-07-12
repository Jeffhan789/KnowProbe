"""Evaluation framework for KnowProbe.

Provides:
- Automatic metrics (BLEU, ROUGE, METEOR, BERTScore, Self-BLEU, Distinct-N, Grammar)
- Question quality evaluation (relevance, type consistency, answerability, fluency, structural grounding)
- RAG pipeline evaluation (retrieval and generation metrics)
- Experiment runner with statistical comparison
- Reporting (CSV, JSON, Markdown, LaTeX)
- Evaluation pipeline orchestration

Usage:
    from knowprobe.evaluators import (
        EvaluationPipeline,
        QuestionEvaluator,
        RAGEvaluator,
        ExperimentRunner,
        EvaluationReporter,
    )
    from knowprobe.evaluators.metrics import MetricRegistry

    # Evaluate a batch of questions
    evaluator = QuestionEvaluator()
    reports = evaluator.evaluate_batch(questions)

    # Run full pipeline
    pipeline = EvaluationPipeline()
    context = pipeline.run(questions, references)
    print(context.report.to_markdown())
"""

from knowprobe.evaluators.experiment_runner import (
    ComparativeAnalysis,
    ConditionResult,
    ExperimentRunner,
)
from knowprobe.evaluators.metrics import (
    AggregateScore,
    BaseMetric,
    BLEUMetric,
    BERTScoreMetric,
    DistinctNMetric,
    GrammarMetric,
    METEORMetric,
    MetricRegistry,
    MetricScore,
    ROUGEMetric,
    SelfBLEUMetric,
)
from knowprobe.evaluators.pipeline import (
    EvaluationPipeline,
    PipelineConfig,
    PipelineContext,
)
from knowprobe.evaluators.question_evaluator import (
    AnswerabilityEvaluator,
    FluencyEvaluator,
    QuestionEvaluator,
    QuestionQualityReport,
    QualityDimension,
    RelevanceEvaluator,
    StructuralGroundingEvaluator,
    TypeConsistencyEvaluator,
)
from knowprobe.evaluators.rag_evaluator import (
    GenerationEvaluator,
    GenerationMetrics,
    RAGEvaluationReport,
    RAGEvaluator,
    RetrievalEvaluator,
    RetrievalMetrics,
)
from knowprobe.evaluators.reporter import (
    ComparisonRow,
    EvaluationReport,
    EvaluationReporter,
    StatisticRow,
)

__all__ = [
    # Metrics
    "BaseMetric",
    "BLEUMetric",
    "ROUGEMetric",
    "METEORMetric",
    "BERTScoreMetric",
    "SelfBLEUMetric",
    "DistinctNMetric",
    "GrammarMetric",
    "MetricRegistry",
    "MetricScore",
    "AggregateScore",
    # Question evaluation
    "QuestionEvaluator",
    "QuestionQualityReport",
    "QualityDimension",
    "RelevanceEvaluator",
    "TypeConsistencyEvaluator",
    "AnswerabilityEvaluator",
    "FluencyEvaluator",
    "StructuralGroundingEvaluator",
    # RAG evaluation
    "RAGEvaluator",
    "RAGEvaluationReport",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "GenerationEvaluator",
    "GenerationMetrics",
    # Experiment runner
    "ExperimentRunner",
    "ConditionResult",
    "ComparativeAnalysis",
    # Reporter
    "EvaluationReporter",
    "EvaluationReport",
    "StatisticRow",
    "ComparisonRow",
    # Pipeline
    "EvaluationPipeline",
    "PipelineConfig",
    "PipelineContext",
]
