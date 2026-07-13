"""Tests for the KnowProbe evaluation framework."""

from __future__ import annotations

import json

import numpy as np
import pytest

from knowprobe.core.models import (
    EvaluationResult,
    ExperimentConfig,
    GeneratedQuestion,
    KnowledgeInput,
    ModelProvider,
    PromptStrategy,
    QuestionType,
    RAGDocument,
    RAGQuery,
    RAGResult,
)
from knowprobe.evaluators import (
    EvaluationPipeline,
    EvaluationReporter,
    ExperimentRunner,
    MetricRegistry,
    QuestionEvaluator,
    RAGEvaluationReport,
    RAGEvaluator,
)
from knowprobe.evaluators.metrics import (
    BERTScoreMetric,
    BLEUMetric,
    DistinctNMetric,
    GrammarMetric,
    METEORMetric,
    ROUGEMetric,
    SelfBLEUMetric,
)
from knowprobe.evaluators.question_evaluator import (
    AnswerabilityEvaluator,
    FluencyEvaluator,
    QuestionQualityReport,
    RelevanceEvaluator,
    StructuralGroundingEvaluator,
    TypeConsistencyEvaluator,
)
from knowprobe.evaluators.rag_evaluator import (
    GenerationEvaluator,
    GenerationMetrics,
    RetrievalEvaluator,
    RetrievalMetrics,
)
from knowprobe.evaluators.reporter import EvaluationReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_knowledge() -> KnowledgeInput:
    return KnowledgeInput(
        source_id="test_kg_001",
        input_type="triple",
        content="Albert Einstein developed the theory of relativity in 1905.",
        structured={
            "subject": "Albert Einstein",
            "predicate": "developed",
            "object": "theory of relativity",
            "year": 1905,
        },
    )


@pytest.fixture
def sample_questions(sample_knowledge: KnowledgeInput) -> list[GeneratedQuestion]:
    return [
        GeneratedQuestion(
            id="q_001",
            question_text="What theory did Albert Einstein develop in 1905?",
            knowledge_input=sample_knowledge,
            question_type=QuestionType.FACTUAL,
            prompt_strategy=PromptStrategy.ZERO_SHOT,
            model_name="llama3.1:8b",
            model_provider=ModelProvider.OLLAMA,
        ),
        GeneratedQuestion(
            id="q_002",
            question_text="Who developed the theory of relativity?",
            knowledge_input=sample_knowledge,
            question_type=QuestionType.FACTUAL,
            prompt_strategy=PromptStrategy.FEW_SHOT,
            model_name="llama3.1:8b",
            model_provider=ModelProvider.OLLAMA,
        ),
        GeneratedQuestion(
            id="q_003",
            question_text="What type of property relates a scientist to their theory?",
            knowledge_input=sample_knowledge,
            question_type=QuestionType.SCHEMA,
            prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            model_name="qwen2.5:7b",
            model_provider=ModelProvider.OLLAMA,
        ),
    ]


@pytest.fixture
def sample_references() -> list[str]:
    return [
        "What theory of physics did Albert Einstein formulate in 1905?",
        "Which scientist developed the theory of relativity?",
        "What relation connects a scientist to their scientific theory?",
    ]


@pytest.fixture
def sample_rag_query() -> RAGQuery:
    return RAGQuery(
        query_id="rag_q_001",
        query_text="What did Einstein develop?",
        expected_answer="The theory of relativity",
        relevant_doc_ids=["doc_001", "doc_002"],
    )


@pytest.fixture
def sample_rag_result() -> RAGResult:
    return RAGResult(
        query_id="rag_q_001",
        retrieved_docs=[
            RAGDocument(
                doc_id="doc_001",
                title="Einstein's Work",
                content="Albert Einstein developed the theory of relativity in 1905.",
            ),
            RAGDocument(
                doc_id="doc_002",
                title="Physics History",
                content="The theory of relativity was a groundbreaking work.",
            ),
            RAGDocument(
                doc_id="doc_003",
                title="Newton",
                content="Isaac Newton discovered gravity.",
            ),
        ],
        generated_answer="Albert Einstein developed the theory of relativity.",
        evaluation_scores={},
        latency_ms=150.0,
    )


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


class TestBLEUMetric:
    def test_bleu_computation(self, sample_questions, sample_references):
        bleu = BLEUMetric(max_order=4)
        predictions = [q.question_text for q in sample_questions]
        scores = bleu.compute(predictions, sample_references)
        assert len(scores) > 0
        assert 0.0 <= scores[0].value <= 1.0

    def test_bleu_empty_input(self):
        bleu = BLEUMetric()
        scores = bleu.compute([], [])
        assert scores == []

    def test_bleu_metric_registry(self):
        metric = MetricRegistry.get("bleu")
        assert isinstance(metric, BLEUMetric)


class TestROUGEMetric:
    def test_rouge_computation(self, sample_questions, sample_references):
        rouge = ROUGEMetric()
        predictions = [q.question_text for q in sample_questions]
        scores = rouge.compute(predictions, sample_references)
        assert len(scores) > 0
        for s in scores:
            assert 0.0 <= s.value <= 1.0

    def test_rouge_lcs_naive(self):
        rouge = ROUGEMetric()
        lcs = rouge._lcs_length(["a", "b", "c"], ["a", "b", "c"])
        assert lcs == 3
        lcs = rouge._lcs_length(["a", "b", "c"], ["a", "c"])
        assert lcs == 2


class TestMETEORMetric:
    def test_meteor_computation(self, sample_questions, sample_references):
        meteor = METEORMetric()
        predictions = [q.question_text for q in sample_questions]
        scores = meteor.compute(predictions, sample_references)
        assert len(scores) > 0
        assert 0.0 <= scores[0].value <= 1.0


class TestGrammarMetric:
    def test_grammar_computation(self, sample_questions):
        grammar = GrammarMetric()
        predictions = [q.question_text for q in sample_questions]
        scores = grammar.compute(predictions)
        assert len(scores) > 0
        assert 0.0 <= scores[0].value <= 1.0

    def test_grammar_repeated_punctuation(self):
        grammar = GrammarMetric()
        scores = grammar.compute(["What is this??!"])
        assert scores[0].value < 1.0  # Should penalize repeated punctuation


class TestSelfBLEUMetric:
    def test_self_bleu_diversity(self, sample_questions):
        self_bleu = SelfBLEUMetric(max_order=2)
        predictions = [q.question_text for q in sample_questions]
        scores = self_bleu.compute(predictions)
        assert len(scores) > 0
        assert 0.0 <= scores[0].value <= 1.0
        assert "diversity_score" in scores[0].details

    def test_self_bleu_insufficient_samples(self):
        self_bleu = SelfBLEUMetric()
        scores = self_bleu.compute(["single question"])
        assert scores[0].value == 0.0
        assert scores[0].details.get("reason") == "insufficient_samples"


class TestDistinctNMetric:
    def test_distinct_n(self, sample_questions):
        distinct = DistinctNMetric(n=2)
        predictions = [q.question_text for q in sample_questions]
        scores = distinct.compute(predictions)
        assert len(scores) > 0
        assert 0.0 <= scores[0].value <= 1.0


class TestBERTScoreMetric:
    def test_bertscore_computation(self, sample_questions, sample_references):
        # Skip if bert-score is not available or too slow for unit tests
        pytest.importorskip("bert_score")
        bert = BERTScoreMetric(
            model_type="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        predictions = [q.question_text for q in sample_questions[:1]]
        refs = sample_references[:1]
        scores = bert.compute(predictions, refs)
        assert len(scores) > 0
        assert 0.0 <= scores[0].value <= 1.0


class TestMetricRegistry:
    def test_list_metrics(self):
        metrics = MetricRegistry.list_metrics()
        assert "bleu" in metrics
        assert "rouge" in metrics
        assert "bert_score" in metrics

    def test_get_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            MetricRegistry.get("nonexistent_metric")

    def test_build_metrics(self):
        metrics = MetricRegistry.build_metrics(["bleu", "rouge"])
        assert len(metrics) == 2
        assert isinstance(metrics[0], BLEUMetric)
        assert isinstance(metrics[1], ROUGEMetric)


# ---------------------------------------------------------------------------
# Question evaluator tests
# ---------------------------------------------------------------------------


class TestQuestionEvaluator:
    def test_evaluate_single(self, sample_questions):
        evaluator = QuestionEvaluator()
        report = evaluator.evaluate_single(sample_questions[0])
        assert isinstance(report, QuestionQualityReport)
        assert report.question_id == "q_001"
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.dimensions) > 0

    def test_evaluate_batch(self, sample_questions):
        evaluator = QuestionEvaluator()
        reports = evaluator.evaluate_batch(sample_questions)
        assert len(reports) == len(sample_questions)
        for report in reports:
            assert 0.0 <= report.overall_score <= 1.0

    def test_compute_auto_metrics(self, sample_questions, sample_references):
        evaluator = QuestionEvaluator()
        results = evaluator.compute_auto_metrics(
            questions=sample_questions[:1],
            references=sample_references[:1],
            metric_names=["bleu", "rouge"],
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, EvaluationResult)
            assert 0.0 <= result.score <= 1.0

    def test_get_dimension_scores(self, sample_questions):
        evaluator = QuestionEvaluator()
        reports = evaluator.evaluate_batch(sample_questions)
        scores = evaluator.get_dimension_scores(reports)
        assert "relevance" in scores
        assert "fluency" in scores
        assert len(scores["relevance"]) == len(sample_questions)


class TestRelevanceEvaluator:
    def test_relevance_score(self, sample_questions):
        evaluator = RelevanceEvaluator()
        dim = evaluator.evaluate(sample_questions[0])
        assert dim.name == "relevance"
        assert 0.0 <= dim.score <= 1.0


class TestTypeConsistencyEvaluator:
    def test_factual_question(self, sample_questions):
        evaluator = TypeConsistencyEvaluator()
        dim = evaluator.evaluate(sample_questions[0])  # FACTUAL question
        assert dim.name == "type_consistency"
        assert 0.0 <= dim.score <= 1.0
        assert dim.details["expected_type"] == "factual"

    def test_schema_question(self, sample_questions):
        evaluator = TypeConsistencyEvaluator()
        dim = evaluator.evaluate(sample_questions[2])  # SCHEMA question
        assert dim.name == "type_consistency"
        assert 0.0 <= dim.score <= 1.0
        assert dim.details["expected_type"] == "schema"


class TestAnswerabilityEvaluator:
    def test_answerability(self, sample_questions):
        evaluator = AnswerabilityEvaluator()
        dim = evaluator.evaluate(sample_questions[0])
        assert dim.name == "answerability"
        assert 0.0 <= dim.score <= 1.0
        assert "term_coverage_ratio" in dim.details


class TestFluencyEvaluator:
    def test_fluency(self, sample_questions):
        evaluator = FluencyEvaluator()
        dim = evaluator.evaluate(sample_questions[0])
        assert dim.name == "fluency"
        assert 0.0 <= dim.score <= 1.0
        assert "grammar_score" in dim.details


class TestStructuralGroundingEvaluator:
    def test_structural_grounding(self, sample_questions):
        evaluator = StructuralGroundingEvaluator()
        dim = evaluator.evaluate(sample_questions[2])  # SCHEMA question
        assert dim.name == "structural_grounding"
        assert 0.0 <= dim.score <= 1.0


# ---------------------------------------------------------------------------
# RAG evaluator tests
# ---------------------------------------------------------------------------


class TestRetrievalEvaluator:
    def test_precision_at_k(self, sample_rag_query, sample_rag_result):
        evaluator = RetrievalEvaluator(k_values=[1, 3])
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.precision_at_k[1] == 1.0  # Top doc is relevant
        assert 0.0 <= metrics.precision_at_k[3] <= 1.0

    def test_recall_at_k(self, sample_rag_query, sample_rag_result):
        evaluator = RetrievalEvaluator(k_values=[1, 3])
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert metrics.recall_at_k[1] == 0.5  # 1 relevant out of 2
        assert metrics.recall_at_k[3] == 1.0  # All relevant retrieved

    def test_mrr(self, sample_rag_query, sample_rag_result):
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert metrics.mrr == 1.0  # First doc is relevant

    def test_ndcg(self, sample_rag_query, sample_rag_result):
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert 0.0 <= metrics.ndcg_at_k[1] <= 1.0

    def test_batch_evaluation(self, sample_rag_query, sample_rag_result):
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate_batch([sample_rag_query], [sample_rag_result])
        assert len(metrics) == 1

    def test_empty_relevant_docs(self, sample_rag_result):
        query = RAGQuery(
            query_id="empty",
            query_text="test",
            relevant_doc_ids=[],
        )
        evaluator = RetrievalEvaluator()
        metrics = evaluator.evaluate_single(query, sample_rag_result)
        assert metrics.precision_at_k == {}


class TestGenerationEvaluator:
    def test_faithfulness(self, sample_rag_query, sample_rag_result):
        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert isinstance(metrics, GenerationMetrics)
        assert 0.0 <= metrics.faithfulness <= 1.0

    def test_answer_relevance(self, sample_rag_query, sample_rag_result):
        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert 0.0 <= metrics.answer_relevance <= 1.0

    def test_context_precision(self, sample_rag_query, sample_rag_result):
        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert 0.0 <= metrics.context_precision <= 1.0

    def test_answer_bleu_with_expected(self, sample_rag_query, sample_rag_result):
        # When expected_answer is provided
        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate_single(sample_rag_query, sample_rag_result)
        assert metrics.answer_bleu >= 0.0

    def test_batch_evaluation(self, sample_rag_query, sample_rag_result):
        evaluator = GenerationEvaluator()
        metrics = evaluator.evaluate_batch([sample_rag_query], [sample_rag_result])
        assert len(metrics) == 1


class TestRAGEvaluator:
    def test_full_evaluation(self, sample_rag_query, sample_rag_result):
        evaluator = RAGEvaluator(k_values=[1, 3, 5])
        report = evaluator.evaluate(
            queries=[sample_rag_query],
            results=[sample_rag_result],
            run_id="test_run",
        )
        assert isinstance(report, RAGEvaluationReport)
        assert report.run_id == "test_run"
        assert len(report.retrieval_metrics) == 1
        assert len(report.generation_metrics) == 1
        assert "avg_mrr" in report.aggregate
        assert "avg_faithfulness" in report.aggregate

    def test_aggregate_computation(self, sample_rag_query, sample_rag_result):
        evaluator = RAGEvaluator()
        report = evaluator.evaluate([sample_rag_query], [sample_rag_result])
        assert 0.0 <= report.aggregate["avg_mrr"] <= 1.0
        assert 0.0 <= report.aggregate["avg_faithfulness"] <= 1.0


# ---------------------------------------------------------------------------
# Experiment runner tests
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_experiment_config(self):
        config = ExperimentConfig(
            experiment_id="test_exp_001",
            name="Test Experiment",
            models=["llama3.1:8b", "qwen2.5:7b"],
            prompt_strategies=[PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT],
            question_types=[QuestionType.FACTUAL],
            evaluation_metrics=["bleu", "rouge"],
        )
        assert config.experiment_id == "test_exp_001"
        assert len(config.models) == 2

    def test_run_experiment_no_generator(self, sample_knowledge):
        config = ExperimentConfig(
            experiment_id="test_exp_002",
            name="Test Experiment",
            models=["llama3.1:8b"],
            prompt_strategies=[PromptStrategy.ZERO_SHOT],
            question_types=[QuestionType.FACTUAL],
            evaluation_metrics=["bleu"],
        )
        runner = ExperimentRunner(config=config)
        result = runner.run(
            knowledge_inputs=[sample_knowledge],
        )
        assert result.experiment_id == "test_exp_002"
        assert len(result.questions) > 0
        assert len(runner.condition_results) == 1

    def test_compare_strategies(self, sample_knowledge):
        config = ExperimentConfig(
            experiment_id="test_exp_003",
            name="Strategy Comparison",
            models=["llama3.1:8b"],
            prompt_strategies=[PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT],
            question_types=[QuestionType.FACTUAL],
            evaluation_metrics=["bleu"],
        )
        runner = ExperimentRunner(config=config)
        runner.run(knowledge_inputs=[sample_knowledge] * 3)
        comp = runner.compare_strategies(
            PromptStrategy.ZERO_SHOT,
            PromptStrategy.FEW_SHOT,
        )
        # May be None if insufficient data, but should not error
        if comp is not None:
            assert comp.baseline_condition == "zero_shot"
            assert comp.comparison_condition == "few_shot"

    def test_save_and_load_results(self, sample_knowledge, tmp_path):
        config = ExperimentConfig(
            experiment_id="test_exp_004",
            name="Persistence Test",
            models=["llama3.1:8b"],
            prompt_strategies=[PromptStrategy.ZERO_SHOT],
            question_types=[QuestionType.FACTUAL],
            evaluation_metrics=["bleu"],
        )
        runner = ExperimentRunner(config=config)
        result = runner.run(knowledge_inputs=[sample_knowledge])
        filepath = runner.save_results(result, tmp_path)
        assert filepath.exists()
        loaded = runner.load_results(filepath)
        assert loaded["experiment_id"] == "test_exp_004"


# ---------------------------------------------------------------------------
# Reporter tests
# ---------------------------------------------------------------------------


class TestEvaluationReporter:
    def test_build_report(self):
        reporter = EvaluationReporter()
        raw_data = {
            "model_a": {"bleu": [0.5, 0.6, 0.7], "rouge": [0.4, 0.5, 0.6]},
            "model_b": {"bleu": [0.3, 0.4, 0.5], "rouge": [0.2, 0.3, 0.4]},
        }
        report = reporter.build_report(
            title="Test Report",
            raw_data=raw_data,
        )
        assert isinstance(report, EvaluationReport)
        assert report.title == "Test Report"
        assert len(report.statistics) > 0
        assert len(report.rankings) > 0

    def test_markdown_export(self, tmp_path):
        reporter = EvaluationReporter()
        raw_data = {
            "model_a": {"bleu": [0.5, 0.6, 0.7]},
            "model_b": {"bleu": [0.3, 0.4, 0.5]},
        }
        report = reporter.build_report("Test", raw_data)
        md_path = reporter.export_markdown(report, tmp_path)
        assert md_path.exists()
        content = md_path.read_text()
        assert "# Test" in content
        assert "| Group" in content

    def test_json_export(self, tmp_path):
        reporter = EvaluationReporter()
        raw_data = {"model_a": {"bleu": [0.5, 0.6, 0.7]}}
        report = reporter.build_report("Test", raw_data)
        json_path = reporter.export_json(report, tmp_path)
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["title"] == "Test"

    def test_csv_export(self, tmp_path):
        reporter = EvaluationReporter()
        raw_data = {"model_a": {"bleu": [0.5, 0.6, 0.7]}}
        report = reporter.build_report("Test", raw_data)
        csv_path = reporter.export_csv(report, tmp_path)
        assert csv_path.exists()
        # Check CSV has headers
        content = csv_path.read_text()
        assert "group" in content
        assert "metric" in content

    def test_thesis_format(self):
        reporter = EvaluationReporter()
        raw_data = {
            "llama3.1:8b": {"bleu": [0.5, 0.6, 0.7]},
            "qwen2.5:7b": {"bleu": [0.4, 0.5, 0.6]},
        }
        comparisons = reporter.compute_pairwise_comparisons(
            {"llama3.1:8b": [0.5, 0.6, 0.7], "qwen2.5:7b": [0.4, 0.5, 0.6]},
            metric_name="bleu",
        )
        report = reporter.build_report("Test", raw_data, comparisons=comparisons)
        latex = reporter.format_for_thesis(report, section="results")
        assert "\\section{Evaluation Results}" in latex
        assert "\\begin{table}" in latex

    def test_confidence_interval(self):
        reporter = EvaluationReporter()
        data = [0.5, 0.6, 0.7, 0.8, 0.9]
        lower, upper = reporter.compute_confidence_interval(data, confidence=0.95)
        assert lower < upper
        assert lower < np.mean(data) < upper

    def test_effect_size(self):
        reporter = EvaluationReporter()
        g1 = [0.5, 0.6, 0.7]
        g2 = [0.3, 0.4, 0.5]
        d = reporter.compute_effect_size(g1, g2)
        assert d != 0.0

    def test_pairwise_comparisons(self):
        reporter = EvaluationReporter()
        groups = {
            "zero_shot": [0.5, 0.6, 0.7],
            "few_shot": [0.6, 0.7, 0.8],
            "cot": [0.7, 0.8, 0.9],
        }
        comps = reporter.compute_pairwise_comparisons(groups, metric_name="bleu")
        assert len(comps) == 3  # 3 choose 2
        # At least one should be significant given the clear separation
        assert any(c["significant"] for c in comps)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestEvaluationPipeline:
    def test_run_pipeline(self, sample_questions, sample_references):
        from knowprobe.evaluators.pipeline import PipelineConfig

        config = PipelineConfig(
            auto_metrics=["bleu", "rouge"],
            export_formats=["json"],
        )
        pipeline = EvaluationPipeline(config)
        context = pipeline.run(
            questions=sample_questions,
            references=sample_references,
        )
        assert context.report is not None
        assert len(context.quality_reports) == len(sample_questions)
        assert len(context.auto_metric_results) > 0

    def test_evaluate_questions_only(self, sample_questions, sample_references):
        pipeline = EvaluationPipeline()
        report = pipeline.evaluate_questions_only(
            questions=sample_questions,
            references=sample_references,
        )
        assert isinstance(report, EvaluationReport)
        assert len(report.statistics) > 0

    def test_pipeline_context(self, sample_questions):
        from knowprobe.evaluators.pipeline import PipelineContext

        ctx = PipelineContext()
        ctx.questions = sample_questions
        d = ctx.to_dict()
        assert d["num_questions"] == len(sample_questions)

    def test_run_experiment_pipeline(self, sample_knowledge):
        from knowprobe.evaluators.pipeline import PipelineConfig

        config = ExperimentConfig(
            experiment_id="pipeline_test_001",
            name="Pipeline Test",
            models=["llama3.1:8b"],
            prompt_strategies=[PromptStrategy.ZERO_SHOT],
            question_types=[QuestionType.FACTUAL],
            evaluation_metrics=["bleu"],
        )
        pipeline_config = PipelineConfig(export_formats=["json"])
        pipeline = EvaluationPipeline(pipeline_config)
        context = pipeline.run_experiment(
            experiment_config=config,
            knowledge_inputs=[sample_knowledge] * 2,
        )
        assert context.experiment_result is not None
        assert context.report is not None


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_end_to_end_evaluation(self, sample_questions, sample_references, tmp_path):
        """Full end-to-end evaluation workflow."""
        from knowprobe.evaluators.pipeline import PipelineConfig

        config = PipelineConfig(
            auto_metrics=["bleu", "rouge", "grammar"],
            export_formats=["json", "csv", "markdown"],
            output_dir=str(tmp_path),
        )
        pipeline = EvaluationPipeline(config)
        context = pipeline.run(
            questions=sample_questions,
            references=sample_references,
            metadata={"test": True, "dataset": "sample"},
        )

        assert context.report is not None
        assert len(context.quality_reports) == 3
        assert len(context.auto_metric_results) > 0

        # Check export files
        assert (tmp_path / "pipeline_context.json").exists()

    def test_rag_end_to_end(self, sample_rag_query, sample_rag_result, tmp_path):
        from knowprobe.evaluators.pipeline import PipelineConfig

        config = PipelineConfig(
            enable_rag_eval=True,
            k_values=[1, 3, 5],
            export_formats=["json"],
            output_dir=str(tmp_path),
        )
        pipeline = EvaluationPipeline(config)
        rag_report = pipeline.evaluate_rag_only(
            queries=[sample_rag_query],
            results=[sample_rag_result],
        )
        assert isinstance(rag_report, RAGEvaluationReport)
        assert len(rag_report.retrieval_metrics) == 1
        assert len(rag_report.generation_metrics) == 1
        assert rag_report.aggregate["avg_mrr"] > 0

    def test_experiment_comparison(self, sample_knowledge, tmp_path):
        """Full experiment with comparison and reporting."""
        config = ExperimentConfig(
            experiment_id="integration_test_001",
            name="Integration Test",
            models=["llama3.1:8b", "qwen2.5:7b"],
            prompt_strategies=[
                PromptStrategy.ZERO_SHOT,
                PromptStrategy.FEW_SHOT,
            ],
            question_types=[QuestionType.FACTUAL],
            evaluation_metrics=["bleu", "rouge"],
        )
        from knowprobe.evaluators.pipeline import PipelineConfig

        pipeline_config = PipelineConfig(
            export_formats=["json", "markdown"],
            output_dir=str(tmp_path),
        )
        pipeline = EvaluationPipeline(pipeline_config)
        context = pipeline.run_experiment(
            experiment_config=config,
            knowledge_inputs=[sample_knowledge] * 3,
        )

        assert context.experiment_result is not None
        assert len(context.comparisons) > 0
        assert context.report is not None

        # Check for model comparison
        model_comps = [
            c
            for c in context.comparisons
            if c.baseline_condition == "llama3.1:8b" or c.comparison_condition == "llama3.1:8b"
        ]
        assert len(model_comps) > 0

    def test_report_content(self, sample_questions, sample_references):
        """Verify report contains expected sections."""
        pipeline = EvaluationPipeline()
        report = pipeline.evaluate_questions_only(
            questions=sample_questions,
            references=sample_references,
        )
        md = report.to_markdown()
        assert "## Descriptive Statistics" in md or "## Rankings" in md or "# " in md

        # JSON should be serializable
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "title" in data
        assert "statistics" in data
