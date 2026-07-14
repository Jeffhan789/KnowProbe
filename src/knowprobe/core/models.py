"""Core data models for KnowProbe."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    FACTUAL = "factual"
    SCHEMA = "schema"
    COMPOSITE = "composite"


class PromptStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "cot"
    COT = "cot"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    CLAUDE = "claude"


class KnowledgeInput(BaseModel):
    """Structured knowledge input for question generation."""

    source_id: str = Field(description="Unique identifier for the knowledge source")
    input_type: str = Field(default="triple", description="triple | schema | text | entity")
    content: str = Field(description="Raw knowledge content")
    structured: dict[str, Any] = Field(default_factory=dict, description="Parsed structured form")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedQuestion(BaseModel):
    """A generated question with provenance."""

    id: str | None = None
    question_text: str
    knowledge_input: KnowledgeInput
    question_type: QuestionType
    prompt_strategy: PromptStrategy
    model_name: str
    model_provider: ModelProvider
    generation_params: dict[str, Any] = Field(default_factory=dict)
    raw_output: str = ""
    confidence: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationResult(BaseModel):
    """Evaluation result for a generated question."""

    question_id: str
    metric_name: str
    score: float
    details: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""

    experiment_id: str
    name: str
    description: str = ""
    models: list[str]
    prompt_strategies: list[PromptStrategy]
    question_types: list[QuestionType]
    evaluation_metrics: list[str]
    knowledge_sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExperimentResult(BaseModel):
    """Result of an experiment run."""

    experiment_id: str
    config: ExperimentConfig
    questions: list[GeneratedQuestion]
    evaluations: list[EvaluationResult]
    summary: dict[str, Any] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class RAGDocument(BaseModel):
    """Document for RAG evaluation."""

    doc_id: str
    title: str = ""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGChunk(BaseModel):
    """Chunked segment of a RAG document for embedding and retrieval."""

    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Result of a single document retrieval operation."""

    chunk: RAGChunk
    score: float = Field(description="Similarity score (higher = more relevant)")
    rank: int = Field(default=0, description="Rank in retrieval results (1-based)")


class RAGQuery(BaseModel):
    """Query for RAG evaluation."""

    query_id: str
    query_text: str
    expected_answer: str = ""
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseModel):
    """Result of a RAG pipeline evaluation."""

    query_id: str
    retrieved_docs: list[RAGDocument]
    generated_answer: str
    evaluation_scores: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0


class RAGMetrics(BaseModel):
    """Comprehensive RAG evaluation metrics."""

    query_id: str
    retrieval: dict[str, float] = Field(default_factory=dict, description="Retrieval metrics")
    generation: dict[str, float] = Field(default_factory=dict, description="Generation metrics")
    end_to_end: dict[str, float] = Field(default_factory=dict, description="End-to-end metrics")
    latency_ms: float = 0.0
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class RAGPipelineResult(BaseModel):
    """Complete result of a RAG pipeline run including all intermediate outputs."""

    query: RAGQuery
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    generated_answer: str = ""
    metrics: RAGMetrics | None = None
    raw_prompt: str = ""
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RAGBenchmarkResult(BaseModel):
    """Result of a full RAG benchmark over multiple queries."""

    benchmark_id: str
    pipeline_name: str
    num_queries: int
    metrics: list[RAGMetrics] = Field(default_factory=list)
    aggregate_scores: dict[str, float] = Field(default_factory=dict)
    per_query_results: list[RAGPipelineResult] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=datetime.utcnow)
