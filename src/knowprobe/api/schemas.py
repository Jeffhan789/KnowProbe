"""API request/response schemas extending core data models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from knowprobe.core.models import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    GeneratedQuestion,
    KnowledgeInput,
    PromptStrategy,
    QuestionType,
    RAGDocument,
    RAGQuery,
    RAGResult,
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    environment: str


# ---------------------------------------------------------------------------
# Generation API
# ---------------------------------------------------------------------------
class GenerateQuestionRequest(BaseModel):
    """Request body for single question generation."""

    knowledge: KnowledgeInput
    question_type: QuestionType = QuestionType.FACTUAL
    prompt_strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT
    model_name: str = "llama3.1:8b"
    generation_params: dict[str, Any] = Field(default_factory=dict)


class GenerateBatchRequest(BaseModel):
    """Request body for batch question generation."""

    knowledge_items: list[KnowledgeInput]
    question_type: QuestionType = QuestionType.FACTUAL
    prompt_strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT
    model_name: str = "llama3.1:8b"
    generation_params: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    """Response wrapper for a generated question."""

    success: bool
    data: GeneratedQuestion | None = None
    error: str | None = None
    latency_ms: float


class BatchGenerationResponse(BaseModel):
    """Response wrapper for batch question generation."""

    success: bool
    results: list[GenerationResponse]
    total_count: int
    success_count: int
    failed_count: int
    total_latency_ms: float


class StrategyInfo(BaseModel):
    """Information about a prompt strategy."""

    name: str
    value: str
    description: str


class TypeInfo(BaseModel):
    """Information about a question type."""

    name: str
    value: str
    description: str


class AvailableStrategiesResponse(BaseModel):
    """List available prompt strategies."""

    strategies: list[StrategyInfo]


class AvailableTypesResponse(BaseModel):
    """List available question types."""

    types: list[TypeInfo]


# ---------------------------------------------------------------------------
# Evaluation API
# ---------------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    """Request body for evaluating a generated question."""

    question: GeneratedQuestion
    reference_question: str | None = None
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge"])


class EvaluateBatchRequest(BaseModel):
    """Request body for batch evaluation."""

    questions: list[GeneratedQuestion]
    references: list[str] | None = None
    metrics: list[str] = Field(default_factory=lambda: ["bleu", "rouge"])


class EvaluationResponse(BaseModel):
    """Response wrapper for evaluation results."""

    success: bool
    question_id: str | None = None
    scores: list[EvaluationResult] = Field(default_factory=list)
    error: str | None = None
    latency_ms: float


class AvailableMetricsResponse(BaseModel):
    """List available evaluation metrics."""

    metrics: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Experiment API
# ---------------------------------------------------------------------------
class CreateExperimentRequest(ExperimentConfig):
    """Request to create a new experiment."""

    pass


class ExperimentResponse(BaseModel):
    """Response wrapper for experiment operations."""

    success: bool
    experiment_id: str
    data: ExperimentResult | ExperimentConfig | None = None
    error: str | None = None


class ExperimentListResponse(BaseModel):
    """Response for listing experiments."""

    experiments: list[ExperimentConfig]
    total: int


class RunExperimentRequest(BaseModel):
    """Request to run an experiment."""

    experiment_id: str
    dry_run: bool = False


# ---------------------------------------------------------------------------
# RAG API
# ---------------------------------------------------------------------------
class RAGQueryRequest(BaseModel):
    """Request body for RAG query."""

    query: RAGQuery
    documents: list[RAGDocument] = Field(default_factory=list)
    top_k: int = 5
    retriever_type: str = "dense"


class RAGQueryResponse(BaseModel):
    """Response wrapper for RAG query."""

    success: bool
    result: RAGResult | None = None
    error: str | None = None
    latency_ms: float


class RAGEvaluateRequest(BaseModel):
    """Request body for evaluating RAG results."""

    rag_result: RAGResult
    metrics: list[str] = Field(default_factory=lambda: ["retrieval_accuracy", "answer_relevance"])


class RAGEvaluationResponse(BaseModel):
    """Response wrapper for RAG evaluation."""

    success: bool
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Error Schemas
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str | None = None
    message: str
    type: str = "validation_error"


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
class PaginatedResponse(BaseModel):
    """Base paginated response."""

    total: int
    page: int
    per_page: int
    total_pages: int
    items: list[Any]
