"""Database models for KnowProbe."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from knowprobe.core.config import get_settings


class Base(DeclarativeBase):
    """Base declarative class."""

    type_annotation_map = {dict[str, Any]: JSON}


class ExperimentRecord(Base):
    """Persisted experiment record."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    models: Mapped[list[str]] = mapped_column(JSON, default=list)
    prompt_strategies: Mapped[list[str]] = mapped_column(JSON, default=list)
    question_types: Mapped[list[str]] = mapped_column(JSON, default=list)
    evaluation_metrics: Mapped[list[str]] = mapped_column(JSON, default=list)
    knowledge_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class GeneratedQuestionRecord(Base):
    """Persisted generated question."""

    __tablename__ = "generated_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input_type: Mapped[str] = mapped_column(String(50), default="triple")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    structured: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    question_type: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    generation_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    raw_output: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EvaluationResultRecord(Base):
    """Persisted evaluation result."""

    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RAGRunRecord(Base):
    """Persisted RAG evaluation run."""

    __tablename__ = "rag_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    generated_answer: Mapped[str] = mapped_column(Text, default="")
    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generation_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def get_engine():
    """Create SQLAlchemy engine from settings."""
    settings = get_settings()
    return create_engine(settings.database.url, echo=settings.database.echo)


def get_session_factory():
    """Create session factory."""
    return sessionmaker(bind=get_engine())


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
