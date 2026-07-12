"""Base classes and exceptions for question generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from knowprobe.core.models import (
    GeneratedQuestion,
    KnowledgeInput,
    PromptStrategy,
    QuestionType,
)
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationError(Exception):
    """Raised when question generation fails.

    Attributes:
        message: Human-readable error description.
        details: Structured context for debugging (model, strategy, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class ModelUnavailableError(Exception):
    """Raised when the configured model is not accessible.

    This can happen due to network issues, the model not being loaded,
    or insufficient compute resources.
    """

    def __init__(self, message: str, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model


class PromptBuildError(Exception):
    """Raised when prompt template rendering fails."""

    def __init__(self, message: str, template_key: str = "") -> None:
        super().__init__(message)
        self.template_key = template_key


class BaseQuestionGenerator(ABC):
    """Abstract base class defining the question generator contract.

    All concrete generators (local Transformers, Ollama, API-based) must
    implement this interface. The design supports both single-item generation
    for interactive use and batch generation for efficient experiment runs.

    Usage:
        async with MyGenerator(...) as gen:
            question = await gen.generate(knowledge, QuestionType.FACTUAL, ...)
    """

    def __init__(
        self,
        model_name: str,
        model_provider: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the generator with model configuration.

        Args:
            model_name: Identifier of the model (e.g. "llama3.1:8b").
            model_provider: Provider tag (e.g. "ollama", "openai").
            **kwargs: Provider-specific configuration overrides.
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self._config = kwargs
        self._initialized = False
        self._logger = get_logger(f"{self.__class__.__name__}.{model_name}")

    @property
    def is_initialized(self) -> bool:
        """Return whether the generator has been initialized."""
        return self._initialized

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the generator (load model, warm up cache, etc.).

        Must be called before ``generate()`` or ``generate_batch()``.
        Raises ModelUnavailableError if the backend cannot be reached.
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release all resources (GPU memory, HTTP clients, etc.).

        Safe to call multiple times; subsequent calls are no-ops.
        """
        ...

    # ------------------------------------------------------------------ #
    # Generation contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def generate(
        self,
        knowledge: KnowledgeInput,
        question_type: QuestionType,
        prompt_strategy: PromptStrategy,
        **kwargs: Any,
    ) -> GeneratedQuestion:
        """Generate a single question from knowledge input.

        Args:
            knowledge: Structured knowledge input with content and metadata.
            question_type: Type of question to generate (factual / schema).
            prompt_strategy: Prompting strategy to apply.
            **kwargs: Additional generation parameters (temperature, etc.).

        Returns:
            GeneratedQuestion with full provenance and raw model output.

        Raises:
            GenerationError: If generation fails after retries.
            ModelUnavailableError: If the model is not accessible.
            RuntimeError: If called before ``initialize()``.
        """
        ...

    @abstractmethod
    async def generate_batch(
        self,
        knowledges: list[KnowledgeInput],
        question_type: QuestionType,
        prompt_strategy: PromptStrategy,
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:
        """Generate questions in batch for efficient experiment runs.

        Args:
            knowledges: List of knowledge inputs.
            question_type: Type of question to generate.
            prompt_strategy: Prompting strategy to apply.
            **kwargs: Additional generation parameters.

        Returns:
            List of GeneratedQuestion, preserving input order.
        """
        ...

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check if the generator is healthy.

        Returns:
            Dict with at least ``status`` ("ok" | "degraded" | "unavailable")
            and provider-specific details (latency, GPU utilisation, etc.).
        """
        ...

    # ------------------------------------------------------------------ #
    # Async context manager
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> BaseQuestionGenerator:
        """Async context manager entry — auto-initialise."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit — auto-shutdown."""
        await self.shutdown()
