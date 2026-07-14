"""Base LLM client abstract interface."""

import time
from abc import ABC, abstractmethod
from typing import Any

from knowprobe.utils.logging import get_logger

from .exceptions import LLMError, LLMResponseError
from .types import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationRequest,
    GenerationResponse,
    LLMMetadata,
    Message,
    Role,
    UsageInfo,
)

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    All concrete LLM providers must implement this interface.
    """

    def __init__(self, model: str, provider: str, **kwargs: Any) -> None:
        self.model = model
        self.provider = provider
        self._metadata: LLMMetadata | None = None
        self._logger = get_logger(f"{__name__}.{provider}")

    @property
    def metadata(self) -> LLMMetadata:
        """Get model metadata."""
        if self._metadata is None:
            self._metadata = LLMMetadata(id=self.model, provider=self.provider)
        return self._metadata

    @abstractmethod
    async def agenerate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text asynchronously.

        Args:
            request: Generation request with prompt and parameters.

        Returns:
            GenerationResponse with generated text and metadata.

        Raises:
            LLMError: On generation failure.
        """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text synchronously.

        Args:
            request: Generation request with prompt and parameters.

        Returns:
            GenerationResponse with generated text and metadata.

        Raises:
            LLMError: On generation failure.
        """

    async def abatch_generate(self, batch: BatchGenerationRequest) -> BatchGenerationResponse:
        """Generate multiple texts asynchronously.

        Default implementation processes sequentially. Override for
        provider-specific batching optimizations.

        Args:
            batch: Batch generation request.

        Returns:
            BatchGenerationResponse with all responses.
        """
        start = time.perf_counter()
        responses: list[GenerationResponse] = []

        for req in batch.requests:
            if batch.common_params is not None:
                req.params = batch.common_params
            try:
                response = await self.agenerate(req)
                responses.append(response)
            except LLMError as e:
                self._logger.error(
                    "batch_request_failed",
                    error=str(e),
                    model=req.model or self.model,
                )
                raise

        total_latency = (time.perf_counter() - start) * 1000
        return BatchGenerationResponse(responses=responses, total_latency_ms=total_latency)

    def batch_generate(self, batch: BatchGenerationRequest) -> BatchGenerationResponse:
        """Generate multiple texts synchronously.

        Default implementation processes sequentially.

        Args:
            batch: Batch generation request.

        Returns:
            BatchGenerationResponse with all responses.
        """
        start = time.perf_counter()
        responses: list[GenerationResponse] = []

        for req in batch.requests:
            if batch.common_params is not None:
                req.params = batch.common_params
            try:
                response = self.generate(req)
                responses.append(response)
            except LLMError as e:
                self._logger.error(
                    "batch_request_failed",
                    error=str(e),
                    model=req.model or self.model,
                )
                raise

        total_latency = (time.perf_counter() - start) * 1000
        return BatchGenerationResponse(responses=responses, total_latency_ms=total_latency)

    @abstractmethod
    async def ahealth_check(self) -> bool:
        """Check if the LLM service is healthy asynchronously.

        Returns:
            True if healthy, False otherwise.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM service is healthy synchronously.

        Returns:
            True if healthy, False otherwise.
        """

    def _build_messages(self, request: GenerationRequest) -> list[Message]:
        """Build message list from request.

        Handles both prompt-only and messages-based requests.
        """
        if request.messages:
            messages = request.messages.copy()
            if request.system_prompt and messages[0].role != Role.SYSTEM:
                messages.insert(0, Message(role=Role.SYSTEM, content=request.system_prompt))
            return messages

        messages = []
        if request.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=request.system_prompt))
        messages.append(Message(role=Role.USER, content=request.prompt))
        return messages

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build single prompt string from request.

        Used for non-chat models.
        """
        if request.prompt:
            return request.prompt

        parts: list[str] = []
        if request.system_prompt:
            parts.append(f"System: {request.system_prompt}")
        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                parts.append(f"System: {msg.content}")
            elif msg.role == Role.USER:
                parts.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"Assistant: {msg.content}")
        return "\n\n".join(parts)

    def _create_response(
        self,
        text: str,
        finish_reason: str | None = None,
        usage: UsageInfo | None = None,
        latency_ms: float = 0.0,
        raw_response: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        """Create a standardized GenerationResponse."""
        return GenerationResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            finish_reason=finish_reason,
            usage=usage or UsageInfo(),
            latency_ms=latency_ms,
            raw_response=raw_response or {},
        )

    def _handle_error(self, error: Exception, operation: str) -> None:
        """Handle and log errors, converting to LLMError types."""
        if isinstance(error, LLMError):
            raise

        self._logger.error(
            "llm_operation_failed",
            operation=operation,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise LLMResponseError(
            f"{operation} failed: {error}",
            provider=self.provider,
        ) from error

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, provider={self.provider})"
