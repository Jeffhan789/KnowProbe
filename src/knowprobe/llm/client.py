"""Unified LLM client with retry, caching, and metrics support."""



import time
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from knowprobe.utils.logging import get_logger

from .base import BaseLLMClient
from .exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from .factory import create_client
from .types import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationParams,
    GenerationRequest,
    GenerationResponse,
)

logger = get_logger(__name__)


class UnifiedLLMClient:
    """Unified LLM client with automatic retry, metrics, and provider abstraction.

    This is the main interface for LLM generation in KnowProbe. It wraps
    provider-specific clients and adds cross-cutting concerns like retry logic,
    latency tracking, and unified error handling.

    Usage:
        >>> client = UnifiedLLMClient("ollama", model="llama3.1:8b")
        >>> response = await client.agenerate("Generate a question about...")
        >>> print(response.text)
    """

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize unified client.

        Args:
            provider: LLM provider name (e.g., 'ollama', 'openai', 'deepseek').
            model: Model name. Uses provider default if not specified.
            max_retries: Maximum retry attempts for failed requests.
            retry_min_wait: Minimum wait seconds between retries.
            retry_max_wait: Maximum wait seconds between retries.
            **kwargs: Additional arguments passed to provider client.
        """
        self._client = create_client(provider, model=model, **kwargs)
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self._logger = get_logger(f"{__name__}.unified")

    @property
    def provider(self) -> str:
        return self._client.provider

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def base_client(self) -> BaseLLMClient:
        """Access the underlying provider client."""
        return self._client

    def _get_retry_decorator(self) -> Any:
        """Build tenacity retry decorator."""
        return retry(
            retry=retry_if_exception_type((LLMError, LLMTimeoutError, LLMRateLimitError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(min=self.retry_min_wait, max=self.retry_max_wait),
            reraise=True,
        )

    async def agenerate(
        self,
        prompt: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate text asynchronously with automatic retry.

        Args:
            prompt: The input prompt text.
            system_prompt: Optional system prompt.
            params: Generation parameters. Uses defaults if not provided.
            **kwargs: Additional request parameters.

        Returns:
            GenerationResponse with generated text.

        Raises:
            LLMError: After all retries are exhausted.
        """
        request = GenerationRequest(
            prompt=prompt or "",
            system_prompt=system_prompt,
            params=params or GenerationParams(),
            **kwargs,
        )

        @self._get_retry_decorator()
        async def _generate() -> GenerationResponse:
            start = time.perf_counter()
            try:
                response = await self._client.agenerate(request)
                latency = (time.perf_counter() - start) * 1000
                self._logger.info(
                    "async_generation_complete",
                    provider=self.provider,
                    model=self.model,
                    latency_ms=round(latency, 2),
                    tokens=response.usage.total_tokens,
                )
                return response
            except LLMError as e:
                self._logger.warning(
                    "async_generation_retry",
                    provider=self.provider,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return await _generate()

    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> GenerationResponse:
        """Generate text synchronously with automatic retry.

        Args:
            prompt: The input prompt text.
            system_prompt: Optional system prompt.
            params: Generation parameters. Uses defaults if not provided.
            **kwargs: Additional request parameters.

        Returns:
            GenerationResponse with generated text.

        Raises:
            LLMError: After all retries are exhausted.
        """
        request = GenerationRequest(
            prompt=prompt or "",
            system_prompt=system_prompt,
            params=params or GenerationParams(),
            **kwargs,
        )

        @self._get_retry_decorator()
        def _generate() -> GenerationResponse:
            start = time.perf_counter()
            try:
                response = self._client.generate(request)
                latency = (time.perf_counter() - start) * 1000
                self._logger.info(
                    "sync_generation_complete",
                    provider=self.provider,
                    model=self.model,
                    latency_ms=round(latency, 2),
                    tokens=response.usage.total_tokens,
                )
                return response
            except LLMError as e:
                self._logger.warning(
                    "sync_generation_retry",
                    provider=self.provider,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return _generate()

    async def abatch_generate(
        self,
        prompts: list[str],
        *,
        system_prompt: Optional[str] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> BatchGenerationResponse:
        """Generate multiple texts asynchronously.

        Args:
            prompts: List of prompts to generate from.
            system_prompt: Optional shared system prompt.
            params: Generation parameters for all requests.
            **kwargs: Additional request parameters.

        Returns:
            BatchGenerationResponse with all responses.
        """
        requests = [
            GenerationRequest(
                prompt=p,
                system_prompt=system_prompt,
                params=params or GenerationParams(),
                **kwargs,
            )
            for p in prompts
        ]
        batch = BatchGenerationRequest(requests=requests)
        return await self._client.abatch_generate(batch)

    def batch_generate(
        self,
        prompts: list[str],
        *,
        system_prompt: Optional[str] = None,
        params: Optional[GenerationParams] = None,
        **kwargs: Any,
    ) -> BatchGenerationResponse:
        """Generate multiple texts synchronously.

        Args:
            prompts: List of prompts to generate from.
            system_prompt: Optional shared system prompt.
            params: Generation parameters for all requests.
            **kwargs: Additional request parameters.

        Returns:
            BatchGenerationResponse with all responses.
        """
        requests = [
            GenerationRequest(
                prompt=p,
                system_prompt=system_prompt,
                params=params or GenerationParams(),
                **kwargs,
            )
            for p in prompts
        ]
        batch = BatchGenerationRequest(requests=requests)
        return self._client.batch_generate(batch)

    async def ahealth_check(self) -> bool:
        """Check if the LLM service is healthy asynchronously."""
        return await self._client.ahealth_check()

    def health_check(self) -> bool:
        """Check if the LLM service is healthy synchronously."""
        return self._client.health_check()

    async def aclose(self) -> None:
        """Close async resources."""
        if hasattr(self._client, "aclose"):
            await self._client.aclose()

    def close(self) -> None:
        """Close sync resources."""
        if hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "UnifiedLLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> "UnifiedLLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"UnifiedLLMClient(provider={self.provider}, model={self.model})"
