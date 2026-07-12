"""LLM provider implementations."""



import asyncio
import time
from typing import Optional, Union,  Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from knowprobe.core.models import ModelProvider
from knowprobe.utils.logging import get_logger

from .base import BaseLLMClient
from .exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from .types import GenerationRequest, GenerationResponse, LLMMetadata, Message, Role, UsageInfo

logger = get_logger(__name__)


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM server."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, provider=ModelProvider.OLLAMA.value, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    async def agenerate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        model = request.model or self.model
        messages = self._build_messages(request)

        payload = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "options": self._map_params(request.params),
            "stream": False,
        }

        try:
            client = self._get_async_client()
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data.get("message", {}).get("content", "")
            latency = (time.perf_counter() - start) * 1000

            return self._create_response(
                text=text,
                finish_reason="stop" if not data.get("done") else None,
                latency_ms=latency,
                raw_response=data,
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.timeout}s",
                provider=self.provider,
            ) from e
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}",
                provider=self.provider,
            ) from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            self._handle_error(e, "ollama_generate")

        # unreachable, but satisfies type checker
        raise LLMError("Unexpected error in ollama generate")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        model = request.model or self.model
        messages = self._build_messages(request)

        payload = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "options": self._map_params(request.params),
            "stream": False,
        }

        try:
            client = self._get_sync_client()
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            text = data.get("message", {}).get("content", "")
            latency = (time.perf_counter() - start) * 1000

            return self._create_response(
                text=text,
                finish_reason="stop" if not data.get("done") else None,
                latency_ms=latency,
                raw_response=data,
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self.timeout}s",
                provider=self.provider,
            ) from e
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}",
                provider=self.provider,
            ) from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            self._handle_error(e, "ollama_generate")

        raise LLMError("Unexpected error in ollama generate")

    async def ahealth_check(self) -> bool:
        try:
            client = self._get_async_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def health_check(self) -> bool:
        try:
            client = self._get_sync_client()
            response = client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def _map_params(self, params: Any) -> dict[str, Any]:
        """Map GenerationParams to Ollama options format."""
        return {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "num_predict": params.max_tokens,
            "repeat_penalty": params.repetition_penalty,
            "seed": params.seed,
            "stop": params.stop_sequences if params.stop_sequences else None,
        }

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> None:
        status = error.response.status_code
        if status == 401:
            raise LLMAuthenticationError("Ollama authentication failed", provider=self.provider) from error
        if status == 404:
            raise LLMModelNotFoundError(f"Model not found: {self.model}", provider=self.provider) from error
        if status == 429:
            raise LLMRateLimitError("Ollama rate limit exceeded", provider=self.provider) from error
        raise LLMResponseError(
            f"Ollama HTTP error: {status}",
            provider=self.provider,
            details={"status_code": status, "body": error.response.text},
        ) from error

    async def aclose(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def close(self) -> None:
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (OpenAI, DeepSeek, etc.)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = ModelProvider.OPENAI.value,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, provider=provider_name, **kwargs)
        if not api_key:
            raise LLMAuthenticationError(f"API key required for {provider_name}")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
        return self._async_client

    async def agenerate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        model = request.model or self.model
        messages = self._build_messages(request)

        payload = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
        }
        if request.params.stop_sequences:
            payload["stop"] = request.params.stop_sequences
        if request.params.seed is not None:
            payload["seed"] = request.params.seed

        try:
            client = self._get_async_client()
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            usage_data = data.get("usage", {})
            usage = UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            latency = (time.perf_counter() - start) * 1000

            return self._create_response(
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                latency_ms=latency,
                raw_response=data,
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"{self.provider} request timed out after {self.timeout}s",
                provider=self.provider,
            ) from e
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to {self.provider} API",
                provider=self.provider,
            ) from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            self._handle_error(e, f"{self.provider}_generate")

        raise LLMError(f"Unexpected error in {self.provider} generate")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        model = request.model or self.model
        messages = self._build_messages(request)

        payload = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
        }
        if request.params.stop_sequences:
            payload["stop"] = request.params.stop_sequences
        if request.params.seed is not None:
            payload["seed"] = request.params.seed

        try:
            client = self._get_sync_client()
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")
            usage_data = data.get("usage", {})
            usage = UsageInfo(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            latency = (time.perf_counter() - start) * 1000

            return self._create_response(
                text=text,
                finish_reason=finish_reason,
                usage=usage,
                latency_ms=latency,
                raw_response=data,
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"{self.provider} request timed out after {self.timeout}s",
                provider=self.provider,
            ) from e
        except httpx.ConnectError as e:
            raise LLMConnectionError(
                f"Cannot connect to {self.provider} API",
                provider=self.provider,
            ) from e
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
        except Exception as e:
            self._handle_error(e, f"{self.provider}_generate")

        raise LLMError(f"Unexpected error in {self.provider} generate")

    async def ahealth_check(self) -> bool:
        try:
            client = self._get_async_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def health_check(self) -> bool:
        try:
            client = self._get_sync_client()
            response = client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> None:
        status = error.response.status_code
        body = error.response.text
        if status == 401:
            raise LLMAuthenticationError(f"{self.provider} authentication failed", provider=self.provider) from error
        if status == 404:
            raise LLMModelNotFoundError(f"Model not found: {self.model}", provider=self.provider) from error
        if status == 429:
            retry_after = None
            try:
                retry_after = float(error.response.headers.get("retry-after", 0))
            except (ValueError, TypeError):
                pass
            raise LLMRateLimitError(
                f"{self.provider} rate limit exceeded",
                provider=self.provider,
                retry_after=retry_after,
            ) from error
        raise LLMResponseError(
            f"{self.provider} HTTP error: {status}",
            provider=self.provider,
            details={"status_code": status, "body": body},
        ) from error

    async def aclose(self) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def close(self) -> None:
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()


class DeepSeekClient(OpenAICompatibleClient):
    """Client for DeepSeek API."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider_name=ModelProvider.DEEPSEEK.value,
            **kwargs,
        )


class ClaudeClient(OpenAICompatibleClient):
    """Client for Anthropic Claude API (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider_name=ModelProvider.CLAUDE.value,
            **kwargs,
        )
