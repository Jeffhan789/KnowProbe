"""Unified model client for multiple backends.

Provides a single ``BaseModelClient`` abstraction over:

- **Ollama** — local HTTP API (llama3.1, qwen2.5, etc.)
- **Transformers** — HuggingFace ``transformers`` direct inference
- **OpenAI-compatible** — OpenAI, DeepSeek, Claude, or any OpenAI-like API

All clients expose ``generate()`` and ``generate_batch()`` with standardised
``ModelResponse`` and automatic retry via ``tenacity``.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

from knowprobe.core.config import GenerationConfig, get_settings
from knowprobe.core.models import ModelProvider
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelResponse:
    """Standardised model response, independent of backend.

    Attributes:
        text: Generated text (already decoded / stripped).
        usage: Token usage dict with keys ``prompt_tokens``,
            ``completion_tokens``, ``total_tokens``. May be ``None``.
        latency_ms: Wall-clock latency in milliseconds.
        model: Model identifier that produced the response.
    """

    text: str
    usage: dict[str, int] | None = field(default=None)
    latency_ms: float = 0.0
    model: str = ""


# --------------------------------------------------------------------------- #
# Base client
# --------------------------------------------------------------------------- #


class BaseModelClient(ABC):
    """Abstract base for all model clients."""

    @abstractmethod
    async def generate(self, prompt: str, **params: Any) -> ModelResponse:
        """Generate text for a single prompt.

        Args:
            prompt: The full prompt string.
            **params: Generation overrides (temperature, max_tokens, etc.).

        Returns:
            Standardised ``ModelResponse``.
        """
        ...

    @abstractmethod
    async def generate_batch(self, prompts: list[str], **params: Any) -> list[ModelResponse]:
        """Generate text for multiple prompts.

        Default implementation falls back to sequential ``generate()`` calls;
        subclasses may override with true batching if the backend supports it.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Return health status dict with ``status`` key."""
        ...

    async def __aenter__(self) -> BaseModelClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    @abstractmethod
    async def close(self) -> None:
        """Release resources (HTTP clients, GPU memory, etc.)."""
        ...


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #


class OllamaClient(BaseModelClient):
    """Client for Ollama local HTTP API.

    Expects Ollama running on ``base_url`` (default http://localhost:11434).
    Supports both ``/api/generate`` (legacy) and ``/api/chat`` endpoints.
    """

    _DEFAULT_TIMEOUT = 300

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._logger = get_logger(f"OllamaClient.{model}")

    async def __aenter__(self) -> OllamaClient:
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException, httpx.ConnectError)
        ),
        reraise=True,
    )
    async def generate(self, prompt: str, **params: Any) -> ModelResponse:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        payload = self._build_payload(prompt, params)
        start = time.monotonic()

        try:
            resp = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._logger.error(
                "Ollama HTTP error",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            raise

        latency = (time.monotonic() - start) * 1000
        data = resp.json()

        return ModelResponse(
            text=data.get("response", "").strip(),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            },
            latency_ms=latency,
            model=self.model,
        )

    async def generate_batch(self, prompts: list[str], **params: Any) -> list[ModelResponse]:
        """Ollama does not natively support batching; run sequentially."""
        # Run with limited concurrency to avoid overwhelming the local GPU
        semaphore = asyncio.Semaphore(4)

        async def _gen(p: str) -> ModelResponse:
            async with semaphore:
                return await self.generate(p, **params)

        return await asyncio.gather(*[_gen(p) for p in prompts])

    async def health_check(self) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10)
        try:
            resp = await self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            available = any(m.get("name") == self.model for m in models)
            return {
                "status": "ok" if available else "degraded",
                "model_available": available,
                "ollama_models": [m.get("name") for m in models],
            }
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            return {"status": "unavailable", "error": str(exc)}
        except httpx.HTTPError as exc:
            return {"status": "degraded", "error": str(exc)}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_payload(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Build Ollama /api/generate payload."""
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 0.9),
                "top_k": params.get("top_k", 50),
                "num_predict": params.get("max_tokens", params.get("max_length", 256)),
            },
        }


# --------------------------------------------------------------------------- #
# Transformers client (local HuggingFace)
# --------------------------------------------------------------------------- #


class TransformersClient(BaseModelClient):
    """Client for direct HuggingFace ``transformers`` inference.

    Loads the model and tokenizer on initialisation. Supports GPU offloading
    via ``accelerate`` and ``device_map``. The client is **not** thread-safe;
    use a single instance per process.
    """

    def __init__(
        self,
        model_name: str,
        *,
        generation_config: GenerationConfig | None = None,
        device_map: str | None = "auto",
        torch_dtype: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.gen_config = generation_config or get_settings().generation
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._model: Any = None
        self._tokenizer: Any = None
        self._logger = get_logger(f"TransformersClient.{model_name}")
        self._initialized = False

    async def initialize(self) -> None:
        """Lazy-load model and tokenizer (may take several seconds)."""
        if self._initialized:
            return

        # Heavy sync I/O — run in thread pool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)
        self._initialized = True
        self._logger.info("Transformers model loaded", model=self.model_name)

    async def __aenter__(self) -> TransformersClient:
        await self.initialize()
        return self

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in executor)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {}
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype, torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **kwargs,
        )

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def generate(self, prompt: str, **params: Any) -> ModelResponse:
        if not self._initialized:
            raise RuntimeError("TransformersClient not initialized. Call initialize() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, prompt, params
        )

    async def generate_batch(self, prompts: list[str], **params: Any) -> list[ModelResponse]:
        """Process prompts sequentially; true batching would need padding logic."""
        return await asyncio.gather(*[self.generate(p, **params) for p in prompts])

    def _generate_sync(self, prompt: str, params: dict[str, Any]) -> ModelResponse:
        """Synchronous generation (runs in executor)."""
        import torch

        start = time.monotonic()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        if self.device_map is None:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": params.get(
                "max_tokens", params.get("max_length", self.gen_config.max_length)
            ),
            "temperature": params.get("temperature", self.gen_config.temperature),
            "top_p": params.get("top_p", self.gen_config.top_p),
            "top_k": params.get("top_k", self.gen_config.top_k),
            "num_beams": params.get("num_beams", self.gen_config.num_beams),
            "do_sample": params.get("do_sample", self.gen_config.do_sample),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_kwargs)

        latency = (time.monotonic() - start) * 1000

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return ModelResponse(
            text=text,
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(new_tokens),
                "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens),
            },
            latency_ms=latency,
            model=self.model_name,
        )

    async def health_check(self) -> dict[str, Any]:
        if not self._initialized:
            return {"status": "degraded", "reason": "model_not_loaded"}
        try:
            # Quick smoke test
            _ = await self.generate("Hello", max_tokens=1)
            return {"status": "ok", "device": str(self._model.device)}
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}

    async def close(self) -> None:
        if self._model is not None:
            import torch

            del self._model
            self._model = None
            torch.cuda.empty_cache()
        self._initialized = False


# --------------------------------------------------------------------------- #
# OpenAI-compatible client
# --------------------------------------------------------------------------- #


class OpenAICompatibleClient(BaseModelClient):
    """Client for OpenAI-compatible APIs (OpenAI, DeepSeek, Claude, etc.).

    Uses the ``/chat/completions`` endpoint with a simple system+user message
    structure. The prompt is sent as the ``user`` message content.
    """

    _DEFAULT_TIMEOUT = 120

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ValueError("API key is required for OpenAI-compatible clients")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._logger = get_logger(f"OpenAICompatibleClient.{model}")

    async def __aenter__(self) -> OpenAICompatibleClient:
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(
            (httpx.HTTPError, httpx.TimeoutException)
        ),
        reraise=True,
    )
    async def generate(self, prompt: str, **params: Any) -> ModelResponse:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

        payload = self._build_payload(prompt, params)
        start = time.monotonic()

        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            self._logger.error(
                "API HTTP error",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            raise

        latency = (time.monotonic() - start) * 1000
        data = resp.json()
        choice = data["choices"][0]

        return ModelResponse(
            text=choice["message"]["content"].strip(),
            usage=data.get("usage"),
            latency_ms=latency,
            model=self.model,
        )

    async def generate_batch(self, prompts: list[str], **params: Any) -> list[ModelResponse]:
        """Sequential execution with concurrency limit."""
        semaphore = asyncio.Semaphore(8)

        async def _gen(p: str) -> ModelResponse:
            async with semaphore:
                return await self.generate(p, **params)

        return await asyncio.gather(*[_gen(p) for p in prompts])

    async def health_check(self) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=10,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        try:
            # Try a minimal models list call
            resp = await self._client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            return {"status": "ok", "api_reachable": True}
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            return {"status": "unavailable", "error": str(exc)}
        except httpx.HTTPStatusError as exc:
            # 401/403 still means the API is reachable
            return {
                "status": "degraded",
                "api_reachable": True,
                "status_code": exc.response.status_code,
            }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _build_payload(self, prompt: str, params: dict[str, Any]) -> dict[str, Any]:
        """Build OpenAI chat completions payload."""
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates high-quality "
                        "questions from knowledge content."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9),
            "max_tokens": params.get(
                "max_tokens", params.get("max_length", 256)
            ),
        }


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


class ModelClientFactory:
    """Factory for creating model clients based on provider configuration.

    Usage::

        client = ModelClientFactory.create(
            ModelProvider.OLLAMA,
            "llama3.1:8b",
            base_url="http://localhost:11434",
        )
    """

    @staticmethod
    def create(
        provider: ModelProvider,
        model_name: str,
        **kwargs: Any,
    ) -> BaseModelClient:
        """Create a model client for the given provider.

        Args:
            provider: The model provider enum value.
            model_name: Model identifier.
            **kwargs: Provider-specific overrides (base_url, api_key, etc.).

        Returns:
            Configured ``BaseModelClient`` instance (not yet initialised).
        """
        settings = get_settings()

        if provider == ModelProvider.OLLAMA:
            return OllamaClient(
                base_url=kwargs.get("base_url", settings.models.local.base_url),
                model=model_name,
                timeout=kwargs.get("timeout", settings.models.local.timeout),
            )

        if provider == ModelProvider.TRANSFORMERS:
            return TransformersClient(
                model_name=model_name,
                generation_config=kwargs.get("generation_config"),
                device_map=kwargs.get("device_map", "auto"),
                torch_dtype=kwargs.get("torch_dtype"),
            )

        if provider in (ModelProvider.OPENAI, ModelProvider.DEEPSEEK, ModelProvider.CLAUDE):
            api_config = settings.models.api.get(provider.value, {})
            return OpenAICompatibleClient(
                api_key=kwargs.get("api_key", api_config.get("api_key", "")),
                base_url=kwargs.get(
                    "base_url", api_config.get("base_url", "")
                ),
                model=model_name,
                timeout=kwargs.get("timeout", 120),
            )

        raise ValueError(f"Unsupported model provider: {provider}")
