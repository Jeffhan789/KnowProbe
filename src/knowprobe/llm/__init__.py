"""LLM Unified Client for KnowProbe.

This package provides a unified interface for interacting with multiple LLM providers
including Ollama, OpenAI, DeepSeek, and Claude.

Quick Start:
    >>> from knowprobe.llm import UnifiedLLMClient
    >>> client = UnifiedLLMClient("ollama", model="llama3.1:8b")
    >>> response = client.generate("Generate a factual question about Python.")
    >>> print(response.text)

Async Usage:
    >>> client = UnifiedLLMClient("openai", model="gpt-4o-mini")
    >>> response = await client.agenerate("What is RAG?")

Available Providers:
    - ollama: Local models via Ollama (default: llama3.1:8b)
    - openai: OpenAI API (default: gpt-4o-mini)
    - deepseek: DeepSeek API (default: deepseek-chat)
    - claude: Anthropic Claude API (default: claude-3-haiku)

Custom Provider Registration:
    >>> from knowprobe.llm import register_provider, BaseLLMClient
    >>> register_provider("my_provider", MyCustomClient)
"""

from knowprobe.llm.base import BaseLLMClient
from knowprobe.llm.client import UnifiedLLMClient
from knowprobe.llm.exceptions import (
    LLMAuthenticationError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from knowprobe.llm.factory import create_client, list_providers, register_provider
from knowprobe.llm.types import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    GenerationParams,
    GenerationRequest,
    GenerationResponse,
    LLMMetadata,
    Message,
    Role,
    UsageInfo,
)

__all__ = [
    # Core client
    "BaseLLMClient",
    "UnifiedLLMClient",
    # Factory
    "create_client",
    "register_provider",
    "list_providers",
    # Exceptions
    "LLMError",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMResponseError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMTimeoutError",
    "LLMModelNotFoundError",
    # Types
    "GenerationRequest",
    "GenerationResponse",
    "GenerationParams",
    "BatchGenerationRequest",
    "BatchGenerationResponse",
    "Message",
    "Role",
    "UsageInfo",
    "LLMMetadata",
]
