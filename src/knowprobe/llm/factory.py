"""LLM client factory."""



from typing import Any, Optional, Union

from knowprobe.core.config import Settings, get_settings
from knowprobe.core.models import ModelProvider
from knowprobe.utils.logging import get_logger

from .base import BaseLLMClient
from .exceptions import LLMConfigError, LLMError
from .providers import ClaudeClient, DeepSeekClient, OllamaClient, OpenAICompatibleClient

logger = get_logger(__name__)

_PROVIDER_MAP: dict[str, type[BaseLLMClient]] = {
    ModelProvider.OLLAMA.value: OllamaClient,
    ModelProvider.OPENAI.value: OpenAICompatibleClient,
    ModelProvider.DEEPSEEK.value: DeepSeekClient,
    ModelProvider.CLAUDE.value: ClaudeClient,
    # vLLM and transformers use Ollama-compatible or direct implementations
    ModelProvider.VLLM.value: OllamaClient,
}


def create_client(
    provider: Union[str, ModelProvider],
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.

    Args:
        provider: The LLM provider to use.
        model: Optional model override. If not provided, uses provider's default.
        settings: Application settings. Uses global settings if not provided.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured BaseLLMClient instance.

    Raises:
        LLMConfigError: If provider configuration is invalid.
        LLMError: If client creation fails.

    Examples:
        >>> client = create_client("ollama", model="llama3.1:8b")
        >>> client = create_client("openai", model="gpt-4o-mini")
    """
    if isinstance(provider, ModelProvider):
        provider = provider.value

    provider = provider.lower().strip()
    settings = settings or get_settings()

    logger.info(
        "creating_llm_client",
        provider=provider,
        model=model or "default",
    )

    client_cls = _PROVIDER_MAP.get(provider)
    if client_cls is None:
        raise LLMConfigError(
            f"Unknown provider: {provider}. Supported: {list(_PROVIDER_MAP.keys())}",
        )

    try:
        if provider == ModelProvider.OLLAMA.value or provider == ModelProvider.VLLM.value:
            local_config = settings.models.local
            resolved_model = model or local_config.default_model
            return client_cls(
                model=resolved_model,
                base_url=local_config.base_url,
                timeout=local_config.timeout,
                **kwargs,
            )

        elif provider == ModelProvider.OPENAI.value:
            api_config = settings.models.api.get("openai", {})
            resolved_model = model or api_config.get("default_model", "gpt-4o-mini")
            api_key = api_config.get("api_key", "")
            base_url = api_config.get("base_url", "https://api.openai.com/v1")
            if not api_key:
                raise LLMConfigError("OpenAI API key not configured")
            return OpenAICompatibleClient(
                model=resolved_model,
                api_key=api_key,
                base_url=base_url,
                provider_name=ModelProvider.OPENAI.value,
                **kwargs,
            )

        elif provider == ModelProvider.DEEPSEEK.value:
            api_config = settings.models.api.get("deepseek", {})
            resolved_model = model or api_config.get("default_model", "deepseek-chat")
            api_key = api_config.get("api_key", "")
            base_url = api_config.get("base_url", "https://api.deepseek.com/v1")
            if not api_key:
                raise LLMConfigError("DeepSeek API key not configured")
            return DeepSeekClient(
                model=resolved_model,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        elif provider == ModelProvider.CLAUDE.value:
            api_config = settings.models.api.get("claude", {})
            resolved_model = model or api_config.get("default_model", "claude-3-haiku-20240307")
            api_key = api_config.get("api_key", "")
            base_url = api_config.get("base_url", "https://api.anthropic.com")
            if not api_key:
                raise LLMConfigError("Claude API key not configured")
            return ClaudeClient(
                model=resolved_model,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        else:
            # Fallback for any registered provider
            return client_cls(model=model or "default", **kwargs)

    except LLMConfigError:
        raise
    except Exception as e:
        logger.error(
            "failed_to_create_client",
            provider=provider,
            error=str(e),
        )
        raise LLMError(f"Failed to create {provider} client: {e}", provider=provider) from e


def register_provider(name: str, client_class: type[BaseLLMClient]) -> None:
    """Register a custom provider client class.

    Args:
        name: Provider identifier.
        client_class: Client class implementing BaseLLMClient.

    Raises:
        LLMConfigError: If client_class is not a valid BaseLLMClient subclass.
    """
    if not issubclass(client_class, BaseLLMClient):
        raise LLMConfigError(f"Client class must inherit from BaseLLMClient: {client_class}")
    _PROVIDER_MAP[name.lower()] = client_class
    logger.info("registered_custom_provider", name=name, client_class=client_class.__name__)


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(_PROVIDER_MAP.keys())
