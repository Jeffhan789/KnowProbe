"""Tests for LLM unified client."""

import pytest

from knowprobe.llm import (
    GenerationParams,
    GenerationRequest,
    LLMConfigError,
    LLMError,
    UnifiedLLMClient,
    create_client,
    list_providers,
    register_provider,
)
from knowprobe.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from knowprobe.llm.types import Message, Role, UsageInfo


class TestTypes:
    """Test data types."""

    def test_generation_params_defaults(self):
        params = GenerationParams()
        assert params.temperature == 0.7
        assert params.max_tokens == 256
        assert params.do_sample is True

    def test_generation_params_to_dict(self):
        params = GenerationParams(temperature=0.5, seed=42)
        d = params.to_dict()
        assert d["temperature"] == 0.5
        assert d["seed"] == 42

    def test_generation_request(self):
        req = GenerationRequest(prompt="test prompt")
        assert req.prompt == "test prompt"
        assert req.params.temperature == 0.7

    def test_message_creation(self):
        msg = Message(role=Role.SYSTEM, content="system prompt")
        assert msg.role == Role.SYSTEM
        assert msg.content == "system prompt"

    def test_usage_info(self):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30


class TestExceptions:
    """Test exception hierarchy."""

    def test_llm_error_base(self):
        err = LLMError("test error", provider="ollama")
        assert err.message == "test error"
        assert err.provider == "ollama"
        assert "ollama" in str(err)

    def test_llm_error_with_details(self):
        err = LLMError("test", details={"code": 500})
        assert err.details["code"] == 500

    def test_rate_limit_error(self):
        err = LLMRateLimitError("rate limited", retry_after=30.0)
        assert err.retry_after == 30.0

    def test_exception_hierarchy(self):
        assert issubclass(LLMConfigError, LLMError)
        assert issubclass(LLMConnectionError, LLMError)
        assert issubclass(LLMResponseError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMAuthenticationError, LLMError)
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMModelNotFoundError, LLMError)


class TestFactory:
    """Test client factory."""

    def test_list_providers(self):
        providers = list_providers()
        assert "ollama" in providers
        assert "openai" in providers
        assert "deepseek" in providers
        assert "claude" in providers

    def test_create_client_unknown_provider(self):
        with pytest.raises(LLMConfigError):
            create_client("unknown_provider")

    def test_create_client_no_api_key(self):
        with pytest.raises(LLMConfigError):
            create_client("openai", api_key="")

    def test_register_provider_invalid(self):
        with pytest.raises(LLMConfigError):
            register_provider("test", str)  # type: ignore[arg-type]


class TestUnifiedClient:
    """Test unified client interface."""

    def test_client_repr(self):
        client = UnifiedLLMClient("ollama", model="llama3.1:8b")
        assert "UnifiedLLMClient" in repr(client)
        assert "ollama" in repr(client)

    def test_client_properties(self):
        client = UnifiedLLMClient("ollama", model="llama3.1:8b")
        assert client.provider == "ollama"
        assert client.model == "llama3.1:8b"

    def test_client_context_manager(self):
        with UnifiedLLMClient("ollama") as client:
            assert client is not None


class TestBaseClientMethods:
    """Test base client helper methods."""

    def test_build_messages_from_prompt(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        req = GenerationRequest(prompt="hello", system_prompt="be helpful")
        messages = client._build_messages(req)
        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER

    def test_build_messages_from_messages(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        req = GenerationRequest(
            messages=[
                Message(role=Role.USER, content="hello"),
                Message(role=Role.ASSISTANT, content="hi"),
            ]
        )
        messages = client._build_messages(req)
        assert len(messages) == 2
        assert messages[0].role == Role.USER

    def test_build_prompt(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        req = GenerationRequest(
            system_prompt="sys",
            messages=[
                Message(role=Role.USER, content="hello"),
            ],
        )
        prompt = client._build_prompt(req)
        assert "System: sys" in prompt
        assert "User: hello" in prompt

    def test_create_response(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        resp = client._create_response(
            text="generated",
            finish_reason="stop",
            usage=UsageInfo(prompt_tokens=5, completion_tokens=10),
            latency_ms=100.0,
        )
        assert resp.text == "generated"
        assert resp.finish_reason == "stop"
        assert resp.usage.total_tokens == 15
        assert resp.latency_ms == 100.0


class TestOllamaClient:
    """Test Ollama client specifics."""

    def test_ollama_default_url(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        assert client.base_url == "http://localhost:11434"

    def test_ollama_custom_url(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test", base_url="http://custom:8080/")
        assert client.base_url == "http://custom:8080"

    def test_ollama_map_params(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        params = GenerationParams(temperature=0.5, max_tokens=100, seed=42)
        mapped = client._map_params(params)
        assert mapped["temperature"] == 0.5
        assert mapped["num_predict"] == 100
        assert mapped["seed"] == 42

    def test_ollama_close(self):
        from knowprobe.llm.providers import OllamaClient

        client = OllamaClient(model="test")
        client.close()  # should not raise


class TestOpenAICompatibleClient:
    """Test OpenAI-compatible client specifics."""

    def test_openai_requires_key(self):
        from knowprobe.llm.providers import OpenAICompatibleClient

        with pytest.raises(LLMAuthenticationError):
            OpenAICompatibleClient(model="gpt-4", api_key="")

    def test_deepseek_client(self):
        from knowprobe.llm.providers import DeepSeekClient

        client = DeepSeekClient(model="deepseek-chat", api_key="test-key")
        assert client.provider == "deepseek"
        assert client.model == "deepseek-chat"

    def test_claude_client(self):
        from knowprobe.llm.providers import ClaudeClient

        client = ClaudeClient(model="claude-3", api_key="test-key")
        assert client.provider == "claude"
        assert client.model == "claude-3"
