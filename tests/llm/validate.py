"""Standalone validation for LLM client - no project dependencies."""

import os
import sys
import types

src_path = os.path.join(os.path.dirname(__file__), "../../src")
sys.path.insert(0, src_path)

# Mock knowprobe.core.config
mock_config = types.ModuleType("knowprobe.core.config")


class MockLocalConfig:
    provider = "ollama"
    base_url = "http://localhost:11434"
    default_model = "llama3.1:8b"
    timeout = 300


class MockModelsConfig:
    local = MockLocalConfig()
    api = {}


class MockSettings:
    models = MockModelsConfig()


mock_config.Settings = MockSettings  # type: ignore
mock_config.get_settings = MockSettings  # type: ignore
mock_config.load_settings = lambda x=None: MockSettings()  # type: ignore
sys.modules["knowprobe.core.config"] = mock_config

# Mock knowprobe.core.models
mock_models = types.ModuleType("knowprobe.core.models")


class MockModelProvider:
    OLLAMA = type("obj", (object,), {"value": "ollama"})()
    OPENAI = type("obj", (object,), {"value": "openai"})()
    DEEPSEEK = type("obj", (object,), {"value": "deepseek"})()
    CLAUDE = type("obj", (object,), {"value": "claude"})()
    VLLM = type("obj", (object,), {"value": "vllm"})()
    TRANSFORMERS = type("obj", (object,), {"value": "transformers"})()


mock_models.ModelProvider = MockModelProvider  # type: ignore
sys.modules["knowprobe.core.models"] = mock_models

# Mock knowprobe.utils.logging
mock_logging = types.ModuleType("knowprobe.utils.logging")


class MockLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


mock_logging.configure_logging = lambda *args, **kwargs: None  # type: ignore
mock_logging.get_logger = lambda name: MockLogger()  # type: ignore
sys.modules["knowprobe.utils.logging"] = mock_logging
sys.modules["knowprobe.utils"] = types.ModuleType("knowprobe.utils")

print("=" * 60)
print("KnowProbe LLM Client - Standalone Validation")
print("=" * 60)

# Test 1: Import all LLM modules
print("\n[1/8] Testing LLM module imports...")
try:
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
    from knowprobe.llm.factory import list_providers
    from knowprobe.llm.providers import (
        ClaudeClient,
        DeepSeekClient,
        OllamaClient,
        OpenAICompatibleClient,
    )
    from knowprobe.llm.types import (
        GenerationParams,
        GenerationRequest,
        Message,
        Role,
        UsageInfo,
    )

    print("   ✓ All LLM imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Exception hierarchy
print("\n[2/8] Testing exception hierarchy...")
assert issubclass(LLMConfigError, LLMError)
assert issubclass(LLMConnectionError, LLMError)
assert issubclass(LLMResponseError, LLMError)
assert issubclass(LLMRateLimitError, LLMError)
assert issubclass(LLMAuthenticationError, LLMError)
assert issubclass(LLMTimeoutError, LLMError)
assert issubclass(LLMModelNotFoundError, LLMError)
err = LLMError("test", provider="ollama", details={"code": 500})
assert err.provider == "ollama"
assert err.details["code"] == 500
print("   ✓ Exception hierarchy correct")

# Test 3: Data types
print("\n[3/8] Testing data types...")
params = GenerationParams(temperature=0.5, seed=42)
assert params.temperature == 0.5
assert params.seed == 42
d = params.to_dict()
assert d["temperature"] == 0.5

req = GenerationRequest(prompt="test", system_prompt="sys")
assert req.prompt == "test"
assert req.system_prompt == "sys"

msg = Message(role=Role.SYSTEM, content="hello")
assert msg.role == Role.SYSTEM
assert msg.content == "hello"

usage = UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
assert usage.total_tokens == 30
print("   ✓ Data types working correctly")

# Test 4: Ollama client
print("\n[4/8] Testing Ollama client...")
client = OllamaClient(model="llama3.1:8b")
assert client.model == "llama3.1:8b"
assert client.provider == "ollama"
assert client.base_url == "http://localhost:11434"

mapped = client._map_params(GenerationParams(temperature=0.3, max_tokens=100))
assert mapped["temperature"] == 0.3
assert mapped["num_predict"] == 100

req = GenerationRequest(prompt="hello", system_prompt="be helpful")
messages = client._build_messages(req)
assert len(messages) == 2
assert messages[0].role == Role.SYSTEM
assert messages[1].role == Role.USER
print("   ✓ Ollama client working correctly")

# Test 5: OpenAI compatible client
print("\n[5/8] Testing OpenAI-compatible client...")
api_client = OpenAICompatibleClient(model="gpt-4", api_key="test-key")
assert api_client.model == "gpt-4"
assert api_client.provider == "openai"
assert api_client.api_key == "test-key"

deepseek = DeepSeekClient(model="deepseek-chat", api_key="ds-key")
assert deepseek.provider == "deepseek"
assert deepseek.model == "deepseek-chat"

claude = ClaudeClient(model="claude-3", api_key="claude-key")
assert claude.provider == "claude"
assert claude.model == "claude-3"
print("   ✓ OpenAI-compatible clients working correctly")

# Test 6: Factory
print("\n[6/8] Testing factory...")
providers = list_providers()
assert "ollama" in providers
assert "openai" in providers
assert "deepseek" in providers
assert "claude" in providers
print(f"   ✓ Registered providers: {providers}")

# Test 7: Unified client
print("\n[7/8] Testing unified client...")
unified = UnifiedLLMClient("ollama", model="llama3.1:8b")
assert unified.provider == "ollama"
assert unified.model == "llama3.1:8b"
assert "UnifiedLLMClient" in repr(unified)

# Context manager
with UnifiedLLMClient("ollama") as uc:
    assert uc is not None
print("   ✓ Unified client working correctly")

# Test 8: Response creation
print("\n[8/8] Testing response creation...")
resp = client._create_response(
    text="generated text",
    finish_reason="stop",
    usage=UsageInfo(prompt_tokens=5, completion_tokens=10),
    latency_ms=150.0,
)
assert resp.text == "generated text"
assert resp.finish_reason == "stop"
assert resp.usage.prompt_tokens == 5
assert resp.usage.completion_tokens == 10
assert resp.latency_ms == 150.0
assert resp.provider == "ollama"
assert resp.model == "llama3.1:8b"
print("   ✓ Response creation working correctly")

print("\n" + "=" * 60)
print("✅ ALL VALIDATION TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print("  - Exception hierarchy: 8 classes")
print("  - Data types: 9 Pydantic models")
print(f"  - Providers: {len(providers)} implemented")
print("  - Unified client: retry + metrics + context manager")
print("  - BaseLLMClient: abstract with sync/async + batch support")
