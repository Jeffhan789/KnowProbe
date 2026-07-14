"""LLM client type definitions."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    """Message roles for chat completions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A chat message."""

    role: Role = Field(default=Role.USER)
    content: str = Field(description="Message content")
    name: str | None = Field(default=None, description="Optional name identifier")


class GenerationParams(BaseModel):
    """Parameters for text generation."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=256, ge=1)
    num_beams: int = Field(default=1, ge=1)
    do_sample: bool = Field(default=True)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    stop_sequences: list[str] = Field(default_factory=list)
    seed: int | None = Field(default=None, description="Random seed for reproducibility")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class GenerationRequest(BaseModel):
    """Request for text generation."""

    prompt: str = Field(
        default="", description="The input prompt; optional when messages are provided"
    )
    messages: list[Message] = Field(
        default_factory=list, description="Chat messages (alternative to prompt)"
    )
    system_prompt: str | None = Field(default=None, description="System prompt for chat models")
    params: GenerationParams = Field(default_factory=GenerationParams)
    model: str | None = Field(default=None, description="Override default model")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode="after")
    def populate_total_tokens(self) -> "UsageInfo":
        """Derive total tokens when a provider omits that aggregate."""
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class GenerationResponse(BaseModel):
    """Response from text generation."""

    text: str = Field(description="Generated text")
    model: str = Field(description="Model used for generation")
    provider: str = Field(description="Provider name")
    finish_reason: str | None = Field(default=None)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency_ms: float = Field(default=0.0, description="Generation latency in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    raw_response: dict[str, Any] = Field(default_factory=dict, description="Raw provider response")


class LLMMetadata(BaseModel):
    """Metadata about an LLM model."""

    id: str = Field(description="Model identifier")
    provider: str = Field(description="Provider name")
    context_length: int | None = Field(default=None)
    supports_chat: bool = Field(default=True)
    supports_functions: bool = Field(default=False)
    supports_vision: bool = Field(default=False)


class BatchGenerationRequest(BaseModel):
    """Batch generation request."""

    requests: list[GenerationRequest]
    common_params: GenerationParams | None = Field(default=None)


class BatchGenerationResponse(BaseModel):
    """Batch generation response."""

    responses: list[GenerationResponse]
    total_latency_ms: float = Field(default=0.0)
