"""LLM client type definitions."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message roles for chat completions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A chat message."""

    role: Role = Field(default=Role.USER)
    content: str = Field(description="Message content")
    name: Optional[str] = Field(default=None, description="Optional name identifier")


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
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class GenerationRequest(BaseModel):
    """Request for text generation."""

    prompt: str = Field(description="The input prompt")
    messages: list[Message] = Field(default_factory=list, description="Chat messages (alternative to prompt)")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for chat models")
    params: GenerationParams = Field(default_factory=GenerationParams)
    model: Optional[str] = Field(default=None, description="Override default model")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GenerationResponse(BaseModel):
    """Response from text generation."""

    text: str = Field(description="Generated text")
    model: str = Field(description="Model used for generation")
    provider: str = Field(description="Provider name")
    finish_reason: Optional[str] = Field(default=None)
    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency_ms: float = Field(default=0.0, description="Generation latency in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    raw_response: dict[str, Any] = Field(default_factory=dict, description="Raw provider response")


class LLMMetadata(BaseModel):
    """Metadata about an LLM model."""

    id: str = Field(description="Model identifier")
    provider: str = Field(description="Provider name")
    context_length: Optional[int] = Field(default=None)
    supports_chat: bool = Field(default=True)
    supports_functions: bool = Field(default=False)
    supports_vision: bool = Field(default=False)


class BatchGenerationRequest(BaseModel):
    """Batch generation request."""

    requests: list[GenerationRequest]
    common_params: Optional[GenerationParams] = Field(default=None)


class BatchGenerationResponse(BaseModel):
    """Batch generation response."""

    responses: list[GenerationResponse]
    total_latency_ms: float = Field(default=0.0)
