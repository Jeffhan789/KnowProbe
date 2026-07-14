"""LLM client exceptions."""

from typing import Any


class LLMError(Exception):
    """Base exception for LLM client errors."""

    def __init__(
        self, message: str, *, provider: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.details = details or {}

    def __str__(self) -> str:
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.details:
            parts.append(f"details={self.details}")
        return " | ".join(parts)


class LLMConfigError(LLMError):
    """Raised when LLM client configuration is invalid."""


class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails."""


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or malformed."""


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, details=details)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """Raised when LLM authentication fails."""


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""


class LLMModelNotFoundError(LLMError):
    """Raised when requested model is not found."""
