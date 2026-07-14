"""Custom exceptions for knowledge input parsing and validation."""

from typing import Any


class KnowledgeParserError(Exception):
    """Base exception for all knowledge parser errors."""

    def __init__(
        self, message: str, *, source_id: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        """Initialize with message and optional context."""
        super().__init__(message)
        self.source_id = source_id
        self.details = details or {}


class UnsupportedFormatError(KnowledgeParserError):
    """Raised when an unsupported input format is encountered."""

    def __init__(
        self, input_type: str, *, supported: list[str] | None = None, **kwargs: Any
    ) -> None:
        """Initialize with input type and supported types."""
        msg = f"Unsupported input type: '{input_type}'"
        if supported:
            msg += f". Supported: {', '.join(supported)}"
        super().__init__(msg, **kwargs)
        self.input_type = input_type
        self.supported = supported or []


class ParseError(KnowledgeParserError):
    """Raised when parsing fails for a specific input."""

    def __init__(
        self,
        message: str,
        *,
        raw_content: str | None = None,
        parser_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with parsing context."""
        super().__init__(message, **kwargs)
        self.raw_content = raw_content
        self.parser_name = parser_name


class ValidationError(KnowledgeParserError):
    """Raised when input validation fails."""

    def __init__(
        self, message: str, *, field: str | None = None, value: Any = None, **kwargs: Any
    ) -> None:
        """Initialize with validation context."""
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class BatchProcessingError(KnowledgeParserError):
    """Raised when batch processing encounters unrecoverable errors."""

    def __init__(
        self, message: str, *, errors: list[KnowledgeParserError] | None = None, **kwargs: Any
    ) -> None:
        """Initialize with error list."""
        super().__init__(message, **kwargs)
        self.errors = errors or []
