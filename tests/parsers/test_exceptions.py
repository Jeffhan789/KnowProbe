"""Tests for knowledge parser exceptions."""

import pytest

from knowprobe.parsers.exceptions import (
    BatchProcessingError,
    KnowledgeParserError,
    ParseError,
    UnsupportedFormatError,
    ValidationError,
)


class TestKnowledgeParserError:
    """Test base exception class."""

    def test_basic_init(self) -> None:
        """Test basic initialization."""
        exc = KnowledgeParserError("test message")
        assert str(exc) == "test message"
        assert exc.source_id is None
        assert exc.details == {}

    def test_full_init(self) -> None:
        """Test initialization with all parameters."""
        exc = KnowledgeParserError("test", source_id="src_123", details={"key": "value"})
        assert exc.source_id == "src_123"
        assert exc.details == {"key": "value"}


class TestUnsupportedFormatError:
    """Test unsupported format exception."""

    def test_basic(self) -> None:
        """Test basic unsupported format."""
        exc = UnsupportedFormatError("custom_type")
        assert exc.input_type == "custom_type"
        assert "custom_type" in str(exc)

    def test_with_supported(self) -> None:
        """Test with supported types list."""
        exc = UnsupportedFormatError("x", supported=["a", "b", "c"])
        assert exc.supported == ["a", "b", "c"]
        assert "a, b, c" in str(exc)


class TestParseError:
    """Test parse error exception."""

    def test_init(self) -> None:
        """Test parse error initialization."""
        exc = ParseError("parse failed", raw_content="test content", parser_name="TestParser")
        assert exc.raw_content == "test content"
        assert exc.parser_name == "TestParser"


class TestValidationError:
    """Test validation error exception."""

    def test_init(self) -> None:
        """Test validation error initialization."""
        exc = ValidationError("invalid", field="content", value="bad")
        assert exc.field == "content"
        assert exc.value == "bad"


class TestBatchProcessingError:
    """Test batch processing error."""

    def test_init(self) -> None:
        """Test batch error with sub-errors."""
        errors = [ParseError("e1"), ParseError("e2")]
        exc = BatchProcessingError("batch failed", errors=errors)
        assert len(exc.errors) == 2

    def test_empty_errors(self) -> None:
        """Test batch error with no sub-errors."""
        exc = BatchProcessingError("batch failed")
        assert exc.errors == []
