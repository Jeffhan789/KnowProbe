"""Tests for KnowledgeInputProcessor."""

import pytest

from knowprobe.core.models import KnowledgeInput
from knowprobe.parsers.exceptions import (
    BatchProcessingError,
    ParseError,
    UnsupportedFormatError,
    ValidationError,
)
from knowprobe.parsers.knowledge_processor import KnowledgeInputProcessor


class TestKnowledgeInputProcessor:
    """Test main processor."""

    @pytest.fixture
    def processor(self) -> KnowledgeInputProcessor:
        """Create processor fixture."""
        return KnowledgeInputProcessor()

    def test_process_triple(self, processor: KnowledgeInputProcessor) -> None:
        """Test processing triple."""
        result = processor.process("Alice | knows | Bob", "triple")
        assert isinstance(result, KnowledgeInput)
        assert result.input_type == "triple"
        assert result.structured["triple_count"] == 1

    def test_process_schema(self, processor: KnowledgeInputProcessor) -> None:
        """Test processing schema."""
        result = processor.process("Person:age -> 30", "schema")
        assert result.input_type == "schema"
        assert result.structured["entry_count"] == 1

    def test_process_text(self, processor: KnowledgeInputProcessor) -> None:
        """Test processing text."""
        result = processor.process("Alice and Bob are friends.", "text")
        assert result.input_type == "text"

    def test_process_entity(self, processor: KnowledgeInputProcessor) -> None:
        """Test processing entity."""
        result = processor.process("Alice [Person]", "entity")
        assert result.input_type == "entity"

    def test_unsupported_type_raises(self, processor: KnowledgeInputProcessor) -> None:
        """Test unsupported type raises."""
        with pytest.raises(UnsupportedFormatError):
            processor.process("test", "unknown_type")

    def test_with_source_id(self, processor: KnowledgeInputProcessor) -> None:
        """Test custom source_id."""
        result = processor.process("Alice | knows | Bob", "triple", source_id="my_id")
        assert result.source_id == "my_id"

    def test_with_metadata(self, processor: KnowledgeInputProcessor) -> None:
        """Test metadata pass-through."""
        meta = {"experiment": "test"}
        result = processor.process("Alice | knows | Bob", "triple", metadata=meta)
        assert result.metadata == meta

    def test_validation_failure_strict(self) -> None:
        """Test strict mode with parse failure for empty content."""
        processor = KnowledgeInputProcessor(strict_mode=True)
        with pytest.raises(ParseError):
            processor.process("", "triple")

    def test_validation_failure_non_strict(self) -> None:
        """Test non-strict mode still raises ParseError for unparsable content."""
        processor = KnowledgeInputProcessor(strict_mode=False)
        with pytest.raises(ParseError):
            processor.process("", "triple")

    def test_disable_validation(self, processor: KnowledgeInputProcessor) -> None:
        """Test disabling validation still requires parseable content."""
        with pytest.raises(ParseError):
            processor.process("", "triple", validate=False)

    def test_process_dict(self, processor: KnowledgeInputProcessor) -> None:
        """Test process_dict method."""
        data = {"content": "Alice | knows | Bob", "input_type": "triple"}
        result = processor.process_dict(data)
        assert result.input_type == "triple"

    def test_supported_types(self, processor: KnowledgeInputProcessor) -> None:
        """Test supported types list."""
        types = processor.supported_types
        assert "triple" in types
        assert "schema" in types
        assert "text" in types
        assert "entity" in types


class TestBatchProcessing:
    """Test batch processing."""

    @pytest.fixture
    def processor(self) -> KnowledgeInputProcessor:
        """Create processor fixture."""
        return KnowledgeInputProcessor()

    def test_batch_success(self, processor: KnowledgeInputProcessor) -> None:
        """Test successful batch."""
        items = [
            {"content": "Alice | knows | Bob", "input_type": "triple"},
            {"content": "Person:age -> 30", "input_type": "schema"},
            {"content": "Hello world.", "input_type": "text"},
        ]
        results = processor.process_batch(items)
        assert len(results) == 3

    def test_batch_fail_fast(self, processor: KnowledgeInputProcessor) -> None:
        """Test fail-fast batch."""
        items = [
            {"content": "Alice | knows | Bob", "input_type": "triple"},
            {"content": "", "input_type": "triple"},  # Will fail validation
        ]
        with pytest.raises(BatchProcessingError):
            processor.process_batch(items, fail_fast=True)

    def test_batch_continue_on_error(self, processor: KnowledgeInputProcessor) -> None:
        """Test continue on error."""
        items = [
            {"content": "Alice | knows | Bob", "input_type": "triple"},
            {"content": "bad format no triple here", "input_type": "triple"},
        ]
        results = processor.process_batch(items, fail_fast=False)
        assert len(results) == 1  # Only first succeeds

    def test_batch_empty(self, processor: KnowledgeInputProcessor) -> None:
        """Test empty batch."""
        results = processor.process_batch([])
        assert results == []


class TestAutoDetection:
    """Test auto-detection."""

    @pytest.fixture
    def processor(self) -> KnowledgeInputProcessor:
        """Create processor fixture."""
        return KnowledgeInputProcessor()

    def test_detect_triple(self, processor: KnowledgeInputProcessor) -> None:
        """Test triple detection."""
        detected = processor.auto_detect_type("(Alice, knows, Bob)")
        assert detected == "triple"

    def test_detect_triple_pipe(self, processor: KnowledgeInputProcessor) -> None:
        """Test pipe triple detection."""
        detected = processor.auto_detect_type("Alice | knows | Bob")
        assert detected == "triple"

    def test_detect_schema(self, processor: KnowledgeInputProcessor) -> None:
        """Test schema detection."""
        detected = processor.auto_detect_type("Person:age -> 30")
        assert detected == "schema"

    def test_detect_entity(self, processor: KnowledgeInputProcessor) -> None:
        """Test entity detection."""
        detected = processor.auto_detect_type("Alice [Person]")
        assert detected == "entity"

    def test_detect_text(self, processor: KnowledgeInputProcessor) -> None:
        """Test text detection."""
        detected = processor.auto_detect_type("This is a long text about Alice and Bob living in New York City.")
        assert detected == "text"

    def test_process_auto(self, processor: KnowledgeInputProcessor) -> None:
        """Test auto processing."""
        result = processor.process_auto("Alice | knows | Bob")
        assert result.input_type == "triple"
        assert result.structured["triple_count"] == 1

    def test_process_auto_text(self, processor: KnowledgeInputProcessor) -> None:
        """Test auto processing text."""
        result = processor.process_auto("Alice and Bob are friends. They live in New York.")
        assert result.input_type == "text"
