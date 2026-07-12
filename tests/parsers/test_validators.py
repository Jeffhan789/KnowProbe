"""Tests for knowledge input validators."""

import pytest

from knowprobe.core.models import KnowledgeInput
from knowprobe.parsers.exceptions import ValidationError
from knowprobe.parsers.validators import (
    CompositeValidator,
    FormatValidator,
    SemanticValidator,
)


class TestFormatValidator:
    """Test format validation."""

    def test_valid_knowledge_input(self) -> None:
        """Test valid input passes."""
        validator = FormatValidator()
        data = KnowledgeInput(
            source_id="test_1",
            input_type="triple",
            content="Alice | knows | Bob",
        )
        validator.validate(data)  # Should not raise

    def test_empty_source_id(self) -> None:
        """Test empty source_id fails."""
        validator = FormatValidator()
        data = {"source_id": "", "input_type": "triple", "content": "test"}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "source_id"

    def test_invalid_input_type(self) -> None:
        """Test invalid input_type fails."""
        validator = FormatValidator()
        data = {"source_id": "test", "input_type": "unknown", "content": "test"}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "input_type"

    def test_empty_content(self) -> None:
        """Test empty content fails."""
        validator = FormatValidator()
        data = {"source_id": "test", "input_type": "triple", "content": ""}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "content"

    def test_content_too_long(self) -> None:
        """Test content exceeding max length fails."""
        validator = FormatValidator()
        data = {"source_id": "test", "input_type": "triple", "content": "x" * 200_000}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "content"

    def test_dict_input(self) -> None:
        """Test dict input validation."""
        validator = FormatValidator()
        data = {"source_id": "test", "input_type": "text", "content": "hello world"}
        validator.validate(data)  # Should not raise


class TestSemanticValidator:
    """Test semantic validation."""

    def test_valid_triple(self) -> None:
        """Test valid triple semantic."""
        validator = SemanticValidator()
        data = KnowledgeInput(
            source_id="test",
            input_type="triple",
            content="Alice | knows | Bob",
            structured={"triples": [{"subject": "Alice", "predicate": "knows", "object": "Bob"}]},
        )
        validator.validate(data)  # Should not raise

    def test_triple_missing_fields(self) -> None:
        """Test triple missing required fields."""
        validator = SemanticValidator()
        data = {
            "source_id": "test",
            "input_type": "triple",
            "content": "Alice | knows | Bob",
            "structured": {"triples": [{"subject": "Alice"}]},
        }
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert "triples[0]" in str(exc_info.value.field)

    def test_schema_without_arrow(self) -> None:
        """Test schema without arrow separator."""
        validator = SemanticValidator()
        data = {"source_id": "test", "input_type": "schema", "content": "random text"}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "content"

    def test_text_too_short(self) -> None:
        """Test text too short."""
        validator = SemanticValidator()
        data = {"source_id": "test", "input_type": "text", "content": "ab"}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "content"

    def test_entity_without_name(self) -> None:
        """Test entity without name."""
        validator = SemanticValidator()
        data = {
            "source_id": "test",
            "input_type": "entity",
            "content": "Alice",
            "structured": {"entities": [{"name": ""}]},
        }
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert "entities" in str(exc_info.value.field)


class TestCompositeValidator:
    """Test composite validator."""

    def test_valid(self) -> None:
        """Test valid input passes all validators."""
        validator = CompositeValidator()
        data = KnowledgeInput(
            source_id="test",
            input_type="triple",
            content="Alice | knows | Bob",
        )
        validator.validate(data)  # Should not raise

    def test_first_validator_fails(self) -> None:
        """Test first validator failure stops pipeline."""
        validator = CompositeValidator()
        data = {"source_id": "", "input_type": "triple", "content": "test"}
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data)
        assert exc_info.value.field == "source_id"

    def test_custom_validators(self) -> None:
        """Test custom validator list."""
        validator = CompositeValidator([FormatValidator()])
        data = {"source_id": "test", "input_type": "triple", "content": "x"}
        validator.validate(data)  # Should pass format only
