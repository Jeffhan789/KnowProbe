"""Tests for knowledge parsers (Triple, Schema, Text, Entity)."""

import pytest

from knowprobe.core.models import KnowledgeInput
from knowprobe.parsers.exceptions import ParseError, UnsupportedFormatError
from knowprobe.parsers.knowledge_parser import (
    EntityParser,
    ParserRegistry,
    SchemaParser,
    TextParser,
    TripleParser,
)


class TestTripleParser:
    """Test triple parser."""

    @pytest.fixture
    def parser(self) -> TripleParser:
        """Create triple parser fixture."""
        return TripleParser()

    def test_parenthesized_format(self, parser: TripleParser) -> None:
        """Test (S, P, O) format."""
        result = parser.parse("(Alice, knows, Bob)")
        assert isinstance(result, KnowledgeInput)
        assert result.input_type == "triple"
        assert result.structured["triple_count"] == 1
        assert result.structured["triples"][0]["subject"] == "Alice"
        assert result.structured["triples"][0]["predicate"] == "knows"
        assert result.structured["triples"][0]["object"] == "Bob"

    def test_pipe_format(self, parser: TripleParser) -> None:
        """Test S|P|O format."""
        result = parser.parse("Alice | knows | Bob")
        assert result.structured["triple_count"] == 1

    def test_multiple_triples(self, parser: TripleParser) -> None:
        """Test multiple triples."""
        result = parser.parse("(Alice, knows, Bob) (Bob, likes, Carol)")
        assert result.structured["triple_count"] == 2
        subjects = result.structured["unique_subjects"]
        assert "Alice" in subjects
        assert "Bob" in subjects

    def test_with_source_id(self, parser: TripleParser) -> None:
        """Test with custom source_id."""
        result = parser.parse("Alice | knows | Bob", source_id="custom_id")
        assert result.source_id == "custom_id"

    def test_with_metadata(self, parser: TripleParser) -> None:
        """Test with metadata."""
        meta = {"domain": "test"}
        result = parser.parse("Alice | knows | Bob", metadata=meta)
        assert result.metadata == meta

    def test_empty_content_raises(self, parser: TripleParser) -> None:
        """Test empty content raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("random text with no triples")

    def test_complex_triple(self, parser: TripleParser) -> None:
        """Test complex entity names."""
        result = parser.parse("(New York City, located_in, United States)")
        assert result.structured["triples"][0]["subject"] == "New York City"


class TestSchemaParser:
    """Test schema parser."""

    @pytest.fixture
    def parser(self) -> SchemaParser:
        """Create schema parser fixture."""
        return SchemaParser()

    def test_basic_schema(self, parser: SchemaParser) -> None:
        """Test basic schema entry."""
        result = parser.parse("Person:age -> 30")
        assert result.input_type == "schema"
        assert result.structured["entry_count"] == 1
        assert "Person" in result.structured["entity_types"]

    def test_multiple_entries(self, parser: SchemaParser) -> None:
        """Test multiple schema entries."""
        content = "Person:age -> 30\nPerson:name -> Alice\nCompany:founder -> Bob"
        result = parser.parse(content)
        assert result.structured["entry_count"] == 3
        assert set(result.structured["entity_types"]) == {"Person", "Company"}

    def test_properties_by_type(self, parser: SchemaParser) -> None:
        """Test properties grouped by type."""
        result = parser.parse("Person:age -> 30\nPerson:name -> Alice")
        props = result.structured["properties_by_type"]
        assert "age" in props["Person"]
        assert "name" in props["Person"]

    def test_invalid_content_raises(self, parser: SchemaParser) -> None:
        """Test invalid content raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("random text without schema format")


class TestTextParser:
    """Test text parser."""

    @pytest.fixture
    def parser(self) -> TextParser:
        """Create text parser fixture."""
        return TextParser()

    def test_basic_text(self, parser: TextParser) -> None:
        """Test basic text parsing."""
        result = parser.parse("This is a simple text about Alice and Bob.")
        assert result.input_type == "text"
        assert result.structured["chunk_count"] >= 1

    def test_chunking(self, parser: TextParser) -> None:
        """Test text chunking."""
        long_text = "This is sentence one. " * 100
        result = parser.parse(long_text)
        assert result.structured["chunk_count"] > 1

    def test_custom_chunk_size(self) -> None:
        """Test custom chunk parameters."""
        parser = TextParser(chunk_size=100, chunk_overlap=10)
        long_text = "Word. " * 50
        result = parser.parse(long_text)
        assert result.structured["chunk_size"] == 100

    def test_short_text_raises(self, parser: TextParser) -> None:
        """Test very short text raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("ab")

    def test_metadata_in_structured(self, parser: TextParser) -> None:
        """Test length metadata."""
        text = "Alice and Bob are friends. They live in New York City."
        result = parser.parse(text)
        assert "original_length" in result.structured
        assert "normalized_length" in result.structured


class TestEntityParser:
    """Test entity parser."""

    @pytest.fixture
    def parser(self) -> EntityParser:
        """Create entity parser fixture."""
        return EntityParser()

    def test_basic_entity(self, parser: EntityParser) -> None:
        """Test basic entity."""
        result = parser.parse("Alice")
        assert result.input_type == "entity"
        assert result.structured["entity_count"] == 1
        assert result.structured["entities"][0]["name"] == "Alice"

    def test_entity_with_type(self, parser: EntityParser) -> None:
        """Test entity with type."""
        result = parser.parse("Alice [Person]")
        assert result.structured["entities"][0]["type"] == "Person"
        assert "Person" in result.structured["entity_types"]

    def test_entity_with_properties(self, parser: EntityParser) -> None:
        """Test entity with properties."""
        result = parser.parse("Alice [Person] {age: 30, city: NYC}")
        props = result.structured["entities"][0]["properties"]
        assert props["age"] == "30"
        assert props["city"] == "NYC"
        assert "age" in result.structured["property_keys"]

    def test_multiple_entities(self, parser: EntityParser) -> None:
        """Test multiple entities."""
        result = parser.parse("Alice [Person]\nBob [Person]\nGoogle [Company]")
        assert result.structured["entity_count"] == 3
        types = result.structured["entity_types"]
        assert "Person" in types
        assert "Company" in types


class TestParserRegistry:
    """Test parser registry."""

    def test_get_registered(self) -> None:
        """Test getting registered parser."""
        parser = ParserRegistry.get("triple")
        assert isinstance(parser, TripleParser)

    def test_get_unregistered_raises(self) -> None:
        """Test getting unregistered parser raises."""
        with pytest.raises(UnsupportedFormatError):
            ParserRegistry.get("nonexistent")

    def test_list_types(self) -> None:
        """Test listing registered types."""
        types = ParserRegistry.list_types()
        assert "triple" in types
        assert "schema" in types
        assert "text" in types
        assert "entity" in types

    def test_register_custom(self) -> None:
        """Test registering custom parser."""
        custom = TextParser()
        custom._input_type = "custom_text"
        ParserRegistry.register(custom)
        assert "custom_text" in ParserRegistry.list_types()
        retrieved = ParserRegistry.get("custom_text")
        assert retrieved is custom
