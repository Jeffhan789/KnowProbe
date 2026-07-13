"""Tests for knowledge parser utility functions."""

from knowprobe.parsers.utils import (
    chunk_text,
    clean_triple_content,
    estimate_token_count,
    extract_entities,
    extract_schema_entries,
    extract_triples,
    generate_source_id,
    is_empty_content,
    normalize_text,
    sanitize_metadata,
)


class TestGenerateSourceId:
    """Test source ID generation."""

    def test_default_prefix(self) -> None:
        """Test default prefix."""
        sid = generate_source_id()
        assert sid.startswith("kp_")
        assert len(sid) > 3

    def test_custom_prefix(self) -> None:
        """Test custom prefix."""
        sid = generate_source_id("test")
        assert sid.startswith("test_")

    def test_uniqueness(self) -> None:
        """Test IDs are unique."""
        sids = {generate_source_id() for _ in range(100)}
        assert len(sids) == 100


class TestNormalizeText:
    """Test text normalization."""

    def test_whitespace(self) -> None:
        """Test whitespace collapsing."""
        assert normalize_text("  hello   world  ") == "hello world"

    def test_newlines(self) -> None:
        """Test newline normalization."""
        assert normalize_text("hello\n\n\nworld") == "hello world"

    def test_empty(self) -> None:
        """Test empty string."""
        assert normalize_text("") == ""


class TestCleanTripleContent:
    """Test triple content cleaning."""

    def test_parentheses(self) -> None:
        """Test parenthesis removal."""
        cleaned = clean_triple_content("(Alice, knows, Bob)")
        assert "Alice" in cleaned
        assert "(" not in cleaned
        assert ")" not in cleaned

    def test_pipe_separators(self) -> None:
        """Test pipe normalization."""
        cleaned = clean_triple_content("Alice, knows, Bob")
        assert "|" in cleaned


class TestExtractTriples:
    """Test triple extraction."""

    def test_parenthesized_format(self) -> None:
        """Test (S, P, O) format."""
        triples = extract_triples("(Alice, knows, Bob)")
        assert len(triples) == 1
        assert triples[0]["subject"] == "Alice"
        assert triples[0]["predicate"] == "knows"
        assert triples[0]["object"] == "Bob"

    def test_pipe_format(self) -> None:
        """Test S|P|O format."""
        triples = extract_triples("Alice | knows | Bob")
        assert len(triples) == 1
        assert triples[0]["subject"] == "Alice"

    def test_multiple(self) -> None:
        """Test multiple triples."""
        triples = extract_triples("(A, B, C) and (D, E, F)")
        assert len(triples) == 2

    def test_no_triples(self) -> None:
        """Test empty result."""
        triples = extract_triples("just random text")
        assert triples == []


class TestExtractSchemaEntries:
    """Test schema entry extraction."""

    def test_basic_schema(self) -> None:
        """Test basic schema format."""
        entries = extract_schema_entries("Person:age -> 30")
        assert len(entries) == 1
        assert entries[0]["entity_type"] == "Person"
        assert entries[0]["property"] == "age"
        assert entries[0]["value"] == "30"

    def test_multi_line(self) -> None:
        """Test multi-line schema."""
        content = "Person:age -> 30\nPerson:name -> Alice"
        entries = extract_schema_entries(content)
        assert len(entries) == 2

    def test_no_match(self) -> None:
        """Test no matching entries."""
        entries = extract_schema_entries("random text")
        assert entries == []


class TestExtractEntities:
    """Test entity extraction."""

    def test_basic_entity(self) -> None:
        """Test basic entity parsing."""
        entities = extract_entities("Alice [Person]")
        assert len(entities) == 1
        assert entities[0]["name"] == "Alice"
        assert entities[0]["type"] == "Person"

    def test_entity_with_properties(self) -> None:
        """Test entity with properties."""
        entities = extract_entities("Alice [Person] {age: 30, city: NYC}")
        assert len(entities) == 1
        assert entities[0]["properties"]["age"] == "30"
        assert entities[0]["properties"]["city"] == "NYC"

    def test_no_match(self) -> None:
        """Test empty result."""
        entities = extract_entities("")
        assert entities == []


class TestChunkText:
    """Test text chunking."""

    def test_basic_chunking(self) -> None:
        """Test basic chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0
        assert all(len(c) <= 60 for c in chunks)  # chunk_size + some sentence overhead

    def test_short_text(self) -> None:
        """Test short text returns single chunk."""
        chunks = chunk_text("Short text.")
        assert len(chunks) == 1


class TestEstimateTokenCount:
    """Test token count estimation."""

    def test_basic(self) -> None:
        """Test basic estimation."""
        assert estimate_token_count("hello world") == 2  # 11 chars / 4

    def test_empty(self) -> None:
        """Test empty string."""
        assert estimate_token_count("") == 0


class TestIsEmptyContent:
    """Test empty content detection."""

    def test_empty(self) -> None:
        """Test empty string."""
        assert is_empty_content("") is True

    def test_whitespace(self) -> None:
        """Test whitespace-only."""
        assert is_empty_content("   ") is True

    def test_null_like(self) -> None:
        """Test null-like values."""
        assert is_empty_content("null") is True
        assert is_empty_content("None") is True

    def test_valid(self) -> None:
        """Test valid content."""
        assert is_empty_content("hello") is False


class TestSanitizeMetadata:
    """Test metadata sanitization."""

    def test_basic(self) -> None:
        """Test basic sanitization."""
        meta = {"a": 1, "b": "str", "c": [1, 2, 3]}
        clean = sanitize_metadata(meta)
        assert clean == meta

    def test_nested_dict(self) -> None:
        """Test nested dict."""
        meta = {"nested": {"key": "value"}}
        clean = sanitize_metadata(meta)
        assert clean == meta

    def test_unsupported_types(self) -> None:
        """Test unsupported type conversion."""
        meta = {"obj": set([1, 2, 3])}
        clean = sanitize_metadata(meta)
        assert isinstance(clean["obj"], str)
