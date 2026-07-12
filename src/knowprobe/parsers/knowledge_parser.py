"""Knowledge input parsers: abstract base and concrete implementations."""

from abc import ABC, abstractmethod
from typing import Any

from knowprobe.core.models import KnowledgeInput
from knowprobe.parsers.exceptions import ParseError, UnsupportedFormatError
from knowprobe.parsers.utils import (
    chunk_text,
    clean_triple_content,
    extract_entities,
    extract_schema_entries,
    extract_triples,
    generate_source_id,
    normalize_text,
)
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeParser(ABC):
    """Abstract base class for all knowledge input parsers."""

    def __init__(self, input_type: str) -> None:
        """Initialize parser with its supported input type."""
        self._input_type = input_type

    @property
    def input_type(self) -> str:
        """Return the parser's supported input type."""
        return self._input_type

    @abstractmethod
    def parse(self, content: str, source_id: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeInput:
        """Parse raw content into a structured KnowledgeInput.

        Args:
            content: Raw knowledge content to parse.
            source_id: Optional identifier for the knowledge source.
            metadata: Optional metadata dictionary.

        Returns:
            A validated KnowledgeInput instance.

        Raises:
            ParseError: If parsing fails.
        """
        raise NotImplementedError

    def _create_input(self, content: str, structured: dict[str, Any], source_id: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeInput:
        """Create a KnowledgeInput with standardized defaults."""
        return KnowledgeInput(
            source_id=source_id or generate_source_id(),
            input_type=self._input_type,
            content=content,
            structured=structured,
            metadata=metadata or {},
        )

    def _log_parse(self, source_id: str | None, content_preview: str) -> None:
        """Log parsing attempt."""
        logger.info(
            "parsing_knowledge_input",
            parser=self.__class__.__name__,
            input_type=self._input_type,
            source_id=source_id,
            content_preview=content_preview[:200],
        )

    def _log_parse_success(self, source_id: str, structured_keys: list[str]) -> None:
        """Log successful parsing."""
        logger.info(
            "parse_success",
            parser=self.__class__.__name__,
            source_id=source_id,
            structured_keys=structured_keys,
        )

    def _log_parse_error(self, source_id: str | None, error: str) -> None:
        """Log parsing failure."""
        logger.error(
            "parse_error",
            parser=self.__class__.__name__,
            input_type=self._input_type,
            source_id=source_id,
            error=error,
        )


class TripleParser(KnowledgeParser):
    """Parse knowledge triples: (subject, predicate, object) format.

    Supports formats:
        - (Subject, Predicate, Object)
        - Subject | Predicate | Object
        - Subject, Predicate, Object
    """

    def __init__(self) -> None:
        """Initialize triple parser."""
        super().__init__("triple")

    def parse(self, content: str, source_id: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeInput:
        """Parse triple content into KnowledgeInput.

        Args:
            content: Raw triple content.
            source_id: Optional source identifier.
            metadata: Optional metadata.

        Returns:
            KnowledgeInput with structured triples.

        Raises:
            ParseError: If no valid triples can be extracted.
        """
        self._log_parse(source_id, content)

        triples = extract_triples(content)

        if not triples:
            self._log_parse_error(source_id, "No valid triples found in content")
            raise ParseError(
                "No valid triples could be extracted from content",
                raw_content=content[:500],
                parser_name=self.__class__.__name__,
                source_id=source_id,
            )

        structured = {
            "triples": triples,
            "triple_count": len(triples),
            "unique_subjects": list({t["subject"] for t in triples}),
            "unique_predicates": list({t["predicate"] for t in triples}),
            "unique_objects": list({t["object"] for t in triples}),
        }

        # Use normalized content for storage but preserve original for parsing
        normalized = normalize_text(content)
        result = self._create_input(normalized, structured, source_id, metadata)
        self._log_parse_success(result.source_id, list(structured.keys()))
        return result


class SchemaParser(KnowledgeParser):
    """Parse schema/ontology definitions.

    Supports format:
        - EntityType:Property -> Value
        - Multi-line entries with one per line
    """

    def __init__(self) -> None:
        """Initialize schema parser."""
        super().__init__("schema")

    def parse(self, content: str, source_id: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeInput:
        """Parse schema content into KnowledgeInput.

        Args:
            content: Raw schema content.
            source_id: Optional source identifier.
            metadata: Optional metadata.

        Returns:
            KnowledgeInput with structured schema entries.

        Raises:
            ParseError: If no valid schema entries can be extracted.
        """
        self._log_parse(source_id, content)
        # Preserve newlines for multi-line parsing; only strip outer whitespace
        stripped = content.strip()

        entries = extract_schema_entries(stripped)

        if not entries:
            self._log_parse_error(source_id, "No valid schema entries found")
            raise ParseError(
                "No valid schema entries could be extracted from content",
                raw_content=content[:500],
                parser_name=self.__class__.__name__,
                source_id=source_id,
            )

        # Build entity-type indexed structure
        entity_types: dict[str, list[dict[str, str]]] = {}
        for entry in entries:
            et = entry["entity_type"]
            entity_types.setdefault(et, []).append(entry)

        structured = {
            "entries": entries,
            "entry_count": len(entries),
            "entity_types": list(entity_types.keys()),
            "properties_by_type": {
                et: list({e["property"] for e in ents})
                for et, ents in entity_types.items()
            },
        }

        normalized = normalize_text(content)
        result = self._create_input(normalized, structured, source_id, metadata)
        self._log_parse_success(result.source_id, list(structured.keys()))
        return result


class TextParser(KnowledgeParser):
    """Parse free-text knowledge into structured chunks.

    Performs sentence segmentation, chunking, and basic NLP extraction.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize text parser with chunking parameters.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        super().__init__("text")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def parse(self, content: str, source_id: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeInput:
        """Parse text content into KnowledgeInput with chunks.

        Args:
            content: Raw text content.
            source_id: Optional source identifier.
            metadata: Optional metadata.

        Returns:
            KnowledgeInput with structured text chunks.

        Raises:
            ParseError: If text content is empty or too short.
        """
        self._log_parse(source_id, content)
        normalized = normalize_text(content)

        if not normalized or len(normalized) < 3:
            self._log_parse_error(source_id, "Text content is too short or empty")
            raise ParseError(
                "Text content is too short or empty after normalization",
                raw_content=content[:500],
                parser_name=self.__class__.__name__,
                source_id=source_id,
            )

        chunks = chunk_text(normalized, self._chunk_size, self._chunk_overlap)

        structured = {
            "chunks": chunks,
            "chunk_count": len(chunks),
            "original_length": len(content),
            "normalized_length": len(normalized),
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }

        result = self._create_input(normalized, structured, source_id, metadata)
        self._log_parse_success(result.source_id, list(structured.keys()))
        return result


class EntityParser(KnowledgeParser):
    """Parse entity definitions with optional type and properties.

    Supports format:
        - EntityName [Type] {prop1: val1, prop2: val2}
        - Simple entity names
    """

    def __init__(self) -> None:
        """Initialize entity parser."""
        super().__init__("entity")

    def parse(self, content: str, source_id: str | None = None, metadata: dict[str, Any] | None = None) -> KnowledgeInput:
        """Parse entity content into KnowledgeInput.

        Args:
            content: Raw entity content.
            source_id: Optional source identifier.
            metadata: Optional metadata.

        Returns:
            KnowledgeInput with structured entity definitions.

        Raises:
            ParseError: If no valid entities can be extracted.
        """
        self._log_parse(source_id, content)
        # Preserve newlines for multi-line parsing; only strip outer whitespace
        stripped = content.strip()

        entities = extract_entities(stripped)

        if not entities:
            # Fallback: treat entire content as a single named entity
            entities = [{"name": stripped}]

        # Collect all entity types and properties
        all_types: set[str] = set()
        all_properties: set[str] = set()
        for entity in entities:
            if "type" in entity:
                all_types.add(entity["type"])
            if "properties" in entity:
                all_properties.update(entity["properties"].keys())

        structured = {
            "entities": entities,
            "entity_count": len(entities),
            "entity_types": sorted(all_types),
            "property_keys": sorted(all_properties),
        }

        normalized = normalize_text(content)
        result = self._create_input(normalized, structured, source_id, metadata)
        self._log_parse_success(result.source_id, list(structured.keys()))
        return result


class ParserRegistry:
    """Registry for knowledge parsers supporting factory and lookup patterns."""

    _parsers: dict[str, KnowledgeParser] = {}

    @classmethod
    def register(cls, parser: KnowledgeParser) -> None:
        """Register a parser instance."""
        cls._parsers[parser.input_type] = parser
        logger.info("parser_registered", parser_type=parser.input_type, parser_class=parser.__class__.__name__)

    @classmethod
    def get(cls, input_type: str) -> KnowledgeParser:
        """Get a parser by input type.

        Args:
            input_type: The input type string.

        Returns:
            The registered parser for this type.

        Raises:
            UnsupportedFormatError: If no parser is registered for this type.
        """
        if input_type not in cls._parsers:
            raise UnsupportedFormatError(
                input_type,
                supported=list(cls._parsers.keys()),
            )
        return cls._parsers[input_type]

    @classmethod
    def list_types(cls) -> list[str]:
        """Return all registered input types."""
        return list(cls._parsers.keys())

    @classmethod
    def create_default_registry(cls) -> None:
        """Register all built-in parsers."""
        cls.register(TripleParser())
        cls.register(SchemaParser())
        cls.register(TextParser())
        cls.register(EntityParser())
        logger.info("default_parser_registry_initialized")


# Initialize default registry on module import
ParserRegistry.create_default_registry()
