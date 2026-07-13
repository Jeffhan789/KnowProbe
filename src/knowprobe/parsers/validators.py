"""Input validation for knowledge processing pipeline."""

from typing import Any

from knowprobe.core.models import KnowledgeInput
from knowprobe.parsers.exceptions import ValidationError
from knowprobe.parsers.utils import is_empty_content


class InputValidator:
    """Base validator for knowledge input."""

    def __init__(self) -> None:
        """Initialize validator."""
        self._errors: list[ValidationError] = []

    def validate(self, data: KnowledgeInput | dict[str, Any]) -> None:
        """Validate input data. Raises ValidationError on failure."""
        raise NotImplementedError

    @property
    def errors(self) -> list[ValidationError]:
        """Return accumulated validation errors."""
        return self._errors.copy()

    def _clear_errors(self) -> None:
        """Clear accumulated errors."""
        self._errors = []


class FormatValidator(InputValidator):
    """Validate basic format requirements for knowledge input."""

    VALID_INPUT_TYPES = {"triple", "schema", "text", "entity"}
    MAX_CONTENT_LENGTH = 100_000
    MAX_METADATA_SIZE = 10_000

    def validate(self, data: KnowledgeInput | dict[str, Any]) -> None:
        """Validate format constraints.

        Raises:
            ValidationError: If any format constraint is violated.
        """
        self._clear_errors()

        if isinstance(data, KnowledgeInput):
            raw = data.model_dump()
        else:
            raw = data

        # Validate source_id
        source_id = raw.get("source_id")
        if not source_id or not isinstance(source_id, str):
            self._raise("source_id must be a non-empty string", field="source_id", value=source_id)
        elif len(source_id) > 256:
            self._raise(
                "source_id exceeds maximum length of 256", field="source_id", value=source_id
            )

        # Validate input_type
        input_type = raw.get("input_type", "triple")
        if input_type not in self.VALID_INPUT_TYPES:
            self._raise(
                f"input_type must be one of {self.VALID_INPUT_TYPES}",
                field="input_type",
                value=input_type,
            )

        # Validate content
        content = raw.get("content")
        if not isinstance(content, str) or is_empty_content(content):
            self._raise("content cannot be empty", field="content", value=content)
        elif len(content) > self.MAX_CONTENT_LENGTH:
            self._raise(
                f"content exceeds maximum length of {self.MAX_CONTENT_LENGTH}",
                field="content",
                value=content[:50] + "...",
            )

        # Validate metadata size
        metadata = raw.get("metadata", {})
        metadata_str = str(metadata)
        if len(metadata_str) > self.MAX_METADATA_SIZE:
            self._raise(
                f"metadata exceeds maximum size of {self.MAX_METADATA_SIZE} chars",
                field="metadata",
            )

        if self._errors:
            raise self._errors[0]

    def _raise(self, message: str, *, field: str | None = None, value: Any = None) -> None:
        """Record a validation error."""
        self._errors.append(ValidationError(message, field=field, value=value))


class SemanticValidator(InputValidator):
    """Validate semantic content of knowledge input."""

    MIN_TRIPLE_PARTS = 3
    MAX_TRIPLE_PARTS = 3
    MIN_SENTENCE_LENGTH = 3
    MAX_ENTITY_NAME_LENGTH = 200

    def validate(self, data: KnowledgeInput | dict[str, Any]) -> None:
        """Validate semantic constraints based on input type.

        Raises:
            ValidationError: If semantic validation fails.
        """
        self._clear_errors()

        if isinstance(data, KnowledgeInput):
            input_type = data.input_type
            content = data.content
            structured = data.structured
        else:
            input_type = data.get("input_type", "triple")
            content = data.get("content", "")
            structured = data.get("structured", {})

        if input_type == "triple":
            self._validate_triple(content, structured)
        elif input_type == "schema":
            self._validate_schema(content, structured)
        elif input_type == "text":
            self._validate_text(content)
        elif input_type == "entity":
            self._validate_entity(content, structured)

        if self._errors:
            raise self._errors[0]

    def _validate_triple(self, content: str, structured: dict[str, Any]) -> None:
        """Validate triple format semantically."""
        # If structured data is present, validate it
        if structured:
            triples = structured.get("triples", [])
            if not triples:
                self._raise("structured triples cannot be empty", field="structured.triples")
            for i, triple in enumerate(triples):
                if not all(k in triple for k in ("subject", "predicate", "object")):
                    self._raise(
                        f"triple at index {i} missing required fields",
                        field=f"structured.triples[{i}]",
                        value=triple,
                    )
        else:
            # Validate raw content has at least one parseable triple
            parts = [p.strip() for p in content.split("|") if p.strip()]
            if len(parts) < self.MIN_TRIPLE_PARTS:
                self._raise(
                    f"triple content must have at least {self.MIN_TRIPLE_PARTS} parts separated by '|'",
                    field="content",
                    value=content[:100],
                )

    def _validate_schema(self, content: str, structured: dict[str, Any]) -> None:
        """Validate schema format semantically."""
        if structured:
            entries = structured.get("entries", [])
            if not entries:
                self._raise("structured schema entries cannot be empty", field="structured.entries")
        else:
            if "->" not in content and ":" not in content:
                self._raise(
                    "schema content must contain '->' or ':' separator",
                    field="content",
                    value=content[:100],
                )

    def _validate_text(self, content: str) -> None:
        """Validate text content semantically."""
        # Must contain at least one sentence-like structure or meet minimum length
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        if len(sentences) < 1 or len(content) < self.MIN_SENTENCE_LENGTH:
            self._raise(
                f"text content must be at least {self.MIN_SENTENCE_LENGTH} characters or contain a sentence",
                field="content",
                value=content[:100],
            )

    def _validate_entity(self, content: str, structured: dict[str, Any]) -> None:
        """Validate entity format semantically."""
        if structured:
            entities = structured.get("entities", [])
            if not entities:
                self._raise(
                    "entity structured data must contain at least one entity",
                    field="structured.entities",
                )
            for i, entity in enumerate(entities):
                name = entity.get("name")
                if not name or not isinstance(name, str):
                    self._raise(
                        f"entity at index {i} must have a 'name' field",
                        field=f"structured.entities[{i}].name",
                        value=name,
                    )
                elif len(name) > self.MAX_ENTITY_NAME_LENGTH:
                    self._raise(
                        f"entity name exceeds maximum length of {self.MAX_ENTITY_NAME_LENGTH}",
                        field=f"structured.entities[{i}].name",
                        value=name[:50] + "...",
                    )
        else:
            if not content.strip():
                self._raise("entity content cannot be empty", field="content")

    def _raise(self, message: str, *, field: str | None = None, value: Any = None) -> None:
        """Record a validation error."""
        self._errors.append(ValidationError(message, field=field, value=value))


class CompositeValidator(InputValidator):
    """Chain multiple validators together."""

    def __init__(self, validators: list[InputValidator] | None = None) -> None:
        """Initialize with a list of validators."""
        super().__init__()
        self._validators = validators or [FormatValidator(), SemanticValidator()]

    def validate(self, data: KnowledgeInput | dict[str, Any]) -> None:
        """Run all validators in sequence.

        Raises:
            ValidationError: If any validator fails.
        """
        self._clear_errors()

        for validator in self._validators:
            try:
                validator.validate(data)
            except ValidationError as exc:
                self._errors.append(exc)
                raise

    @property
    def validators(self) -> list[InputValidator]:
        """Return configured validators."""
        return self._validators.copy()
