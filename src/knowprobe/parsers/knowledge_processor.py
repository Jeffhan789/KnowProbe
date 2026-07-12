"""Knowledge input processor: orchestrates parsing, validation, and batch processing."""

from __future__ import annotations

from typing import Any

from knowprobe.core.models import KnowledgeInput
from knowprobe.parsers.exceptions import (
    BatchProcessingError,
    ParseError,
    UnsupportedFormatError,
    ValidationError,
)
from knowprobe.parsers.knowledge_parser import ParserRegistry
from knowprobe.parsers.validators import CompositeValidator, FormatValidator, SemanticValidator
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeInputProcessor:
    """Main processor for knowledge input pipeline.

    Orchestrates format detection, parsing, validation, and structured output
    generation for all supported knowledge input types.
    """

    def __init__(
        self,
        validator: CompositeValidator | None = None,
        strict_mode: bool = True,
    ) -> None:
        """Initialize processor with optional custom validator.

        Args:
            validator: Optional composite validator. Defaults to Format+Semantic.
            strict_mode: If True, validation errors halt processing.
        """
        self._validator = validator or CompositeValidator([FormatValidator(), SemanticValidator()])
        self._strict_mode = strict_mode

    @property
    def supported_types(self) -> list[str]:
        """Return list of supported input types."""
        return ParserRegistry.list_types()

    def process(
        self,
        content: str,
        input_type: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> KnowledgeInput:
        """Process a single knowledge input through the full pipeline.

        Pipeline: Parse -> Validate -> Return KnowledgeInput

        Args:
            content: Raw knowledge content.
            input_type: Type of input (triple, schema, text, entity).
            source_id: Optional source identifier.
            metadata: Optional metadata dictionary.
            validate: Whether to run validation after parsing.

        Returns:
            A fully parsed and validated KnowledgeInput.

        Raises:
            UnsupportedFormatError: If input_type is not supported.
            ParseError: If parsing fails.
            ValidationError: If validation fails and strict_mode is True.
        """
        logger.info(
            "processing_knowledge_input",
            input_type=input_type,
            source_id=source_id,
            content_length=len(content),
            validate=validate,
        )

        # Step 1: Get appropriate parser
        parser = ParserRegistry.get(input_type)

        # Step 2: Parse
        try:
            result = parser.parse(content, source_id=source_id, metadata=metadata)
        except ParseError:
            raise
        except Exception as exc:
            logger.error(
                "unexpected_parse_error",
                input_type=input_type,
                source_id=source_id,
                error=str(exc),
            )
            raise ParseError(
                f"Unexpected error during parsing: {exc}",
                raw_content=content[:500],
                parser_name=parser.__class__.__name__,
                source_id=source_id,
            ) from exc

        # Step 3: Validate
        if validate:
            try:
                self._validator.validate(result)
            except ValidationError as exc:
                logger.warning(
                    "validation_failed",
                    source_id=result.source_id,
                    field=exc.field,
                    error=str(exc),
                )
                if self._strict_mode:
                    raise
                # In non-strict mode, attach validation error to metadata
                result.metadata.setdefault("_validation_errors", []).append({
                    "field": exc.field,
                    "message": str(exc),
                })

        logger.info(
            "processing_complete",
            source_id=result.source_id,
            input_type=input_type,
            structured_keys=list(result.structured.keys()),
        )
        return result

    def process_batch(
        self,
        items: list[dict[str, Any]],
        fail_fast: bool = False,
        validate: bool = True,
    ) -> list[KnowledgeInput]:
        """Process a batch of knowledge inputs.

        Args:
            items: List of dicts with keys: content, input_type, source_id?, metadata?.
            fail_fast: If True, stop on first error. Otherwise, collect partial results.
            validate: Whether to run validation.

        Returns:
            List of successfully parsed KnowledgeInput instances.

        Raises:
            BatchProcessingError: If fail_fast=False and some items fail, or if
                                  fail_fast=True and the first failure occurs.
        """
        logger.info(
            "batch_processing_started",
            batch_size=len(items),
            fail_fast=fail_fast,
            validate=validate,
        )

        results: list[KnowledgeInput] = []
        errors: list[KnowledgeParserError] = []

        for i, item in enumerate(items):
            try:
                result = self.process(
                    content=item["content"],
                    input_type=item["input_type"],
                    source_id=item.get("source_id"),
                    metadata=item.get("metadata"),
                    validate=validate,
                )
                results.append(result)
            except (ParseError, UnsupportedFormatError, ValidationError) as exc:
                logger.error(
                    "batch_item_failed",
                    index=i,
                    source_id=item.get("source_id"),
                    error=str(exc),
                )
                errors.append(exc)
                if fail_fast:
                    raise BatchProcessingError(
                        f"Batch failed at item {i}: {exc}",
                        errors=[exc],
                    ) from exc

        if errors and not fail_fast:
            logger.warning(
                "batch_completed_with_errors",
                success_count=len(results),
                error_count=len(errors),
            )

        logger.info(
            "batch_processing_complete",
            success_count=len(results),
            error_count=len(errors),
        )
        return results

    def process_dict(
        self,
        data: dict[str, Any],
        validate: bool = True,
    ) -> KnowledgeInput:
        """Process a dictionary representation directly.

        Args:
            data: Dict with required keys 'content' and 'input_type'.
            validate: Whether to run validation.

        Returns:
            Parsed KnowledgeInput.
        """
        return self.process(
            content=data["content"],
            input_type=data["input_type"],
            source_id=data.get("source_id"),
            metadata=data.get("metadata"),
            validate=validate,
        )

    def auto_detect_type(self, content: str) -> str | None:
        """Attempt to auto-detect the input type from content.

        Heuristic detection based on content patterns.

        Returns:
            Detected input type or None if ambiguous.
        """
        stripped = content.strip()

        # Check for triple patterns
        if "(" in stripped and "," in stripped and ")" in stripped:
            return "triple"
        if stripped.count("|") >= 2:
            return "triple"

        # Check for schema patterns
        if "->" in stripped or (":" in stripped and any(c in stripped for c in ["Entity", "Class", "Type"])):
            return "schema"

        # Check for entity patterns with brackets or braces
        if "[" in stripped and "]" in stripped:
            return "entity"
        if "{" in stripped and "}" in stripped and ":" in stripped:
            return "entity"

        # Detect text: multiple words without structural markers
        words = stripped.split()
        if len(words) > 5 and len(stripped) > 20:
            return "text"

        # Long content defaults to text
        if len(stripped) > 50:
            return "text"

        # Short content with multiple words could be text or entity
        if len(words) > 2:
            return "text"

        # Very short content could be a simple entity
        return "entity"

    def process_auto(
        self,
        content: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> KnowledgeInput:
        """Process content with automatic type detection.

        Args:
            content: Raw knowledge content.
            source_id: Optional source identifier.
            metadata: Optional metadata dictionary.
            validate: Whether to run validation.

        Returns:
            Parsed KnowledgeInput.

        Raises:
            UnsupportedFormatError: If type cannot be detected.
        """
        detected_type = self.auto_detect_type(content)
        if detected_type is None:
            raise UnsupportedFormatError(
                "auto_detect",
                supported=self.supported_types,
            )
        logger.info(
            "auto_detected_type",
            detected_type=detected_type,
            content_preview=content[:200],
        )
        return self.process(
            content=content,
            input_type=detected_type,
            source_id=source_id,
            metadata=metadata,
            validate=validate,
        )
