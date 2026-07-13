"""Template management for prompt strategies.

Provides Jinja2-based template loading, registration, and rendering
with support for strategy-specific and question-type-specific overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import (
    Environment,
    FileSystemLoader,
    TemplateError,
)
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ConfigDict

from knowprobe.core.models import PromptStrategy, QuestionType
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class TemplateRenderError(Exception):
    """Raised when template rendering fails."""


class TemplateLoadError(Exception):
    """Raised when template loading fails."""


class PromptTemplate(BaseModel):
    """A registered prompt template with metadata.

    Attributes:
        name: Unique template identifier (e.g., "cot_factual").
        strategy: The prompt strategy this template belongs to.
        question_type: The question type this template targets.
        source_path: Path to the template file, or None for inline templates.
        content: Raw Jinja2 template content (loaded lazily if source_path is set).
        description: Human-readable description of the template.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    strategy: PromptStrategy
    question_type: QuestionType
    source_path: Path | None = None
    content: str = ""
    description: str = ""

    def _get_template(self, env: Environment) -> Any:
        """Compile the Jinja2 template from content or file."""
        if self.content:
            return env.from_string(self.content)
        if self.source_path and self.source_path.exists():
            return env.get_template(str(self.source_path))
        raise TemplateRenderError(
            f"Template '{self.name}' has no content and no valid source path."
        )

    def render(self, context: dict[str, Any], env: Environment | None = None) -> str:
        """Render the template with the given context.

        Args:
            context: Dictionary of variables to pass into the template.
            env: Optional Jinja2 environment. If None, a default environment is created.

        Returns:
            The rendered prompt string.

        Raises:
            TemplateRenderError: If the template cannot be rendered.
        """
        try:
            jinja_env = env or _default_env()
            template = self._get_template(jinja_env)
            return template.render(**context)
        except TemplateError as exc:
            logger.error(
                "template_render_failed",
                template_name=self.name,
                error=str(exc),
            )
            raise TemplateRenderError(f"Failed to render template '{self.name}': {exc}") from exc


class TemplateRegistry:
    """Registry for prompt templates with discovery and lookup.

    The registry loads templates from a directory tree organized as:
        templates_dir/
            {strategy}/
                {question_type}.j2
            _defaults/
                {strategy}.j2

    It supports fallback resolution: if a strategy+question_type combo is not found,
    it falls back to the strategy default.
    """

    def __init__(self, templates_dir: str | Path) -> None:
        """Initialize the registry with a templates directory.

        Args:
            templates_dir: Path to the root templates directory.
        """
        self._dir = Path(templates_dir)
        self._templates: dict[str, PromptTemplate] = {}
        self._env = self._create_env()
        self._load_templates()

    @staticmethod
    def _create_env() -> SandboxedEnvironment:
        """Create a sandboxed Jinja2 environment with safe defaults."""
        return SandboxedEnvironment(
            loader=FileSystemLoader("."),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _load_templates(self) -> None:
        """Discover and load all template files from the directory."""
        if not self._dir.exists():
            logger.warning(
                "templates_dir_not_found",
                path=str(self._dir),
            )
            return

        # Load strategy-specific / question-type-specific templates
        for strategy_dir in self._dir.iterdir():
            if not strategy_dir.is_dir() or strategy_dir.name.startswith("_"):
                continue
            try:
                strategy = PromptStrategy(strategy_dir.name)
            except ValueError:
                logger.warning(
                    "unknown_strategy_directory",
                    directory=strategy_dir.name,
                )
                continue

            for template_file in strategy_dir.glob("*.j2"):
                try:
                    qtype = QuestionType(template_file.stem)
                except ValueError:
                    logger.warning(
                        "unknown_question_type_file",
                        file=str(template_file),
                    )
                    continue
                self._register_file(strategy, qtype, template_file)

        # Load default templates from _defaults/
        defaults_dir = self._dir / "_defaults"
        if defaults_dir.exists():
            for template_file in defaults_dir.glob("*.j2"):
                try:
                    strategy = PromptStrategy(template_file.stem)
                except ValueError:
                    continue
                self._register_file(strategy, QuestionType.FACTUAL, template_file, is_default=True)

    def _register_file(
        self,
        strategy: PromptStrategy,
        question_type: QuestionType,
        path: Path,
        is_default: bool = False,
    ) -> None:
        """Register a single template file."""
        name = f"{strategy.value}_{question_type.value}"
        if is_default:
            name = f"{strategy.value}_default"

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error(
                "template_read_failed",
                path=str(path),
                error=str(exc),
            )
            raise TemplateLoadError(f"Cannot read template file {path}: {exc}") from exc

        self._templates[name] = PromptTemplate(
            name=name,
            strategy=strategy,
            question_type=question_type,
            source_path=path,
            content=content,
            description=f"{'Default ' if is_default else ''}{strategy.value} template for {question_type.value} questions",
        )
        logger.debug(
            "template_registered",
            name=name,
            path=str(path),
        )

    def register(self, template: PromptTemplate) -> None:
        """Manually register a prompt template."""
        key = f"{template.strategy.value}_{template.question_type.value}"
        self._templates[key] = template
        logger.info("template_registered_manually", name=template.name, key=key)

    def get(
        self,
        strategy: PromptStrategy,
        question_type: QuestionType,
    ) -> PromptTemplate:
        """Retrieve a template with fallback resolution.

        Resolution order:
        1. {strategy}_{question_type}
        2. {strategy}_default
        3. Raise TemplateRenderError
        """
        primary_key = f"{strategy.value}_{question_type.value}"
        fallback_key = f"{strategy.value}_default"

        if primary_key in self._templates:
            return self._templates[primary_key]
        if fallback_key in self._templates:
            logger.debug(
                "template_fallback_used",
                requested=primary_key,
                fallback=fallback_key,
            )
            return self._templates[fallback_key]

        raise TemplateRenderError(
            f"No template found for strategy='{strategy.value}' question_type='{question_type.value}'."
        )

    def list_templates(self) -> list[PromptTemplate]:
        """Return all registered templates."""
        return list(self._templates.values())

    def render(
        self,
        strategy: PromptStrategy,
        question_type: QuestionType,
        context: dict[str, Any],
    ) -> str:
        """Convenience method: get template and render in one call."""
        template = self.get(strategy, question_type)
        return template.render(context, env=self._env)

    def __contains__(self, key: str) -> bool:
        return key in self._templates

    def __len__(self) -> int:
        return len(self._templates)


def _default_env() -> Environment:
    """Create a default Jinja2 environment for inline templates."""
    return Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


# ── Built-in fallback templates (used when no template files exist) ──────────

_ZERO_SHOT_TEMPLATE = """You are a knowledgeable assistant. Given the knowledge below, generate a concise and clear question.

Knowledge:
{{ knowledge }}

Requirements:
- The question must be answerable using ONLY the provided knowledge.
- Do not add information not present in the knowledge.
{% if question_type == "factual" %}
- Ask a factual question about a specific entity, relation, or attribute.
{% elif question_type == "schema" %}
- Ask about the schema, structure, or relationships within the knowledge.
{% elif question_type == "composite" %}
- Ask a multi-hop question that requires combining multiple facts.
{% endif %}

Question:"""

_FEW_SHOT_TEMPLATE = """You are a knowledgeable assistant. Given the knowledge below, generate a concise and clear question.

{% if examples %}
Here are some examples:
{% for ex in examples %}
Knowledge: {{ ex.knowledge }}
Question: {{ ex.question }}
{% endfor %}
{% endif %}

Knowledge:
{{ knowledge }}

Generate a similar question:

Question:"""

_COT_TEMPLATE = """You are a knowledgeable assistant. Given the knowledge below, generate a concise and clear question.

Think step by step:
1. Identify the key information in the knowledge.
2. Determine what kind of question can be asked.
3. Formulate the question.

Knowledge:
{{ knowledge }}

Step-by-step reasoning and final question:

1."""

_SELF_CONSISTENCY_TEMPLATE = """You are a knowledgeable assistant. Given the knowledge below, generate a concise and clear question.

Think step by step and reason through multiple paths. After exploring different angles, provide the most consistent question.

Knowledge:
{{ knowledge }}

Reasoning paths and final question:

Path 1:"""

_REACT_TEMPLATE = """You are a knowledgeable assistant. Given the knowledge below, generate a concise and clear question.

Use the following reasoning approach:
Thought: Consider what question would best test understanding of this knowledge.
Action: Formulate the question based on your thought.
Observation: Review the question to ensure it is clear and answerable from the knowledge.

Repeat this process until you are confident in the question.

Knowledge:
{{ knowledge }}

Thought:"""


_BUILTIN_TEMPLATES: dict[str, str] = {
    "zero_shot": _ZERO_SHOT_TEMPLATE,
    "few_shot": _FEW_SHOT_TEMPLATE,
    "cot": _COT_TEMPLATE,
    "self_consistency": _SELF_CONSISTENCY_TEMPLATE,
    "react": _REACT_TEMPLATE,
}


def load_builtin_templates() -> dict[str, PromptTemplate]:
    """Load the built-in fallback templates as PromptTemplate objects."""
    templates: dict[str, PromptTemplate] = {}
    for strategy_value, content in _BUILTIN_TEMPLATES.items():
        strategy = PromptStrategy(strategy_value)
        for qtype in QuestionType:
            key = f"{strategy_value}_{qtype.value}"
            templates[key] = PromptTemplate(
                name=key,
                strategy=strategy,
                question_type=qtype,
                content=content,
                description=f"Built-in fallback template for {strategy_value} / {qtype.value}",
            )
    return templates
