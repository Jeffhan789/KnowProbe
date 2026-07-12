"""Prompt building system for different question types and strategies.

Supports Jinja2 template loading from disk with built-in fallback templates.
All templates are parameterised by ``knowledge``, ``strategy``, and
``question_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import (
    Environment,
    FileSystemLoader,
    Template,
    TemplateNotFound,
    select_autoescape,
)

from knowprobe.core.models import PromptStrategy, QuestionType
from knowprobe.generators.base import PromptBuildError
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Built-in fallback templates (used when no external template file is found)
# --------------------------------------------------------------------------- #

_BUILTIN_ZERO_SHOT_FACTUAL = """你是一个知识问答生成专家。根据以下知识内容生成一个**事实性问题**。

要求：
- 问题必须基于知识内容，确保可回答
- 只生成一个问题，以问号结尾
- 不要添加任何解释或前缀

知识内容：
{{ knowledge.content }}

问题："""

_BUILTIN_ZERO_SHOT_SCHEMA = """你是一个知识图谱Schema分析专家。根据以下Schema描述生成一个**Schema层面的问题**。

要求：
- 问题关注Schema结构、类型定义、属性约束或关系模式
- 只生成一个问题，以问号结尾
- 不要添加任何解释或前缀

Schema内容：
{{ knowledge.content }}

Schema问题："""

_BUILTIN_COT_FACTUAL = """你是一个知识问答生成专家。请**逐步思考**后生成一个事实性问题。

思考步骤：
1. 识别知识中的核心实体和关键属性
2. 确定哪些信息可以作为答案
3. 设计一个清晰、无歧义的问题

知识内容：
{{ knowledge.content }}

思考过程：
（请展示推理）

最终问题："""

_BUILTIN_COT_SCHEMA = """你是一个知识图谱Schema分析专家。请**逐步思考**后生成一个Schema问题。

思考步骤：
1. 分析Schema包含哪些类型/类
2. 识别类型之间的关系和继承层次
3. 分析属性定义（domain、range、约束）
4. 设计测试Schema理解的问题

Schema内容：
{{ knowledge.content }}

思考过程：
（请展示推理）

最终Schema问题："""

_BUILTIN_FEW_SHOT_FACTUAL = """你是一个知识问答生成专家。请参考以下示例，生成一个事实性问题。

{% for ex in examples %}
示例{{ loop.index }}：
知识：{{ ex.knowledge.content }}
问题：{{ ex.question }}
{% endfor %}

现在请为以下知识生成问题：
知识内容：{{ knowledge.content }}

问题："""

_BUILTIN_FEW_SHOT_SCHEMA = """你是一个知识图谱Schema分析专家。请参考以下示例，生成一个Schema问题。

{% for ex in examples %}
示例{{ loop.index }}：
Schema：{{ ex.knowledge.content }}
问题：{{ ex.question }}
{% endfor %}

现在请为以下Schema生成问题：
Schema内容：{{ knowledge.content }}

Schema问题："""

_BUILTIN_SELF_CONSISTENCY_FACTUAL = """你是一个知识问答生成专家。请从多个角度生成一个事实性问题，然后选择最佳答案。

知识内容：
{{ knowledge.content }}

请生成{{ self_consistency_n | default(5) }}个不同角度的问题候选，然后选择最佳的一个：

最终问题："""

_BUILTIN_SELF_CONSISTENCY_SCHEMA = """你是一个知识图谱Schema分析专家。请从多个角度生成一个Schema问题，然后选择最佳答案。

Schema内容：
{{ knowledge.content }}

请生成{{ self_consistency_n | default(5) }}个不同角度的问题候选，然后选择最佳的一个：

最终Schema问题："""

_BUILTIN_TEMPLATES: dict[str, str] = {
    "zero_shot_factual": _BUILTIN_ZERO_SHOT_FACTUAL,
    "zero_shot_schema": _BUILTIN_ZERO_SHOT_SCHEMA,
    "few_shot_factual": _BUILTIN_FEW_SHOT_FACTUAL,
    "few_shot_schema": _BUILTIN_FEW_SHOT_SCHEMA,
    "cot_factual": _BUILTIN_COT_FACTUAL,
    "cot_schema": _BUILTIN_COT_SCHEMA,
    "self_consistency_factual": _BUILTIN_SELF_CONSISTENCY_FACTUAL,
    "self_consistency_schema": _BUILTIN_SELF_CONSISTENCY_SCHEMA,
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptTemplate:
    """A loaded prompt template with metadata.

    Attributes:
        name: Template identifier (e.g. "cot_factual").
        template: Compiled Jinja2 ``Template`` object.
        strategy: The prompt strategy this template implements.
        question_type: The question type this template targets.
        description: Optional human-readable description.
    """

    name: str
    template: Template
    strategy: PromptStrategy
    question_type: QuestionType
    description: str = ""


# --------------------------------------------------------------------------- #
# PromptBuilder
# --------------------------------------------------------------------------- #


class PromptBuilder:
    """Builds prompts for question generation using Jinja2.

    The builder maintains an in-memory cache of compiled templates. Templates are
    looked up in the following order:

    1. In-memory cache
    2. Filesystem templates directory (if configured)
    3. Built-in fallback templates

    Usage::

        builder = PromptBuilder("configs/prompts")
        prompt = builder.build(
            knowledge={"content": "巴黎是法国的首都"},
            strategy=PromptStrategy.ZERO_SHOT,
            question_type=QuestionType.FACTUAL,
        )
    """

    def __init__(self, templates_dir: str | Path | None = None) -> None:
        """Initialise the builder.

        Args:
            templates_dir: Directory containing ``*.jinja2`` template files.
                If ``None``, only built-in templates are available.
        """
        self._logger = get_logger(__name__)
        self._cache: dict[str, PromptTemplate] = {}
        self._jinja_env: Environment | None = None

        if templates_dir is not None:
            self._init_filesystem_loader(templates_dir)
        else:
            self._logger.debug("No templates_dir provided; using built-in templates only")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(
        self,
        knowledge: dict[str, Any],
        strategy: PromptStrategy,
        question_type: QuestionType,
        *,
        examples: list[dict[str, Any]] | None = None,
        reasoning_steps: list[str] | None = None,
        self_consistency_n: int = 5,
        **extra: Any,
    ) -> str:
        """Build a prompt string for the given configuration.

        Args:
            knowledge: Structured knowledge input data (must contain at least
                ``content`` key). May also include ``input_type``,
                ``structured``, and ``metadata``.
            strategy: Prompting strategy to apply.
            question_type: Type of question to generate.
            examples: Few-shot examples (required when strategy is ``FEW_SHOT``).
            reasoning_steps: Optional pre-defined reasoning steps for CoT.
            self_consistency_n: Number of samples for self-consistency strategy.
            **extra: Additional template variables forwarded to Jinja2.

        Returns:
            Rendered prompt string ready for model inference.

        Raises:
            PromptBuildError: If template rendering fails.
        """
        template = self._get_template(strategy, question_type)

        ctx: dict[str, Any] = {
            "knowledge": knowledge,
            "examples": examples or [],
            "reasoning_steps": reasoning_steps or [],
            "self_consistency_n": self_consistency_n,
            **extra,
        }

        try:
            rendered = template.template.render(ctx)
        except Exception as exc:
            self._logger.error(
                "Template rendering failed",
                template=template.name,
                error=str(exc),
            )
            raise PromptBuildError(
                f"Failed to render template {template.name}: {exc}",
                template_key=template.name,
            ) from exc

        return rendered.strip()

    def list_available_templates(self) -> list[str]:
        """Return a list of all available template names."""
        return list(self._cache.keys()) + list(_BUILTIN_TEMPLATES.keys())

    def warm_cache(self) -> None:
        """Pre-load all built-in templates into the cache."""
        for key in _BUILTIN_TEMPLATES:
            self._load_builtin(key)
        self._logger.info("Prompt cache warmed", count=len(self._cache))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _init_filesystem_loader(self, templates_dir: str | Path) -> None:
        """Set up Jinja2 FileSystemLoader for external templates."""
        path = Path(templates_dir).resolve()
        if not path.exists() or not path.is_dir():
            self._logger.warning(
                "Templates directory not found; using built-in templates",
                path=str(path),
            )
            return

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(path)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._logger.info("Filesystem template loader initialised", path=str(path))

    def _get_template(
        self,
        strategy: PromptStrategy,
        question_type: QuestionType,
    ) -> PromptTemplate:
        """Look up a template by strategy + question type."""
        key = f"{strategy.value}_{question_type.value}"

        if key in self._cache:
            return self._cache[key]

        # Try filesystem first
        if self._jinja_env is not None:
            try:
                jinja_tpl = self._jinja_env.get_template(f"{key}.jinja2")
                pt = PromptTemplate(
                    name=key,
                    template=jinja_tpl,
                    strategy=strategy,
                    question_type=question_type,
                )
                self._cache[key] = pt
                self._logger.debug("Loaded template from filesystem", key=key)
                return pt
            except TemplateNotFound:
                self._logger.debug(
                    "Filesystem template not found; falling back to built-in",
                    key=key,
                )

        # Fall back to built-in
        return self._load_builtin(key)

    def _load_builtin(self, key: str) -> PromptTemplate:
        """Load a built-in template by key."""
        if key not in _BUILTIN_TEMPLATES:
            # Graceful fallback: try to use the zero_shot variant of the same type
            fallback_key = key.replace(key.split("_", 1)[0], "zero_shot")
            if fallback_key in _BUILTIN_TEMPLATES:
                self._logger.warning(
                    "Template not found; using zero_shot fallback",
                    requested=key,
                    fallback=fallback_key,
                )
                key = fallback_key
            else:
                raise PromptBuildError(
                    f"No template available for key '{key}'",
                    template_key=key,
                )

        tpl = Environment().from_string(_BUILTIN_TEMPLATES[key])
        strategy_str, question_type_str = key.rsplit("_", 1)
        pt = PromptTemplate(
            name=key,
            template=tpl,
            strategy=PromptStrategy(strategy_str),
            question_type=QuestionType(question_type_str),
            description="Built-in fallback template",
        )
        self._cache[key] = pt
        return pt
