"""Prompt strategy implementations.

Provides five concrete prompt strategies used in the KnowProbe experiments:
- Zero-shot: Direct generation without examples.
- Few-shot: In-context learning with selected examples.
- Chain-of-Thought (CoT): Step-by-step reasoning before generating the question.
- Self-Consistency: Multiple reasoning paths with consensus.
- ReAct: Reasoning + Acting loop for structured generation.

Each strategy is a self-contained class that accepts a PromptContext and produces
one or more prompt strings ready for LLM inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from knowprobe.core.models import (
    KnowledgeInput,
    PromptStrategy,
    QuestionType,
)
from knowprobe.prompts.examples import (
    Example,
    ExampleBank,
    ExampleSelector,
    ExampleSelectorFactory,
    RandomExampleSelector,
)
from knowprobe.prompts.templates import (
    PromptTemplate,
    TemplateRegistry,
    load_builtin_templates,
)
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyError(Exception):
    """Raised when a strategy fails to build a prompt."""


@dataclass
class PromptContext:
    """Context required to build a prompt.

    Attributes:
        knowledge_input: The structured knowledge input.
        question_type: The target question type.
        examples: Optional pre-selected few-shot examples.
        generation_params: Extra generation parameters (temperature, max_length, etc.).
        metadata: Additional context for template rendering.
    """

    knowledge_input: KnowledgeInput
    question_type: QuestionType
    examples: list[Example] = field(default_factory=list)
    generation_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all prompt strategies.

    Subclasses must implement:
      - `strategy_type`: class attribute mapping to PromptStrategy enum.
      - `build_prompts`: method that returns a list of prompt strings.
    """

    strategy_type: PromptStrategy

    def __init__(
        self,
        template_registry: TemplateRegistry | None = None,
        example_bank: ExampleBank | None = None,
        example_selector: ExampleSelector | None = None,
        few_shot_k: int = 3,
    ) -> None:
        """Initialize the strategy.

        Args:
            template_registry: Registry for resolving templates. If None,
                built-in fallback templates are used.
            example_bank: Bank of few-shot examples. Required for Few-shot and
                optional for other strategies.
            example_selector: Selector for choosing examples. Defaults to Random.
            few_shot_k: Number of examples to select for few-shot strategies.
        """
        self._registry = template_registry or self._builtin_registry()
        self._example_bank = example_bank or ExampleBank()
        self._selector = example_selector or RandomExampleSelector()
        self._few_shot_k = few_shot_k

    @staticmethod
    def _builtin_registry() -> TemplateRegistry:
        """Create a registry populated with built-in templates (no file I/O)."""
        registry = TemplateRegistry(".")  # dummy path
        for key, tmpl in load_builtin_templates().items():
            registry.register(tmpl)
        return registry

    def _render_template(
        self,
        question_type: QuestionType,
        context: dict[str, Any],
    ) -> str:
        """Render the strategy's template for the given question type."""
        return self._registry.render(self.strategy_type, question_type, context)

    def _select_examples(
        self,
        knowledge_input: KnowledgeInput,
        question_type: QuestionType,
        k: int | None = None,
    ) -> list[Example]:
        """Select few-shot examples if available."""
        if not self._example_bank:
            return []
        k = k if k is not None else self._few_shot_k
        return self._selector.select(
            bank=self._example_bank,
            knowledge_input=knowledge_input.content,
            question_type=question_type,
            strategy=self.strategy_type,
            k=k,
        )

    @abstractmethod
    def build_prompts(self, context: PromptContext) -> list[str]:
        """Build one or more prompt strings from the given context.

        Returns a list because some strategies (e.g., Self-Consistency)
        produce multiple prompts per input.
        """
        ...

    def build(self, context: PromptContext) -> list[str]:
        """Public entry point with logging and error handling."""
        logger.debug(
            "strategy_building_prompts",
            strategy=self.strategy_type.value,
            question_type=context.question_type.value,
            knowledge_id=context.knowledge_input.source_id,
        )
        try:
            prompts = self.build_prompts(context)
        except Exception as exc:
            logger.error(
                "strategy_build_failed",
                strategy=self.strategy_type.value,
                error=str(exc),
            )
            raise StrategyError(
                f"Failed to build prompts for strategy '{self.strategy_type.value}': {exc}"
            ) from exc

        logger.debug(
            "strategy_prompts_built",
            strategy=self.strategy_type.value,
            count=len(prompts),
        )
        return prompts


# ── Concrete Strategy Implementations ─────────────────────────────────────────


class ZeroShotStrategy(BaseStrategy):
    """Zero-shot prompt strategy.

    Directly instructs the model to generate a question from the knowledge
    without providing any examples.
    """

    strategy_type = PromptStrategy.ZERO_SHOT

    def build_prompts(self, context: PromptContext) -> list[str]:
        """Build a single zero-shot prompt."""
        tmpl_context = {
            "knowledge": context.knowledge_input.content,
            "structured": context.knowledge_input.structured,
            "question_type": context.question_type.value,
            "metadata": context.metadata,
        }
        prompt = self._render_template(context.question_type, tmpl_context)
        return [prompt]


class FewShotStrategy(BaseStrategy):
    """Few-shot prompt strategy.

    Prepends a set of in-context examples before the target knowledge to
    guide the model's generation style and format.
    """

    strategy_type = PromptStrategy.FEW_SHOT

    def __init__(
        self,
        template_registry: TemplateRegistry | None = None,
        example_bank: ExampleBank | None = None,
        example_selector: ExampleSelector | None = None,
        few_shot_k: int = 3,
    ) -> None:
        super().__init__(
            template_registry=template_registry,
            example_bank=example_bank,
            example_selector=example_selector,
            few_shot_k=few_shot_k,
        )

    def build_prompts(self, context: PromptContext) -> list[str]:
        """Build a single few-shot prompt with selected examples."""
        # Use pre-selected examples if provided; otherwise select dynamically
        examples = context.examples or self._select_examples(
            context.knowledge_input,
            context.question_type,
        )
        example_dicts = [
            {
                "knowledge": ex.knowledge,
                "question": ex.question,
                "question_type": ex.question_type.value,
            }
            for ex in examples
        ]
        tmpl_context = {
            "knowledge": context.knowledge_input.content,
            "structured": context.knowledge_input.structured,
            "question_type": context.question_type.value,
            "examples": example_dicts,
            "metadata": context.metadata,
        }
        prompt = self._render_template(context.question_type, tmpl_context)
        return [prompt]


class CoTStrategy(BaseStrategy):
    """Chain-of-Thought (CoT) prompt strategy.

    Appends reasoning instructions ("Think step by step...") to elicit
    intermediate reasoning before the final question.
    """

    strategy_type = PromptStrategy.CHAIN_OF_THOUGHT

    def build_prompts(self, context: PromptContext) -> list[str]:
        """Build a single CoT prompt with step-by-step instructions."""
        tmpl_context = {
            "knowledge": context.knowledge_input.content,
            "structured": context.knowledge_input.structured,
            "question_type": context.question_type.value,
            "metadata": context.metadata,
            "cot_instruction": (
                "Let's think step by step. "
                "First, identify the key entities and relations. "
                "Then, determine what question type is most appropriate. "
                "Finally, formulate a clear question."
            ),
        }
        prompt = self._render_template(context.question_type, tmpl_context)
        return [prompt]


class SelfConsistencyStrategy(BaseStrategy):
    """Self-Consistency prompt strategy.

    Generates `n` independent CoT-style prompts. Each prompt contains the
    same knowledge but with a slightly different reasoning instruction or
    ordering to encourage diverse reasoning paths. The answers are later
    aggregated by majority voting or similarity clustering.
    """

    strategy_type = PromptStrategy.SELF_CONSISTENCY

    def __init__(
        self,
        template_registry: TemplateRegistry | None = None,
        example_bank: ExampleBank | None = None,
        example_selector: ExampleSelector | None = None,
        few_shot_k: int = 3,
        num_samples: int = 5,
    ) -> None:
        super().__init__(
            template_registry=template_registry,
            example_bank=example_bank,
            example_selector=example_selector,
            few_shot_k=few_shot_k,
        )
        self._num_samples = num_samples

    def build_prompts(self, context: PromptContext) -> list[str]:
        """Build N CoT prompts with varied reasoning instructions."""
        variations = [
            (
                "Let's think step by step. "
                "Start by listing the entities, then identify relations, "
                "and finally compose the question."
            ),
            (
                "Approach this from the perspective of a student trying to "
                "understand the material. Break down the reasoning."
            ),
            (
                "First, summarize the knowledge in your own words. "
                "Then, identify what makes a good question from this summary."
            ),
            (
                "Consider multiple angles: factual recall, relational inference, "
                "and structural understanding. Pick the strongest angle."
            ),
            (
                "Analyze the subject-predicate-object structure. "
                "Then reframe one component as a question."
            ),
        ]

        prompts: list[str] = []
        for i in range(self._num_samples):
            instruction = variations[i % len(variations)]
            tmpl_context = {
                "knowledge": context.knowledge_input.content,
                "structured": context.knowledge_input.structured,
                "question_type": context.question_type.value,
                "metadata": context.metadata,
                "cot_instruction": instruction,
                "sample_index": i + 1,
            }
            prompt = self._render_template(context.question_type, tmpl_context)
            prompts.append(prompt)
        return prompts


class ReActStrategy(BaseStrategy):
    """ReAct (Reasoning + Acting) prompt strategy.

    Structures the generation as an interleaved loop of:
      Thought -> Action -> Observation
    This encourages the model to explicitly plan before generating the question.
    """

    strategy_type = PromptStrategy.REACT

    def build_prompts(self, context: PromptContext) -> list[str]:
        """Build a single ReAct prompt with reasoning-acting structure."""
        tmpl_context = {
            "knowledge": context.knowledge_input.content,
            "structured": context.knowledge_input.structured,
            "question_type": context.question_type.value,
            "metadata": context.metadata,
            "max_steps": 3,
        }
        prompt = self._render_template(context.question_type, tmpl_context)
        return [prompt]


# ── Strategy Factory ──────────────────────────────────────────────────────────


class StrategyFactory:
    """Factory for creating strategy instances by enum value.

    Usage:
        strategy = StrategyFactory.create(PromptStrategy.COT)
        prompts = strategy.build(context)
    """

    _strategies: dict[PromptStrategy, type[BaseStrategy]] = {
        PromptStrategy.ZERO_SHOT: ZeroShotStrategy,
        PromptStrategy.FEW_SHOT: FewShotStrategy,
        PromptStrategy.CHAIN_OF_THOUGHT: CoTStrategy,
        PromptStrategy.SELF_CONSISTENCY: SelfConsistencyStrategy,
        PromptStrategy.REACT: ReActStrategy,
    }

    @classmethod
    def create(
        cls,
        strategy: PromptStrategy,
        **kwargs: Any,
    ) -> BaseStrategy:
        """Create a strategy instance.

        Args:
            strategy: The desired prompt strategy enum value.
            **kwargs: Passed to the strategy constructor.

        Returns:
            An initialized strategy instance.

        Raises:
            ValueError: If the strategy enum is not registered.
        """
        if strategy not in cls._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy.value}'. "
                f"Available: {[s.value for s in cls._strategies]}"
            )
        strategy_cls = cls._strategies[strategy]
        return strategy_cls(**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return human-readable names of registered strategies."""
        return [s.value for s in cls._strategies]

    @classmethod
    def register(
        cls,
        strategy: PromptStrategy,
        strategy_cls: type[BaseStrategy],
    ) -> None:
        """Register a custom strategy class.

        Args:
            strategy: The enum value to map.
            strategy_cls: A concrete subclass of BaseStrategy.
        """
        if not issubclass(strategy_cls, BaseStrategy):
            raise TypeError(
                f"Strategy class must inherit from BaseStrategy, got {strategy_cls}"
            )
        cls._strategies[strategy] = strategy_cls
        logger.info(
            "strategy_registered",
            strategy=strategy.value,
            class_name=strategy_cls.__name__,
        )
