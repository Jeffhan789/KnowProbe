"""Prompt builder — high-level interface for assembling prompts.

The PromptBuilder coordinates template resolution, example selection,
and strategy dispatch. It is the primary internal tool used by the
PromptStrategyEngine.
"""

from __future__ import annotations

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
)
from knowprobe.prompts.strategies import (
    BaseStrategy,
    PromptContext,
    StrategyFactory,
)
from knowprobe.prompts.templates import TemplateRegistry
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class PromptBuilderError(Exception):
    """Raised when prompt building fails."""


class PromptBuilder:
    """Builds prompts for a given strategy, question type, and knowledge input.

    The builder maintains references to the template registry, example bank,
    and example selector, and uses them to construct PromptContext objects
    that are passed to the appropriate strategy.
    """

    def __init__(
        self,
        template_registry: TemplateRegistry | None = None,
        example_bank: ExampleBank | None = None,
        example_selector: ExampleSelector | None = None,
        default_few_shot_k: int = 3,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            template_registry: Registry for template resolution.
            example_bank: Bank of few-shot examples.
            example_selector: Strategy for selecting examples.
            default_few_shot_k: Default number of examples for few-shot strategies.
        """
        self._template_registry = template_registry
        self._example_bank = example_bank or ExampleBank()
        self._example_selector = example_selector or ExampleSelectorFactory.create("random")
        self._default_few_shot_k = default_few_shot_k

    @property
    def example_selector(self) -> ExampleSelector:
        """Return the current example selector."""
        return self._example_selector

    @property
    def default_few_shot_k(self) -> int:
        """Return the default number of few-shot examples."""
        return self._default_few_shot_k
        self._example_bank = example_bank or ExampleBank()
        self._example_selector = example_selector or ExampleSelectorFactory.create("random")
        self._default_few_shot_k = default_few_shot_k

    def build(
        self,
        strategy: PromptStrategy,
        knowledge_input: KnowledgeInput,
        question_type: QuestionType,
        examples: list[Example] | None = None,
        generation_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **strategy_kwargs: Any,
    ) -> list[str]:
        """Build prompt strings for the given configuration.

        Args:
            strategy: The prompt strategy to use.
            knowledge_input: The knowledge input to generate a question from.
            question_type: The target question type.
            examples: Optional pre-selected examples (overrides selector).
            generation_params: Extra generation parameters.
            metadata: Additional metadata for template context.
            **strategy_kwargs: Additional constructor args for the strategy.

        Returns:
            A list of prompt strings (one for most strategies, N for Self-Consistency).

        Raises:
            PromptBuilderError: If building fails.
        """
        try:
            strategy_instance = StrategyFactory.create(
                strategy,
                template_registry=self._template_registry,
                example_bank=self._example_bank,
                example_selector=self._example_selector,
                few_shot_k=self._default_few_shot_k,
                **strategy_kwargs,
            )
        except Exception as exc:
            logger.error(
                "strategy_instantiation_failed",
                strategy=strategy.value,
                error=str(exc),
            )
            raise PromptBuilderError(
                f"Failed to instantiate strategy '{strategy.value}': {exc}"
            ) from exc

        context = PromptContext(
            knowledge_input=knowledge_input,
            question_type=question_type,
            examples=examples or [],
            generation_params=generation_params or {},
            metadata=metadata or {},
        )

        return strategy_instance.build(context)

    def build_batch(
        self,
        strategy: PromptStrategy,
        knowledge_inputs: list[KnowledgeInput],
        question_type: QuestionType,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Build prompts for a batch of knowledge inputs.

        Args:
            strategy: The prompt strategy to use.
            knowledge_inputs: List of knowledge inputs.
            question_type: The target question type.
            **kwargs: Passed to `build()`.

        Returns:
            A list where each element is a list of prompts for the corresponding
            knowledge input.
        """
        results: list[list[str]] = []
        for ki in knowledge_inputs:
            try:
                prompts = self.build(strategy, ki, question_type, **kwargs)
                results.append(prompts)
            except Exception as exc:
                logger.error(
                    "batch_build_item_failed",
                    knowledge_id=ki.source_id,
                    error=str(exc),
                )
                results.append([])
        return results
