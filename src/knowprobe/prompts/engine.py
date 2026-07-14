"""Prompt Strategy Engine — unified facade for the prompt engineering subsystem.

The engine is the primary interface for the rest of the KnowProbe application.
It wires together templates, examples, strategies, and the builder, and exposes
a clean API for generating prompts for experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from knowprobe.core.config import Settings, get_settings
from knowprobe.core.models import (
    KnowledgeInput,
    PromptStrategy,
    QuestionType,
)
from knowprobe.prompts.builder import PromptBuilder
from knowprobe.prompts.examples import (
    Example,
    ExampleBank,
    ExampleSelectorFactory,
)
from knowprobe.prompts.strategies import StrategyFactory
from knowprobe.prompts.templates import TemplateRegistry, load_builtin_templates
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class PromptEngineError(Exception):
    """Raised when the prompt engine encounters a fatal error."""


class PromptStrategyEngine:
    """Unified facade for prompt strategy execution.

    This class orchestrates:
      - Template loading and resolution
      - Few-shot example bank management
      - Strategy instantiation and dispatch
      - Prompt building and batch processing

    It is designed to be instantiated once per application lifecycle and reused
    across multiple generation calls.

    Example:
        engine = PromptStrategyEngine.from_settings()
        prompts = engine.build(
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
    """

    def __init__(
        self,
        template_registry: TemplateRegistry,
        example_bank: ExampleBank,
        prompt_builder: PromptBuilder,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the engine with its subcomponents.

        Args:
            template_registry: Resolved template registry.
            example_bank: Loaded example bank.
            prompt_builder: Configured prompt builder.
            settings: Optional settings reference for logging/debugging.
        """
        self._registry = template_registry
        self._example_bank = example_bank
        self._builder = prompt_builder
        self._settings = settings
        self._available_strategies = StrategyFactory.list_strategies()

        logger.info(
            "prompt_engine_initialized",
            templates=len(self._registry),
            examples=len(self._example_bank),
            strategies=self._available_strategies,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        templates_dir: str | Path | None = None,
        examples_path: str | Path | None = None,
    ) -> PromptStrategyEngine:
        """Create an engine instance from application settings.

        This is the recommended factory method. It:
        1. Loads settings from config files / env vars.
        2. Loads templates from the configured templates directory.
        3. Loads few-shot examples from the configured examples file.
        4. Configures the example selector and prompt builder.

        Args:
            settings: Optional Settings object. If None, loads from default paths.
            templates_dir: Override the templates directory path.
            examples_path: Override the few-shot examples YAML path.

        Returns:
            A fully initialized PromptStrategyEngine.
        """
        cfg = settings or get_settings()

        # ── Template Registry ──
        tmpl_dir = Path(templates_dir or cfg.prompts.templates_dir)
        registry = TemplateRegistry(tmpl_dir)

        # If no templates were found on disk, fall back to built-ins
        if len(registry) == 0:
            logger.warning(
                "no_templates_found_on_disk",
                path=str(tmpl_dir),
                fallback="built_in_templates",
            )
            for _key, tmpl in load_builtin_templates().items():
                registry.register(tmpl)

        # ── Example Bank ──
        bank = ExampleBank()
        ex_path = examples_path or (tmpl_dir / "examples.yaml")
        if isinstance(ex_path, str):
            ex_path = Path(ex_path)
        if ex_path.exists():
            try:
                bank.load_from_yaml(ex_path)
            except Exception as exc:
                logger.warning(
                    "example_load_failed",
                    path=str(ex_path),
                    error=str(exc),
                )
        else:
            logger.info(
                "no_examples_file_found",
                path=str(ex_path),
            )

        # ── Example Selector ──
        selector_type = getattr(cfg.prompts, "example_selector", "random")
        selector = ExampleSelectorFactory.create(selector_type)

        # ── Prompt Builder ──
        few_shot_k = cfg.prompts.few_shot_examples
        builder = PromptBuilder(
            template_registry=registry,
            example_bank=bank,
            example_selector=selector,
            default_few_shot_k=few_shot_k,
        )

        return cls(
            template_registry=registry,
            example_bank=bank,
            prompt_builder=builder,
            settings=cfg,
        )

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
        """Build prompt(s) for a single knowledge input.

        Args:
            strategy: Prompt strategy to apply.
            knowledge_input: Knowledge to generate a question from.
            question_type: Target question type.
            examples: Optional pre-selected few-shot examples.
            generation_params: Extra generation parameters.
            metadata: Additional metadata for templates.
            **strategy_kwargs: Strategy-specific overrides (e.g., num_samples for Self-Consistency).

        Returns:
            List of prompt strings. Length is 1 for most strategies, N for Self-Consistency.
        """
        logger.info(
            "engine_build_prompt",
            strategy=strategy.value,
            question_type=question_type.value,
            knowledge_id=knowledge_input.source_id,
        )
        return self._builder.build(
            strategy=strategy,
            knowledge_input=knowledge_input,
            question_type=question_type,
            examples=examples,
            generation_params=generation_params,
            metadata=metadata,
            **strategy_kwargs,
        )

    def build_batch(
        self,
        strategy: PromptStrategy,
        knowledge_inputs: list[KnowledgeInput],
        question_type: QuestionType,
        **kwargs: Any,
    ) -> list[list[str]]:
        """Build prompts for a batch of knowledge inputs.

        Args:
            strategy: Prompt strategy to apply.
            knowledge_inputs: List of knowledge inputs.
            question_type: Target question type.
            **kwargs: Passed to `build()`.

        Returns:
            List of prompt lists, one per knowledge input.
        """
        logger.info(
            "engine_build_batch",
            strategy=strategy.value,
            question_type=question_type.value,
            batch_size=len(knowledge_inputs),
        )
        return self._builder.build_batch(
            strategy=strategy,
            knowledge_inputs=knowledge_inputs,
            question_type=question_type,
            **kwargs,
        )

    def build_experiment_matrix(
        self,
        knowledge_inputs: list[KnowledgeInput],
        strategies: list[PromptStrategy],
        question_types: list[QuestionType],
        **kwargs: Any,
    ) -> dict[str, dict[str, list[list[str]]]]:
        """Build the full prompt matrix for an experiment.

        This generates prompts for every combination of strategy × question_type
        across all knowledge inputs. The return structure is:

            {
                "zero_shot": {
                    "factual": [[prompt1], [prompt2], ...],
                    "schema": [[prompt1], [prompt2], ...],
                },
                "few_shot": { ... },
                ...
            }

        Args:
            knowledge_inputs: All knowledge inputs for the experiment.
            strategies: All prompt strategies to evaluate.
            question_types: All question types to evaluate.
            **kwargs: Passed to `build()`.

        Returns:
            Nested dict: strategy -> question_type -> list of prompt lists.
        """
        matrix: dict[str, dict[str, list[list[str]]]] = {}
        for strategy in strategies:
            matrix[strategy.value] = {}
            for qtype in question_types:
                try:
                    batch = self.build_batch(
                        strategy=strategy,
                        knowledge_inputs=knowledge_inputs,
                        question_type=qtype,
                        **kwargs,
                    )
                    matrix[strategy.value][qtype.value] = batch
                except Exception as exc:
                    logger.error(
                        "matrix_build_failed",
                        strategy=strategy.value,
                        question_type=qtype.value,
                        error=str(exc),
                    )
                    matrix[strategy.value][qtype.value] = []
        logger.info(
            "experiment_matrix_built",
            strategies=len(strategies),
            question_types=len(question_types),
            knowledge_inputs=len(knowledge_inputs),
        )
        return matrix

    def add_example(self, example: Example) -> None:
        """Add a few-shot example to the engine's example bank."""
        self._example_bank.add(example)
        logger.info("example_added_to_engine", question_type=example.question_type.value)

    def reload_templates(self, templates_dir: str | Path | None = None) -> None:
        """Reload templates from disk. Useful for hot-reloading during development."""
        path = Path(
            templates_dir or (self._settings.prompts.templates_dir if self._settings else ".")
        )
        self._registry = TemplateRegistry(path)
        self._builder = PromptBuilder(
            template_registry=self._registry,
            example_bank=self._example_bank,
            example_selector=self._builder.example_selector,
            default_few_shot_k=self._builder.default_few_shot_k,
        )
        logger.info("templates_reloaded", path=str(path), count=len(self._registry))

    def list_strategies(self) -> list[str]:
        """Return the names of available strategies."""
        return self._available_strategies.copy()

    def list_templates(self) -> list[str]:
        """Return the names of registered templates."""
        return [t.name for t in self._registry.list_templates()]

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"templates={len(self._registry)} "
            f"examples={len(self._example_bank)} "
            f"strategies={self._available_strategies}>"
        )
