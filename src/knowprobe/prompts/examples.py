"""Few-shot example management for prompt strategies.

Provides example storage, selection strategies (random, similarity-based, diversity-based),
and integration with the prompt builder.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from knowprobe.core.models import PromptStrategy, QuestionType
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class ExampleLoadError(Exception):
    """Raised when example loading fails."""


class ExampleNotFoundError(Exception):
    """Raised when no suitable examples are found."""


class Example(BaseModel):
    """A single few-shot example.

    Attributes:
        knowledge: The raw knowledge content (e.g., triple, schema, text).
        question: The generated question based on the knowledge.
        question_type: The type of question.
        strategy: The prompt strategy used to generate this example (optional).
        metadata: Additional metadata such as source, difficulty, domain.
    """

    knowledge: str = Field(description="Knowledge content that the question is based on")
    question: str = Field(description="The generated question")
    question_type: QuestionType = Field(default=QuestionType.FACTUAL)
    strategy: PromptStrategy | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_prompt_block(self) -> str:
        """Format this example as a prompt text block."""
        return (
            f"Knowledge: {self.knowledge}\n"
            f"Question: {self.question}"
        )


@dataclass
class ExampleBank:
    """In-memory storage and persistence for few-shot examples.

    Examples can be loaded from YAML files and filtered by question type and strategy.
    """

    examples: list[Example] = field(default_factory=list)

    def add(self, example: Example) -> None:
        """Add an example to the bank."""
        self.examples.append(example)
        logger.debug("example_added", question_type=example.question_type.value)

    def extend(self, examples: list[Example]) -> None:
        """Add multiple examples at once."""
        self.examples.extend(examples)
        logger.debug("examples_added", count=len(examples))

    def filter(
        self,
        question_type: QuestionType | None = None,
        strategy: PromptStrategy | None = None,
        limit: int | None = None,
    ) -> list[Example]:
        """Filter examples by question type and/or strategy.

        Args:
            question_type: Optional filter by question type.
            strategy: Optional filter by prompt strategy.
            limit: Maximum number of examples to return.

        Returns:
            A list of matching examples.
        """
        results = self.examples
        if question_type is not None:
            results = [ex for ex in results if ex.question_type == question_type]
        if strategy is not None:
            results = [ex for ex in results if ex.strategy == strategy]
        if limit is not None:
            results = results[:limit]
        return results

    def load_from_yaml(self, path: str | Path) -> None:
        """Load examples from a YAML file.

        Expected YAML format:
            examples:
              - knowledge: "Albert Einstein was a physicist."
                question: "What was Albert Einstein's profession?"
                question_type: "factual"
              - ...
        """
        p = Path(path)
        if not p.exists():
            raise ExampleLoadError(f"Example file not found: {p}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ExampleLoadError(f"Invalid YAML in {p}: {exc}") from exc

        raw_examples = data.get("examples", [])
        if not isinstance(raw_examples, list):
            raise ExampleLoadError(
                f"Expected 'examples' to be a list in {p}, got {type(raw_examples)}"
            )

        loaded: list[Example] = []
        for idx, raw in enumerate(raw_examples):
            try:
                ex = Example(**raw)
                loaded.append(ex)
            except Exception as exc:
                logger.warning(
                    "example_parse_failed",
                    file=str(p),
                    index=idx,
                    error=str(exc),
                )

        self.extend(loaded)
        logger.info("examples_loaded_from_yaml", path=str(p), count=len(loaded))

    def save_to_yaml(self, path: str | Path) -> None:
        """Save all examples to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"examples": [ex.model_dump(mode="json") for ex in self.examples]}
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        logger.info("examples_saved_to_yaml", path=str(p), count=len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __bool__(self) -> bool:
        return bool(self.examples)


class ExampleSelector(ABC):
    """Abstract base class for few-shot example selection strategies.

    Subclasses must implement the `select` method to choose a subset of examples
    from a given ExampleBank based on a specific selection criterion.
    """

    @abstractmethod
    def select(
        self,
        bank: ExampleBank,
        knowledge_input: str,
        question_type: QuestionType,
        strategy: PromptStrategy,
        k: int,
        **kwargs: Any,
    ) -> list[Example]:
        """Select up to k examples from the bank.

        Args:
            bank: The example bank to select from.
            knowledge_input: The current knowledge input (for similarity-based selection).
            question_type: The target question type to filter by.
            strategy: The target prompt strategy to filter by.
            k: Maximum number of examples to select.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            A list of selected examples.
        """
        ...

    def _filter_candidates(
        self,
        bank: ExampleBank,
        question_type: QuestionType,
        strategy: PromptStrategy,
    ) -> list[Example]:
        """Filter the bank by question type and strategy."""
        return bank.filter(
            question_type=question_type,
            strategy=strategy,
        )


class RandomExampleSelector(ExampleSelector):
    """Selects examples uniformly at random.

    Supports optional seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(
        self,
        bank: ExampleBank,
        knowledge_input: str,
        question_type: QuestionType,
        strategy: PromptStrategy,
        k: int,
        **kwargs: Any,
    ) -> list[Example]:
        candidates = self._filter_candidates(bank, question_type, strategy)
        if not candidates:
            candidates = bank.filter(question_type=question_type)
        if not candidates:
            logger.warning(
                "no_examples_for_random_selection",
                question_type=question_type.value,
                strategy=strategy.value,
            )
            return []
        k = min(k, len(candidates))
        return self._rng.sample(candidates, k)


class SimilarityExampleSelector(ExampleSelector):
    """Selects examples based on simple lexical similarity to the knowledge input.

    Uses Jaccard similarity on word tokens as a lightweight proxy for semantic similarity.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple word tokenization."""
        return set(text.lower().split())

    def _jaccard(self, a: str, b: str) -> float:
        """Compute Jaccard similarity between two strings."""
        tokens_a = self._tokenize(a)
        tokens_b = self._tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

    def select(
        self,
        bank: ExampleBank,
        knowledge_input: str,
        question_type: QuestionType,
        strategy: PromptStrategy,
        k: int,
        **kwargs: Any,
    ) -> list[Example]:
        candidates = self._filter_candidates(bank, question_type, strategy)
        if not candidates:
            candidates = bank.filter(question_type=question_type)
        if not candidates:
            logger.warning(
                "no_examples_for_similarity_selection",
                question_type=question_type.value,
                strategy=strategy.value,
            )
            return []

        scored = [
            (ex, self._jaccard(knowledge_input, ex.knowledge))
            for ex in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        k = min(k, len(scored))
        return [ex for ex, _ in scored[:k]]


class DiversityExampleSelector(ExampleSelector):
    """Selects a diverse set of examples using Maximal Marginal Relevance (MMR)-style selection.

    Balances similarity to the knowledge input with dissimilarity among selected examples.
    """

    def __init__(self, lambda_param: float = 0.5) -> None:
        """Initialize the diversity selector.

        Args:
            lambda_param: Trade-off between relevance (to knowledge_input) and diversity.
                          1.0 = pure relevance; 0.0 = pure diversity.
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be in [0.0, 1.0]")
        self._lambda = lambda_param

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(text.lower().split())

    def _similarity(self, a: str, b: str) -> float:
        tokens_a = self._tokenize(a)
        tokens_b = self._tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union if union > 0 else 0.0

    def select(
        self,
        bank: ExampleBank,
        knowledge_input: str,
        question_type: QuestionType,
        strategy: PromptStrategy,
        k: int,
        **kwargs: Any,
    ) -> list[Example]:
        candidates = self._filter_candidates(bank, question_type, strategy)
        if not candidates:
            candidates = bank.filter(question_type=question_type)
        if not candidates:
            logger.warning(
                "no_examples_for_diversity_selection",
                question_type=question_type.value,
                strategy=strategy.value,
            )
            return []

        selected: list[Example] = []
        remaining = candidates.copy()

        # First selection: highest similarity to knowledge_input
        if remaining:
            best = max(remaining, key=lambda ex: self._similarity(knowledge_input, ex.knowledge))
            selected.append(best)
            remaining.remove(best)

        while len(selected) < k and remaining:
            mmr_scores: list[tuple[Example, float]] = []
            for ex in remaining:
                sim_to_query = self._similarity(knowledge_input, ex.knowledge)
                max_sim_to_selected = max(
                    self._similarity(ex.knowledge, s.knowledge)
                    for s in selected
                )
                mmr_score = (
                    self._lambda * sim_to_query
                    - (1 - self._lambda) * max_sim_to_selected
                )
                mmr_scores.append((ex, mmr_score))

            best_ex, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_ex)
            remaining.remove(best_ex)

        return selected


class ExampleSelectorFactory:
    """Factory for creating example selectors by name."""

    _selectors: dict[str, type[ExampleSelector]] = {
        "random": RandomExampleSelector,
        "similarity": SimilarityExampleSelector,
        "diversity": DiversityExampleSelector,
    }

    @classmethod
    def create(
        cls,
        selector_type: str,
        **kwargs: Any,
    ) -> ExampleSelector:
        """Create an example selector by type name.

        Args:
            selector_type: One of "random", "similarity", "diversity".
            **kwargs: Constructor arguments for the selector.

        Returns:
            An instance of the requested selector.

        Raises:
            ValueError: If the selector type is unknown.
        """
        selector_type = selector_type.lower()
        if selector_type not in cls._selectors:
            raise ValueError(
                f"Unknown example selector '{selector_type}'. "
                f"Available: {', '.join(cls._selectors.keys())}"
            )
        return cls._selectors[selector_type](**kwargs)

    @classmethod
    def list_selectors(cls) -> list[str]:
        """Return the list of available selector names."""
        return list(cls._selectors.keys())
