"""Prompt strategy engine for KnowProbe.

This module provides the complete prompt engineering infrastructure:
- Template management with Jinja2
- Few-shot example selection strategies
- Five prompt strategies: Zero-shot, Few-shot, CoT, Self-Consistency, ReAct
- Unified strategy engine facade

Usage:
    from knowprobe.prompts import PromptStrategyEngine

    engine = PromptStrategyEngine.from_settings()
    prompt = engine.build(
        strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        knowledge_input=knowledge_input,
        question_type=QuestionType.FACTUAL,
    )
"""

from knowprobe.prompts.engine import PromptStrategyEngine
from knowprobe.prompts.strategies import (
    BaseStrategy,
    CoTStrategy,
    FewShotStrategy,
    ReActStrategy,
    SelfConsistencyStrategy,
    StrategyFactory,
    ZeroShotStrategy,
)
from knowprobe.prompts.templates import PromptTemplate, TemplateRegistry
from knowprobe.prompts.examples import Example, ExampleBank, ExampleSelector

__all__ = [
    "PromptStrategyEngine",
    "BaseStrategy",
    "ZeroShotStrategy",
    "FewShotStrategy",
    "CoTStrategy",
    "SelfConsistencyStrategy",
    "ReActStrategy",
    "StrategyFactory",
    "PromptTemplate",
    "TemplateRegistry",
    "Example",
    "ExampleBank",
    "ExampleSelector",
]
