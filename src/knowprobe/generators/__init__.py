"""Question generation package exports.

All public symbols needed to consume the generator engine from other
modules (API, CLI, evaluators) are re-exported here.
"""

from knowprobe.generators.base import (
    BaseQuestionGenerator,
    GenerationError,
    ModelUnavailableError,
    PromptBuildError,
)
from knowprobe.generators.model_client import (
    BaseModelClient,
    ModelClientFactory,
    ModelResponse,
    OllamaClient,
    OpenAICompatibleClient,
    TransformersClient,
)
from knowprobe.generators.prompt_builder import PromptBuilder, PromptTemplate
from knowprobe.generators.question_generator import QuestionGeneratorEngine

__all__ = [
    # Base
    "BaseQuestionGenerator",
    "GenerationError",
    "ModelUnavailableError",
    "PromptBuildError",
    # Model clients
    "BaseModelClient",
    "ModelClientFactory",
    "ModelResponse",
    "OllamaClient",
    "OpenAICompatibleClient",
    "TransformersClient",
    # Prompt building
    "PromptBuilder",
    "PromptTemplate",
    # Engine
    "QuestionGeneratorEngine",
]
