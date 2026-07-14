"""Main question generation engine.

``QuestionGeneratorEngine`` is the central orchestrator that:

1. Builds prompts via ``PromptBuilder`` (strategy-aware, type-aware)
2. Dispatches inference via ``ModelClientFactory`` (multi-backend unified)
3. Parses and sanitises raw model output
4. Computes confidence heuristics
5. Returns fully-provenanced ``GeneratedQuestion`` objects

The engine supports both single-item generation (interactive) and batch
processing (experiment runs), and integrates with the project's structured
logging and configuration systems.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

from knowprobe.core.config import GenerationConfig, PromptsConfig, get_settings
from knowprobe.core.models import (
    GeneratedQuestion,
    KnowledgeInput,
    ModelProvider,
    PromptStrategy,
    QuestionType,
)
from knowprobe.generators.base import (
    BaseQuestionGenerator,
    GenerationError,
    ModelUnavailableError,
)
from knowprobe.generators.model_client import (
    BaseModelClient,
    ModelClientFactory,
    ModelResponse,
)
from knowprobe.generators.prompt_builder import PromptBuilder
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Default few-shot examples (embedded for portability)
# --------------------------------------------------------------------------- #

_DEFAULT_FEW_SHOT_FACTUAL = [
    {
        "knowledge": {
            "content": "(巴黎, 首都, 法国) — 巴黎是法国的首都，位于塞纳河畔。",
        },
        "question": "法国的首都是哪座城市？",
    },
    {
        "knowledge": {
            "content": "(爱因斯坦, 出生日期, 1879年3月14日) — 阿尔伯特·爱因斯坦出生于德国乌尔姆。",
        },
        "question": "爱因斯坦出生于哪一年？",
    },
]

_DEFAULT_FEW_SHOT_SCHEMA = [
    {
        "knowledge": {
            "content": "Class: Person | Properties: name (string), birthDate (date), nationality (string) | Relations: worksAt (Company), hasChild (Person)",
        },
        "question": "Person类的worksAt关系的range是什么类型？",
    },
    {
        "knowledge": {
            "content": "Class: Organization | SubClassOf: Agent | Properties: name (string), founded (date) | Constraints: name cardinality 1",
        },
        "question": "Organization类继承了哪个父类？",
    },
]


# --------------------------------------------------------------------------- #
# QuestionGeneratorEngine
# --------------------------------------------------------------------------- #


class QuestionGeneratorEngine(BaseQuestionGenerator):
    """Core engine for knowledge-grounded question generation.

    Orchestrates the full pipeline:

    ::

        KnowledgeInput → PromptBuilder → ModelClient → Parse → GeneratedQuestion

    The engine is **reusable** across multiple experiments; initialise once and
    call ``generate()`` or ``generate_batch()`` repeatedly. All generation
    parameters are overridable per-call via ``**kwargs``.

    Example::

        engine = QuestionGeneratorEngine(
            model_name="llama3.1:8b",
            model_provider=ModelProvider.OLLAMA,
        )
        async with engine:
            q = await engine.generate(
                knowledge=KnowledgeInput(content="巴黎是法国的首都"),
                question_type=QuestionType.FACTUAL,
                prompt_strategy=PromptStrategy.COT,
            )
    """

    def __init__(
        self,
        model_name: str,
        model_provider: ModelProvider,
        *,
        prompt_builder: PromptBuilder | None = None,
        generation_config: GenerationConfig | None = None,
        prompts_config: PromptsConfig | None = None,
        templates_dir: str | Path | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            model_name: Model identifier (e.g. "llama3.1:8b", "gpt-4o-mini").
            model_provider: Backend provider enum.
            prompt_builder: Optional pre-configured ``PromptBuilder``.
            generation_config: Overrides for generation hyperparameters.
            prompts_config: Overrides for prompt system configuration.
            templates_dir: Directory with Jinja2 templates; defaults to
                ``configs/prompts`` relative to the project root.
        """
        super().__init__(model_name, model_provider.value)
        self.provider = model_provider
        self._gen_config = generation_config or get_settings().generation
        self._prompts_config = prompts_config or get_settings().prompts

        # Prompt builder
        if prompt_builder is not None:
            self._prompt_builder = prompt_builder
        elif templates_dir is not None:
            self._prompt_builder = PromptBuilder(templates_dir)
        else:
            # Try default location relative to project root
            default_path = Path("configs/prompts")
            if not default_path.exists():
                # Fallback: search upward from package
                pkg = Path(__file__).resolve().parent
                candidate = pkg.parents[2] / "configs" / "prompts"
                if candidate.exists():
                    default_path = candidate
            self._prompt_builder = PromptBuilder(str(default_path))

        self._model_client: BaseModelClient | None = None
        self._logger = get_logger(f"QuestionGeneratorEngine.{model_name}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialise the model client and verify connectivity."""
        if self._initialized:
            return

        self._model_client = ModelClientFactory.create(
            self.provider,
            self.model_name,
        )
        await self._model_client.__aenter__()

        health = await self._model_client.health_check()
        status = health.get("status", "unknown")
        if status == "unavailable":
            raise ModelUnavailableError(
                f"Model {self.model_name} is unavailable: {health}",
                provider=self.provider.value,
                model=self.model_name,
            )

        self._initialized = True
        self._logger.info(
            "Generator engine initialised",
            model=self.model_name,
            provider=self.provider.value,
            health_status=status,
        )

    async def shutdown(self) -> None:
        """Release model client resources."""
        if self._model_client is not None:
            await self._model_client.close()
            self._model_client = None
        self._initialized = False
        self._logger.info("Generator engine shut down")

    # ------------------------------------------------------------------ #
    # Single generation
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        knowledge: KnowledgeInput,
        question_type: QuestionType,
        prompt_strategy: PromptStrategy,
        **kwargs: Any,
    ) -> GeneratedQuestion:
        """Generate a single question from a knowledge input.

        Args:
            knowledge: Structured knowledge input.
            question_type: Target question type (factual / schema / composite).
            prompt_strategy: Prompting strategy to apply.
            **kwargs: Per-call overrides:
                - ``temperature``, ``top_p``, ``max_tokens`` — generation params
                - ``examples`` — custom few-shot examples
                - ``reasoning_steps`` — custom CoT reasoning steps
                - ``self_consistency_n`` — number of self-consistency samples

        Returns:
            ``GeneratedQuestion`` with full provenance metadata.

        Raises:
            RuntimeError: If called before ``initialize()``.
            GenerationError: If generation fails after retries.
            ModelUnavailableError: If the model is unreachable.
        """
        if not self._initialized or self._model_client is None:
            raise RuntimeError("Engine not initialized. Call initialize() or use async with.")

        start_time = time.monotonic()
        self._logger.debug(
            "Starting generation",
            source_id=knowledge.source_id,
            strategy=prompt_strategy.value,
            question_type=question_type.value,
        )

        # 1. Build prompt
        prompt = self._build_prompt(knowledge, question_type, prompt_strategy, **kwargs)

        # 2. Model inference
        try:
            response = await self._model_client.generate(
                prompt,
                max_tokens=kwargs.get("max_tokens", self._gen_config.max_length),
                temperature=kwargs.get("temperature", self._gen_config.temperature),
                top_p=kwargs.get("top_p", self._gen_config.top_p),
                top_k=kwargs.get("top_k", self._gen_config.top_k),
                num_beams=kwargs.get("num_beams", self._gen_config.num_beams),
                do_sample=kwargs.get("do_sample", self._gen_config.do_sample),
            )
        except Exception as exc:
            self._logger.error(
                "Model generation failed",
                error=str(exc),
                model=self.model_name,
                strategy=prompt_strategy.value,
            )
            raise GenerationError(
                f"Model generation failed: {exc}",
                details={
                    "model": self.model_name,
                    "provider": self.provider.value,
                    "strategy": prompt_strategy.value,
                    "question_type": question_type.value,
                    "source_id": knowledge.source_id,
                },
            ) from exc

        # 3. Parse and sanitise
        question_text = self._parse_output(response.text, question_type, prompt_strategy)

        # 4. Confidence estimation
        confidence = self._estimate_confidence(response, question_text, question_type)

        latency = (time.monotonic() - start_time) * 1000

        # 5. Build provenanced result
        return GeneratedQuestion(
            id=str(uuid.uuid4()),
            question_text=question_text,
            knowledge_input=knowledge,
            question_type=question_type,
            prompt_strategy=prompt_strategy,
            model_name=self.model_name,
            model_provider=self.provider,
            generation_params={
                "temperature": self._gen_config.temperature,
                "top_p": self._gen_config.top_p,
                "max_length": self._gen_config.max_length,
                "latency_ms": round(latency, 2),
                **{k: v for k, v in kwargs.items() if k not in ("examples", "reasoning_steps")},
            },
            raw_output=response.text,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # Batch generation
    # ------------------------------------------------------------------ #

    async def generate_batch(
        self,
        knowledges: list[KnowledgeInput],
        question_type: QuestionType,
        prompt_strategy: PromptStrategy,
        **kwargs: Any,
    ) -> list[GeneratedQuestion]:
        """Generate questions in batch for efficient experiment runs.

        The batch pipeline builds all prompts upfront, then dispatches them
        through the model client's ``generate_batch()`` (which may implement
        true concurrency or sequential processing with limits).

        Args:
            knowledges: List of knowledge inputs.
            question_type: Target question type.
            prompt_strategy: Prompting strategy.
            **kwargs: Per-call overrides (same as ``generate()``).

        Returns:
            List of ``GeneratedQuestion``, preserving input order.
        """
        if not self._initialized or self._model_client is None:
            raise RuntimeError("Engine not initialized. Call initialize() or use async with.")

        if not knowledges:
            return []

        batch_size = len(knowledges)
        self._logger.info(
            "Starting batch generation",
            batch_size=batch_size,
            strategy=prompt_strategy.value,
            question_type=question_type.value,
        )
        start_time = time.monotonic()

        # 1. Build all prompts
        prompts: list[str] = []
        for knowledge in knowledges:
            prompt = self._build_prompt(knowledge, question_type, prompt_strategy, **kwargs)
            prompts.append(prompt)

        # 2. Batch inference
        try:
            responses = await self._model_client.generate_batch(
                prompts,
                max_tokens=kwargs.get("max_tokens", self._gen_config.max_length),
                temperature=kwargs.get("temperature", self._gen_config.temperature),
                top_p=kwargs.get("top_p", self._gen_config.top_p),
                top_k=kwargs.get("top_k", self._gen_config.top_k),
                num_beams=kwargs.get("num_beams", self._gen_config.num_beams),
                do_sample=kwargs.get("do_sample", self._gen_config.do_sample),
            )
        except Exception as exc:
            self._logger.error(
                "Batch generation failed",
                error=str(exc),
                batch_size=batch_size,
            )
            raise GenerationError(
                f"Batch generation failed: {exc}",
                details={
                    "model": self.model_name,
                    "provider": self.provider.value,
                    "batch_size": batch_size,
                    "strategy": prompt_strategy.value,
                },
            ) from exc

        total_latency = (time.monotonic() - start_time) * 1000
        avg_latency = total_latency / batch_size

        # 3. Build results preserving order
        results: list[GeneratedQuestion] = []
        for knowledge, response in zip(knowledges, responses, strict=False):
            question_text = self._parse_output(response.text, question_type, prompt_strategy)
            confidence = self._estimate_confidence(response, question_text, question_type)
            results.append(
                GeneratedQuestion(
                    id=str(uuid.uuid4()),
                    question_text=question_text,
                    knowledge_input=knowledge,
                    question_type=question_type,
                    prompt_strategy=prompt_strategy,
                    model_name=self.model_name,
                    model_provider=self.provider,
                    generation_params={
                        "temperature": self._gen_config.temperature,
                        "top_p": self._gen_config.top_p,
                        "max_length": self._gen_config.max_length,
                        "latency_ms": round(avg_latency, 2),
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in ("examples", "reasoning_steps")
                        },
                    },
                    raw_output=response.text,
                    confidence=confidence,
                )
            )

        self._logger.info(
            "Batch generation completed",
            batch_size=batch_size,
            total_latency_ms=round(total_latency, 2),
            avg_latency_ms=round(avg_latency, 2),
        )
        return results

    # ------------------------------------------------------------------ #
    # Health check
    # ------------------------------------------------------------------ #

    async def health_check(self) -> dict[str, Any]:
        """Return health status of the engine and underlying model client."""
        if not self._initialized or self._model_client is None:
            return {
                "status": "unavailable",
                "reason": "not_initialized",
                "model": self.model_name,
                "provider": self.provider.value,
            }
        client_health = await self._model_client.health_check()
        return {
            "status": client_health.get("status", "unknown"),
            "model": self.model_name,
            "provider": self.provider.value,
            "client_details": client_health,
        }

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        knowledge: KnowledgeInput,
        question_type: QuestionType,
        prompt_strategy: PromptStrategy,
        **kwargs: Any,
    ) -> str:
        """Build a prompt via ``PromptBuilder``."""
        knowledge_dict = {
            "source_id": knowledge.source_id,
            "input_type": knowledge.input_type,
            "content": knowledge.content,
            "structured": knowledge.structured,
            "metadata": knowledge.metadata,
        }

        # Resolve few-shot examples
        examples = kwargs.get("examples")
        if prompt_strategy == PromptStrategy.FEW_SHOT and examples is None:
            examples = self._load_default_examples(question_type)

        return self._prompt_builder.build(
            knowledge=knowledge_dict,
            strategy=prompt_strategy,
            question_type=question_type,
            examples=examples,
            reasoning_steps=kwargs.get("reasoning_steps"),
            self_consistency_n=kwargs.get(
                "self_consistency_n", self._prompts_config.self_consistency_samples
            ),
        )

    def _load_default_examples(self, question_type: QuestionType) -> list[dict[str, Any]]:
        """Return default embedded examples for few-shot prompting."""
        if question_type == QuestionType.SCHEMA:
            return _DEFAULT_FEW_SHOT_SCHEMA
        return _DEFAULT_FEW_SHOT_FACTUAL

    # ------------------------------------------------------------------ #
    # Output parsing
    # ------------------------------------------------------------------ #

    def _parse_output(
        self,
        raw_output: str,
        question_type: QuestionType,
        strategy: PromptStrategy,
    ) -> str:
        """Parse and sanitise raw model output into a clean question.

        The parser handles:

        - ``<think>`` / ``</think>`` tags (reasoning traces)
        - CoT-style "最终问题:" / "Final question:" prefixes
        - Self-consistency "最终选择" sections
        - Multiple lines (keeps only the first question-like line)
        - Trailing whitespace and punctuation cleanup
        """
        cleaned = raw_output.strip()

        # Strip <think> tags (common in reasoning models)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # Extract final question from CoT-style output
        if strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            for marker in ["最终问题：", "最终问题:", "Final question:", "问题：", "问题:"]:
                if marker in cleaned:
                    cleaned = cleaned.split(marker, 1)[-1].strip()
                    break

        # Extract from self-consistency output
        if strategy == PromptStrategy.SELF_CONSISTENCY:
            for marker in ["最终选择", "最终选择:", "Final answer:", "最终答案:", "最佳问题:"]:
                if marker in cleaned:
                    cleaned = cleaned.split(marker, 1)[-1].strip()
                    break

        # Remove quoted wrappers
        cleaned = cleaned.strip('"').strip("'").strip("`").strip()

        # If multi-line, keep the first non-empty line that looks like a question
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        for line in lines:
            # Heuristic: first line ending with ? or containing interrogative words
            if line.endswith("?") or line.endswith("？"):
                cleaned = line
                break
        else:
            # Fallback: first line
            cleaned = lines[0] if lines else cleaned

        # Ensure it ends with a question mark
        if cleaned and not (cleaned.endswith("?") or cleaned.endswith("？")):
            cleaned += "？"

        return cleaned.strip()

    # ------------------------------------------------------------------ #
    # Confidence estimation
    # ------------------------------------------------------------------ #

    def _estimate_confidence(
        self,
        response: ModelResponse,
        question_text: str,
        question_type: QuestionType,
    ) -> float | None:
        """Estimate a confidence score [0, 1] for the generated question.

        This is a heuristic composite based on:

        1. **Length penalty**: Very short (<10 chars) or very long (>200 chars)
           questions are penalised.
        2. **Question mark check**: Missing terminal punctuation reduces score.
        3. **Type heuristic**: Schema questions are generally harder to validate
           automatically; they receive a small base penalty.
        4. **Token usage**: If the model used a reasonable number of tokens
           (not truncated), boost slightly.

        This is intentionally **not** a probabilistic confidence from the
        model logits; it is a post-hoc quality heuristic useful for filtering
        and ranking during evaluation.
        """
        score = 1.0

        # Length penalty
        q_len = len(question_text)
        if q_len < 10:
            score *= 0.5
        elif q_len > 200:
            score *= 0.8
        elif 20 <= q_len <= 100:
            score *= 1.05  # Slight bonus for "goldilocks" length

        # Question mark check
        if not (question_text.endswith("?") or question_text.endswith("？")):
            score *= 0.7

        # Schema type penalty
        if question_type == QuestionType.SCHEMA:
            score *= 0.95

        # Token usage check (anti-truncation)
        if response.usage is not None:
            completion_tokens = response.usage.get("completion_tokens", 0)
            max_expected = self._gen_config.max_length
            if completion_tokens >= max_expected * 0.95:
                # Likely truncated
                score *= 0.85

        return round(min(max(score, 0.0), 1.0), 3)
