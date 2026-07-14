"""Question generation endpoints for the KnowProbe API."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from knowprobe.api.dependencies import RequestIdDep, SettingsDep
from knowprobe.api.schemas import (
    AvailableStrategiesResponse,
    AvailableTypesResponse,
    BatchGenerationResponse,
    GenerateBatchRequest,
    GenerateQuestionRequest,
    GenerationResponse,
    StrategyInfo,
    TypeInfo,
)
from knowprobe.core.models import (
    GeneratedQuestion,
    ModelProvider,
    PromptStrategy,
    QuestionType,
)
from knowprobe.utils.logging import get_logger

logger = get_logger("api.routes.generation")
router = APIRouter(prefix="/generate", tags=["Question Generation"])


# ---------------------------------------------------------------------------
# Strategy / Type metadata
# ---------------------------------------------------------------------------
@router.get(
    "/strategies",
    response_model=AvailableStrategiesResponse,
    summary="List available prompt strategies",
)
async def list_strategies(
    request_id: RequestIdDep,
) -> AvailableStrategiesResponse:
    """Return all supported prompt strategies with descriptions."""
    logger.info("list_strategies", request_id=request_id)
    strategies = [
        StrategyInfo(
            name="Zero-shot",
            value=PromptStrategy.ZERO_SHOT,
            description="Direct generation without examples. Tests raw model capability.",
        ),
        StrategyInfo(
            name="Few-shot",
            value=PromptStrategy.FEW_SHOT,
            description="Provides N examples to guide the model output format.",
        ),
        StrategyInfo(
            name="Chain-of-Thought",
            value=PromptStrategy.CHAIN_OF_THOUGHT,
            description="Asks the model to reason step-by-step before producing the final answer.",
        ),
        StrategyInfo(
            name="Self-Consistency",
            value=PromptStrategy.SELF_CONSISTENCY,
            description="Samples multiple CoT chains and returns the most consistent answer.",
        ),
        StrategyInfo(
            name="ReAct",
            value=PromptStrategy.REACT,
            description="Reasoning + Acting interleaved with tool use.",
        ),
    ]
    return AvailableStrategiesResponse(strategies=strategies)


@router.get(
    "/types",
    response_model=AvailableTypesResponse,
    summary="List available question types",
)
async def list_question_types(
    request_id: RequestIdDep,
) -> AvailableTypesResponse:
    """Return all supported question types with descriptions."""
    logger.info("list_question_types", request_id=request_id)
    types = [
        TypeInfo(
            name="Factual",
            value=QuestionType.FACTUAL,
            description="Questions requiring specific facts from the knowledge base.",
        ),
        TypeInfo(
            name="Schema",
            value=QuestionType.SCHEMA,
            description="Questions about the structure, schema, or ontology of the knowledge base.",
        ),
        TypeInfo(
            name="Composite",
            value=QuestionType.COMPOSITE,
            description="Complex questions combining multiple facts or reasoning steps.",
        ),
    ]
    return AvailableTypesResponse(types=types)


# ---------------------------------------------------------------------------
# Single question generation
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=GenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a single question",
    description="Generate a question from a single knowledge input using the specified model and strategy.",
)
async def generate_question(
    request: GenerateQuestionRequest,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> GenerationResponse:
    """Generate a single question from knowledge input.

    This is the primary question generation endpoint. It takes a structured
    knowledge input and produces a question using the configured model and
    prompt strategy.

    Args:
        request: Generation parameters including knowledge, model, and strategy.
        settings: Application configuration.
        request_id: Unique request correlation ID.

    Returns:
        GenerationResponse with the generated question or error details.
    """
    start = time.perf_counter()
    logger.info(
        "generate_question_start",
        question_type=request.question_type,
        strategy=request.prompt_strategy,
        model=request.model_name,
        request_id=request_id,
    )

    try:
        # Validate model name
        if not request.model_name or len(request.model_name) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model name",
            )

        # Determine provider (simple heuristic)
        provider = ModelProvider.OLLAMA
        if any(k in request.model_name.lower() for k in ("gpt", "openai")):
            provider = ModelProvider.OPENAI
        elif "deepseek" in request.model_name.lower():
            provider = ModelProvider.DEEPSEEK
        elif "claude" in request.model_name.lower():
            provider = ModelProvider.CLAUDE

        # Merge generation params with defaults
        gen_params: dict[str, Any] = {
            "max_length": settings.generation.max_length,
            "temperature": settings.generation.temperature,
            "top_p": settings.generation.top_p,
        }
        gen_params.update(request.generation_params)

        # Build the GeneratedQuestion (placeholder for actual LLM call)
        # In production, this would delegate to the generator pipeline.
        question_text = _mock_generate_question(request)

        question = GeneratedQuestion(
            question_text=question_text,
            knowledge_input=request.knowledge,
            question_type=request.question_type,
            prompt_strategy=request.prompt_strategy,
            model_name=request.model_name,
            model_provider=provider,
            generation_params=gen_params,
            raw_output=question_text,
            confidence=0.85,
        )

        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "generate_question_complete",
            latency_ms=latency,
            request_id=request_id,
        )
        return GenerationResponse(
            success=True,
            data=question,
            latency_ms=latency,
        )

    except HTTPException:
        raise
    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "generate_question_failed",
            error=str(exc),
            latency_ms=latency,
            request_id=request_id,
        )
        return GenerationResponse(
            success=False,
            error=str(exc),
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    response_model=BatchGenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate questions in batch",
    description="Generate multiple questions from a list of knowledge inputs.",
)
async def generate_batch(
    request: GenerateBatchRequest,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> BatchGenerationResponse:
    """Generate questions in batch from multiple knowledge inputs.

    Args:
        request: Batch generation parameters.
        settings: Application configuration.
        request_id: Unique request correlation ID.

    Returns:
        BatchGenerationResponse containing results for each input.
    """
    start = time.perf_counter()
    total = len(request.knowledge_items)
    logger.info(
        "generate_batch_start",
        total=total,
        question_type=request.question_type,
        strategy=request.prompt_strategy,
        request_id=request_id,
    )

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="knowledge_items must not be empty",
        )
    if total > settings.generation.batch_size * 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds maximum ({settings.generation.batch_size * 10})",
        )

    results: list[GenerationResponse] = []
    success_count = 0
    failed_count = 0

    for idx, knowledge in enumerate(request.knowledge_items):
        item_start = time.perf_counter()
        try:
            question_text = _mock_generate_question(
                GenerateQuestionRequest(
                    knowledge=knowledge,
                    question_type=request.question_type,
                    prompt_strategy=request.prompt_strategy,
                    model_name=request.model_name,
                    generation_params=request.generation_params,
                )
            )
            provider = ModelProvider.OLLAMA
            if any(k in request.model_name.lower() for k in ("gpt", "openai")):
                provider = ModelProvider.OPENAI
            elif "deepseek" in request.model_name.lower():
                provider = ModelProvider.DEEPSEEK
            elif "claude" in request.model_name.lower():
                provider = ModelProvider.CLAUDE

            question = GeneratedQuestion(
                question_text=question_text,
                knowledge_input=knowledge,
                question_type=request.question_type,
                prompt_strategy=request.prompt_strategy,
                model_name=request.model_name,
                model_provider=provider,
                generation_params=request.generation_params,
                raw_output=question_text,
                confidence=0.85,
            )
            latency = round((time.perf_counter() - item_start) * 1000, 2)
            results.append(
                GenerationResponse(
                    success=True,
                    data=question,
                    latency_ms=latency,
                )
            )
            success_count += 1
        except Exception as exc:
            latency = round((time.perf_counter() - item_start) * 1000, 2)
            results.append(
                GenerationResponse(
                    success=False,
                    error=str(exc),
                    latency_ms=latency,
                )
            )
            failed_count += 1
            logger.warning(
                "batch_item_failed",
                index=idx,
                error=str(exc),
                request_id=request_id,
            )

    total_latency = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "generate_batch_complete",
        total=total,
        success=success_count,
        failed=failed_count,
        latency_ms=total_latency,
        request_id=request_id,
    )
    return BatchGenerationResponse(
        success=failed_count == 0,
        results=results,
        total_count=total,
        success_count=success_count,
        failed_count=failed_count,
        total_latency_ms=total_latency,
    )


# ---------------------------------------------------------------------------
# Mock helper (placeholder until generator pipeline is wired)
# ---------------------------------------------------------------------------
def _mock_generate_question(request: GenerateQuestionRequest) -> str:
    """Produce a placeholder question text based on the input.

    In production, this will be replaced by an actual call to the
    question generation pipeline (LLM + prompt builder).
    """
    strategy_label = request.prompt_strategy.replace("_", " ").title()
    type_label = request.question_type.replace("_", " ").title()
    content_preview = request.knowledge.content[:50]
    return f"[{strategy_label}] {type_label} question about: {content_preview}..."
