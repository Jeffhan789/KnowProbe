"""Evaluation endpoints for the KnowProbe API."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, status

from knowprobe.api.dependencies import RequestIdDep, SettingsDep
from knowprobe.api.schemas import (
    AvailableMetricsResponse,
    EvaluateBatchRequest,
    EvaluateRequest,
    EvaluationResponse,
)
from knowprobe.core.models import EvaluationResult
from knowprobe.utils.logging import get_logger

logger = get_logger("api.routes.evaluation")
router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


# ---------------------------------------------------------------------------
# Available metrics
# ---------------------------------------------------------------------------
@router.get(
    "/metrics",
    response_model=AvailableMetricsResponse,
    summary="List available evaluation metrics",
)
async def list_metrics(
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> AvailableMetricsResponse:
    """Return all configured evaluation metrics with descriptions."""
    logger.info("list_metrics", request_id=request_id)
    metrics = [
        {
            "name": "bleu",
            "full_name": "BLEU-4",
            "description": "N-gram overlap precision between generated and reference questions.",
            "range": "0.0 - 1.0",
            "category": "lexical",
        },
        {
            "name": "rouge",
            "full_name": "ROUGE-L",
            "description": "Longest common subsequence overlap for recall-oriented evaluation.",
            "range": "0.0 - 1.0",
            "category": "lexical",
        },
        {
            "name": "bert_score",
            "full_name": "BERTScore",
            "description": "Contextual embedding similarity using pre-trained BERT.",
            "range": "0.0 - 1.0",
            "category": "semantic",
        },
        {
            "name": "llm_judge",
            "full_name": "LLM Judge",
            "description": "Reference-free evaluation using an LLM as a judge.",
            "range": "1 - 5",
            "category": "model_based",
        },
    ]
    # Add any custom metrics from settings
    for metric in settings.evaluation.metrics:
        if metric not in ["bleu", "rouge", "bert_score", "llm_judge"]:
            metrics.append(
                {
                    "name": metric,
                    "full_name": metric,
                    "description": "Custom configured metric",
                    "range": "N/A",
                    "category": "custom",
                }
            )
    return AvailableMetricsResponse(metrics=metrics)


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a generated question",
    description="Run evaluation metrics on a single generated question against an optional reference.",
)
async def evaluate_question(
    request: EvaluateRequest,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> EvaluationResponse:
    """Evaluate a generated question using specified metrics.

    Args:
        request: Evaluation parameters including the generated question,
            optional reference, and metric names.
        settings: Application configuration.
        request_id: Unique request correlation ID.

    Returns:
        EvaluationResponse with computed scores.
    """
    start = time.perf_counter()
    qid = request.question.id or "unknown"
    logger.info(
        "evaluate_question_start",
        question_id=qid,
        metrics=request.metrics,
        request_id=request_id,
    )

    # Validate metrics
    invalid = [m for m in request.metrics if m not in settings.evaluation.metrics]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported metrics: {invalid}. Available: {settings.evaluation.metrics}",
        )

    try:
        scores = _compute_metrics(
            generated=request.question.question_text,
            reference=request.reference_question,
            metrics=request.metrics,
            question_id=qid,
        )
        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "evaluate_question_complete",
            question_id=qid,
            metric_count=len(scores),
            latency_ms=latency,
            request_id=request_id,
        )
        return EvaluationResponse(
            success=True,
            question_id=qid,
            scores=scores,
            latency_ms=latency,
        )
    except Exception as exc:
        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "evaluate_question_failed",
            question_id=qid,
            error=str(exc),
            latency_ms=latency,
            request_id=request_id,
        )
        return EvaluationResponse(
            success=False,
            question_id=qid,
            error=str(exc),
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
@router.post(
    "/batch",
    response_model=list[EvaluationResponse],
    status_code=status.HTTP_200_OK,
    summary="Evaluate questions in batch",
)
async def evaluate_batch(
    request: EvaluateBatchRequest,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> list[EvaluationResponse]:
    """Evaluate multiple questions in a single batch.

    Args:
        request: Batch evaluation parameters.
        settings: Application configuration.
        request_id: Unique request correlation ID.

    Returns:
        List of EvaluationResponse, one per question.
    """
    start = time.perf_counter()
    total = len(request.questions)
    logger.info(
        "evaluate_batch_start",
        total=total,
        metrics=request.metrics,
        request_id=request_id,
    )

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="questions must not be empty",
        )

    references = request.references or [None] * total
    if len(references) != total:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="references length must match questions length",
        )

    invalid = [m for m in request.metrics if m not in settings.evaluation.metrics]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported metrics: {invalid}",
        )

    results: list[EvaluationResponse] = []
    for question, ref in zip(request.questions, references):
        item_start = time.perf_counter()
        qid = question.id or "unknown"
        try:
            scores = _compute_metrics(
                generated=question.question_text,
                reference=ref,
                metrics=request.metrics,
                question_id=qid,
            )
            latency = round((time.perf_counter() - item_start) * 1000, 2)
            results.append(
                EvaluationResponse(
                    success=True,
                    question_id=qid,
                    scores=scores,
                    latency_ms=latency,
                )
            )
        except Exception as exc:
            latency = round((time.perf_counter() - item_start) * 1000, 2)
            results.append(
                EvaluationResponse(
                    success=False,
                    question_id=qid,
                    error=str(exc),
                    latency_ms=latency,
                )
            )

    total_latency = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "evaluate_batch_complete",
        total=total,
        latency_ms=total_latency,
        request_id=request_id,
    )
    return results


# ---------------------------------------------------------------------------
# Metric computation (placeholder)
# ---------------------------------------------------------------------------
def _compute_metrics(
    generated: str,
    reference: str | None,
    metrics: list[str],
    question_id: str,
) -> list[EvaluationResult]:
    """Compute evaluation metrics for a generated question.

    This is a placeholder that returns synthetic scores. In production,
    this will delegate to the evaluator pipeline (BLEU, ROUGE, BERTScore, etc.).
    """
    scores: list[EvaluationResult] = []
    for metric in metrics:
        if metric == "bleu":
            score = 0.42 if reference else 0.0
        elif metric == "rouge":
            score = 0.55 if reference else 0.0
        elif metric == "bert_score":
            score = 0.78 if reference else 0.0
        elif metric == "llm_judge":
            score = 4.2
        else:
            score = 0.0

        scores.append(
            EvaluationResult(
                question_id=question_id,
                metric_name=metric,
                score=score,
                details={
                    "reference": reference,
                    "generated_length": len(generated),
                },
            )
        )
    return scores
