"""Experiment management endpoints for the KnowProbe API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status

from knowprobe.api.dependencies import CommonParamsDep, RequestIdDep, SettingsDep
from knowprobe.api.schemas import (
    CreateExperimentRequest,
    ExperimentListResponse,
    ExperimentResponse,
    RunExperimentRequest,
)
from knowprobe.core.models import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    GeneratedQuestion,
)
from knowprobe.utils.logging import get_logger

logger = get_logger("api.routes.experiments")
router = APIRouter(prefix="/experiments", tags=["Experiments"])

# In-memory experiment store (replace with database in production)
_experiments: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Create experiment
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new experiment",
    description="Register a new experiment configuration for later execution.",
)
async def create_experiment(
    request: CreateExperimentRequest,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> ExperimentResponse:
    """Create a new experiment configuration.

    Args:
        request: Experiment configuration parameters.
        settings: Application configuration.
        request_id: Unique request correlation ID.

    Returns:
        ExperimentResponse with the created experiment ID.
    """
    logger.info(
        "create_experiment",
        experiment_id=request.experiment_id,
        name=request.name,
        request_id=request_id,
    )

    # Validate experiment ID
    if not request.experiment_id or len(request.experiment_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="experiment_id must be non-empty and <= 100 characters",
        )
    if request.experiment_id in _experiments:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment {request.experiment_id} already exists",
        )
    if not request.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="models must not be empty",
        )
    if not request.prompt_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt_strategies must not be empty",
        )
    if not request.question_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question_types must not be empty",
        )

    _experiments[request.experiment_id] = {
        "config": request.model_dump(),
        "status": "created",
        "created_at": datetime.utcnow().isoformat(),
        "results": None,
    }

    return ExperimentResponse(
        success=True,
        experiment_id=request.experiment_id,
        data=request,
    )


# ---------------------------------------------------------------------------
# List experiments
# ---------------------------------------------------------------------------
@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List all experiments",
)
async def list_experiments(
    params: CommonParamsDep,
    request_id: RequestIdDep,
) -> ExperimentListResponse:
    """List all registered experiments with pagination.

    Args:
        params: Pagination and sorting parameters.
        request_id: Unique request correlation ID.

    Returns:
        ExperimentListResponse with paginated experiment list.
    """
    logger.info("list_experiments", request_id=request_id)
    all_configs = [ExperimentConfig(**item["config"]) for item in _experiments.values()]

    # Sort
    if params.sort_by:
        reverse = params.sort_order == "desc"
        sort_by = params.sort_by
        all_configs.sort(
            key=lambda x: getattr(x, sort_by, x.created_at),
            reverse=reverse,
        )

    total = len(all_configs)
    start = (params.page - 1) * params.per_page
    end = start + params.per_page
    paged = all_configs[start:end]

    return ExperimentListResponse(experiments=paged, total=total)


# ---------------------------------------------------------------------------
# Get experiment
# ---------------------------------------------------------------------------
@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment details",
)
async def get_experiment(
    experiment_id: str,
    request_id: RequestIdDep,
) -> ExperimentResponse:
    """Retrieve a single experiment by ID.

    Args:
        experiment_id: Unique experiment identifier.
        request_id: Unique request correlation ID.

    Returns:
        ExperimentResponse with experiment configuration and status.
    """
    logger.info("get_experiment", experiment_id=experiment_id, request_id=request_id)
    if experiment_id not in _experiments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    item = _experiments[experiment_id]
    config = ExperimentConfig(**item["config"])
    return ExperimentResponse(
        success=True,
        experiment_id=experiment_id,
        data=config,
    )


# ---------------------------------------------------------------------------
# Delete experiment
# ---------------------------------------------------------------------------
@router.delete(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete an experiment",
)
async def delete_experiment(
    experiment_id: str,
    request_id: RequestIdDep,
) -> ExperimentResponse:
    """Delete an experiment by ID.

    Args:
        experiment_id: Unique experiment identifier.
        request_id: Unique request correlation ID.

    Returns:
        ExperimentResponse confirming deletion.
    """
    logger.info("delete_experiment", experiment_id=experiment_id, request_id=request_id)
    if experiment_id not in _experiments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    del _experiments[experiment_id]
    return ExperimentResponse(
        success=True,
        experiment_id=experiment_id,
    )


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
@router.post(
    "/{experiment_id}/run",
    response_model=ExperimentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run an experiment",
    description="Execute the experiment and return results. In production, this would be an async task.",
)
async def run_experiment(
    experiment_id: str,
    request: RunExperimentRequest,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> ExperimentResponse:
    """Run an experiment and compute results.

    Args:
        experiment_id: Unique experiment identifier.
        request: Run parameters including dry_run flag.
        settings: Application configuration.
        request_id: Unique request correlation ID.

    Returns:
        ExperimentResponse with the experiment results.
    """
    logger.info(
        "run_experiment",
        experiment_id=experiment_id,
        dry_run=request.dry_run,
        request_id=request_id,
    )

    if experiment_id not in _experiments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    item = _experiments[experiment_id]
    config = ExperimentConfig(**item["config"])

    if request.dry_run:
        return ExperimentResponse(
            success=True,
            experiment_id=experiment_id,
            data=ExperimentResult(
                experiment_id=experiment_id,
                config=config,
                questions=[],
                evaluations=[],
                summary={"dry_run": True, "estimated_models": len(config.models)},
            ),
        )

    # Placeholder: generate synthetic results
    _experiments[experiment_id]["status"] = "running"

    questions = _mock_generate_for_experiment(config)
    evaluations: list[EvaluationResult] = []
    summary = {
        "total_questions": len(questions),
        "models_used": config.models,
        "strategies_used": [s.value for s in config.prompt_strategies],
        "types_used": [t.value for t in config.question_types],
    }

    result = ExperimentResult(
        experiment_id=experiment_id,
        config=config,
        questions=questions,
        evaluations=evaluations,
        summary=summary,
    )
    _experiments[experiment_id]["status"] = "completed"
    _experiments[experiment_id]["results"] = result.model_dump()

    return ExperimentResponse(
        success=True,
        experiment_id=experiment_id,
        data=result,
    )


def _mock_generate_for_experiment(config: ExperimentConfig) -> list[GeneratedQuestion]:
    """Generate placeholder questions for an experiment."""
    from knowprobe.core.models import KnowledgeInput, ModelProvider

    questions: list[GeneratedQuestion] = []
    for model in config.models:
        for strategy in config.prompt_strategies:
            for qtype in config.question_types:
                q = GeneratedQuestion(
                    question_text=f"[{model}][{strategy}][{qtype}] What is the relationship between X and Y?",
                    knowledge_input=KnowledgeInput(
                        source_id="mock-source",
                        content="Mock knowledge content for testing.",
                    ),
                    question_type=qtype,
                    prompt_strategy=strategy,
                    model_name=model,
                    model_provider=ModelProvider.OLLAMA,
                )
                questions.append(q)
    return questions
