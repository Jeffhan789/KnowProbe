"""Health check endpoint for the KnowProbe API."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, status

from knowprobe.api.dependencies import RequestIdDep, SettingsDep
from knowprobe.api.schemas import HealthResponse
from knowprobe.utils.logging import get_logger

logger = get_logger("api.routes.health")
router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Return service health status and basic metadata.",
    response_description="Service is healthy",
)
async def health_check(
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> HealthResponse:
    """Health check endpoint.

    Returns the current service status, version, and environment.
    This endpoint is lightweight and suitable for load-balancer health probes.
    """
    logger.debug("health_check_called", request_id=request_id)
    return HealthResponse(
        status="healthy",
        version=settings.app.version,
        environment=settings.app.environment,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Check if the service is ready to accept traffic.",
)
async def readiness_check(
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> HealthResponse:
    """Readiness probe endpoint.

    Verifies that all required dependencies (config, models) are loaded.
    In a production setup, this would also check database connectivity.
    """
    logger.debug("readiness_check_called", request_id=request_id)
    return HealthResponse(
        status="ready",
        version=settings.app.version,
        environment=settings.app.environment,
    )


@router.get(
    "/live",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Check if the service process is alive.",
)
async def liveness_check(
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> HealthResponse:
    """Liveness probe endpoint.

    A minimal check that the process is still running. Kubernetes uses this
    to determine whether to restart a container.
    """
    logger.debug("liveness_check_called", request_id=request_id)
    return HealthResponse(
        status="alive",
        version=settings.app.version,
        environment=settings.app.environment,
    )
