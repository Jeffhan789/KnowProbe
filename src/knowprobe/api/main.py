"""Main FastAPI application for KnowProbe."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from knowprobe.api.middleware import (
    ExceptionHandlerMiddleware,
    LoggingMiddleware,
    RequestIdMiddleware,
    setup_cors,
)
from knowprobe.api.routes import evaluation, experiments, generation, health, rag
from knowprobe.core.config import get_settings
from knowprobe.utils.logging import configure_logging, get_logger

logger = get_logger("api.main")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events.

    On startup: configure logging and load settings.
    On shutdown: release any held resources.
    """
    settings = get_settings()
    configure_logging(
        level=settings.app.log_level,
        debug=settings.app.debug,
    )
    logger.info(
        "application_startup",
        name=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment,
    )
    yield
    logger.info("application_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with all routes, middleware, and
        exception handlers wired up.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Knowledge-Grounded Question Generation and RAG Evaluation Platform",
        docs_url="/docs" if settings.app.debug else None,
        redoc_url="/redoc" if settings.app.debug else None,
        openapi_url="/openapi.json" if settings.app.debug else None,
        lifespan=lifespan,
    )

    # Middleware (order matters: request ID → logging → exception handler)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ExceptionHandlerMiddleware)

    # CORS
    setup_cors(app, settings)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return structured validation errors."""
        from knowprobe.api.schemas import ErrorDetail, ErrorResponse

        request_id = getattr(request.state, "request_id", "unknown")
        errors = [
            ErrorDetail(
                field=" ".join(str(loc) for loc in err.get("loc", [])),
                message=err.get("msg", ""),
                type=err.get("type", "validation_error"),
            )
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Validation error",
                details=errors,
                request_id=request_id,
            ).model_dump(),
        )

    # Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Routers
    app.include_router(health.router)
    app.include_router(generation.router)
    app.include_router(evaluation.router)
    app.include_router(experiments.router)
    app.include_router(rag.router)

    return app


# Global app instance (for uvicorn)
app = create_app()
