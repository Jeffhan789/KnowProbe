"""Custom middleware for the KnowProbe API."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from knowprobe.core.config import Settings
from knowprobe.utils.logging import get_logger

logger = get_logger("api.middleware")


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------
class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique correlation ID to every incoming request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Logging middleware
# ---------------------------------------------------------------------------
class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request/response with structured data."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")

        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
            client=request.client.host if request.client else "unknown",
            request_id=request_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                request_id=request_id,
            )
            raise

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        return response


# ---------------------------------------------------------------------------
# Exception handling middleware
# ---------------------------------------------------------------------------
class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured error responses."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.exception(
                "unhandled_exception",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                error_type=type(exc).__name__,
                request_id=request_id,
            )

            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "details": [],
                    "request_id": request_id,
                },
            )


# ---------------------------------------------------------------------------
# CORS setup helper
# ---------------------------------------------------------------------------
def setup_cors(app: Any, settings: Settings) -> None:
    """Configure CORS middleware for the FastAPI application.

    Args:
        app: The FastAPI application instance.
        settings: Application settings containing allowed origins.
    """
    origins = settings.api.cors_origins
    if settings.app.environment == "development":
        # In development, also allow common local ports
        origins = list(
            set(
                origins
                + [
                    "http://localhost:3000",
                    "http://localhost:5173",
                    "http://localhost:8000",
                    "http://localhost:8501",
                ]
            )
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
    )
