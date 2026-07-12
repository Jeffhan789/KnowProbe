"""Structured logging utilities."""

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", debug: bool = False) -> None:
    """Configure structured logging."""
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        timestamper,
        structlog.processors.format_exc_info,
    ]

    if debug:
        shared_processors.append(structlog.dev.ConsoleRenderer())
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)
