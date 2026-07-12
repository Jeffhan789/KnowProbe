#!/usr/bin/env python3
"""Start script for the KnowProbe API server."""

from __future__ import annotations

import sys

from knowprobe.api.main import app
from knowprobe.core.config import get_settings
from knowprobe.utils.logging import configure_logging, get_logger

logger = get_logger("scripts.start_api")


def main() -> None:
    """Entry point to start the Uvicorn server."""
    try:
        import uvicorn
    except ImportError as exc:
        logger.error("uvicorn_not_installed", error=str(exc))
        sys.exit(1)

    settings = get_settings()
    configure_logging(level=settings.app.log_level, debug=settings.app.debug)

    logger.info(
        "starting_api_server",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        debug=settings.app.debug,
    )

    uvicorn.run(
        "knowprobe.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers if not settings.app.debug else 1,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower(),
        access_log=settings.app.debug,
    )


if __name__ == "__main__":
    main()
