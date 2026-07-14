#!/usr/bin/env python3
"""Start script for the KnowProbe Streamlit Dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from knowprobe.core.config import get_settings
from knowprobe.utils.logging import configure_logging, get_logger

logger = get_logger("scripts.start_dashboard")


def main() -> None:
    """Entry point to start the Streamlit dashboard."""
    settings = get_settings()
    configure_logging(level=settings.app.log_level, debug=settings.app.debug)

    dashboard_path = Path(__file__).parent.parent / "src" / "knowprobe" / "dashboard" / "app.py"
    if not dashboard_path.exists():
        logger.error("dashboard_app_not_found", path=str(dashboard_path))
        sys.exit(1)

    logger.info(
        "starting_dashboard",
        port=settings.dashboard.port,
        app_path=str(dashboard_path),
    )

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_path),
                "--server.port",
                str(settings.dashboard.port),
                "--server.address",
                "0.0.0.0",
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("dashboard_exited_with_error", returncode=exc.returncode)
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        logger.info("dashboard_shutdown_by_user")


if __name__ == "__main__":
    main()
