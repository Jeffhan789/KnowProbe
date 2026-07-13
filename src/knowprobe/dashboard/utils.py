"""Utility functions for the KnowProbe Dashboard."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from knowprobe.core.config import get_settings
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.utils")
settings = get_settings()

API_BASE_URL = f"http://{settings.api.host}:{settings.api.port}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def api_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Perform a GET request to the API with caching.

    Args:
        endpoint: API path (e.g. /health).
        params: Optional query parameters.

    Returns:
        Parsed JSON response or None on failure.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.warning("api_get_failed", url=url, error=str(exc))
        return None


def api_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Perform a POST request to the API.

    Args:
        endpoint: API path (e.g. /generate).
        payload: JSON body payload.

    Returns:
        Parsed JSON response or None on failure.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.warning("api_post_failed", url=url, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_strategy_label(strategy: str) -> str:
    """Convert a strategy enum value to a human-readable label."""
    mapping = {
        "zero_shot": "Zero-shot",
        "few_shot": "Few-shot",
        "cot": "Chain-of-Thought",
        "self_consistency": "Self-Consistency",
        "react": "ReAct",
    }
    return mapping.get(strategy, strategy.replace("_", " ").title())


def format_question_type_label(qtype: str) -> str:
    """Convert a question type enum value to a human-readable label."""
    mapping = {
        "factual": "Factual",
        "schema": "Schema",
        "composite": "Composite",
    }
    return mapping.get(qtype, qtype.replace("_", " ").title())


def format_score(score: float, metric: str) -> str:
    """Format a metric score for display.

    Args:
        score: Raw numeric score.
        metric: Metric name (affects formatting rules).

    Returns:
        Formatted string representation.
    """
    if metric == "llm_judge":
        return f"{score:.1f}/5"
    return f"{score:.3f}"


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def ensure_session_state(key: str, default: Any) -> Any:
    """Ensure a key exists in Streamlit session state with a default value."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def clear_session_state(pattern: str) -> None:
    """Remove all session state keys matching a substring."""
    for key in list(st.session_state.keys()):
        if pattern in str(key):
            del st.session_state[key]
