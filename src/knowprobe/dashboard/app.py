"""KnowProbe Dashboard - Streamlit application."""

from __future__ import annotations

from types import ModuleType

import streamlit as st

from knowprobe.core.config import get_settings
from knowprobe.dashboard.components import render_footer, render_header
from knowprobe.dashboard.pages import evaluation, experiments, generation, rag
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.app")
settings = get_settings()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=settings.dashboard.title,
    page_icon=settings.dashboard.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGE_MAP: dict[str, ModuleType] = {
    "Question Generation": generation,
    "Evaluation": evaluation,
    "Experiments": experiments,
    "RAG Evaluation": rag,
}


def main() -> None:
    """Render the KnowProbe dashboard."""
    render_header()

    with st.sidebar:
        st.title(f"{settings.dashboard.page_icon} KnowProbe")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            list(PAGE_MAP.keys()),
            index=0,
        )
        st.markdown("---")
        st.markdown(f"**Version:** {settings.app.version}")
        st.markdown(f"**Environment:** `{settings.app.environment}`")
        st.markdown("---")

    # Render selected page
    selected_module = PAGE_MAP[page]
    selected_module.render()

    render_footer()


if __name__ == "__main__":
    main()
