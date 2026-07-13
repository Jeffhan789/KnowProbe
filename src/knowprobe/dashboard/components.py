"""Reusable Streamlit components for the KnowProbe Dashboard."""

from __future__ import annotations

from datetime import datetime

import streamlit as st
from plotly import graph_objects as go

from knowprobe.core.config import get_settings
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.components")
settings = get_settings()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the dashboard header with title and description."""
    st.title(f"{settings.dashboard.page_icon} {settings.dashboard.title}")
    st.markdown(
        "An interactive platform for knowledge-grounded question generation and RAG evaluation."
    )
    st.markdown("---")


def render_footer() -> None:
    """Render the dashboard footer with timestamp and version."""
    st.markdown("---")
    st.caption(
        f"KnowProbe v{settings.app.version} · {settings.app.environment} · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ---------------------------------------------------------------------------
# Cards / metrics
# ---------------------------------------------------------------------------
def metric_card(title: str, value: str | float, delta: str | None = None) -> None:
    """Render a styled metric card."""
    if delta is not None:
        st.metric(label=title, value=value, delta=delta)
    else:
        st.metric(label=title, value=value)


def info_card(title: str, content: str) -> None:
    """Render an informational card with a title and body."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(content)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def render_bar_chart(
    data: dict[str, float],
    title: str = "",
    x_label: str = "",
    y_label: str = "Score",
    color: str = "#636EFA",
) -> go.Figure:
    """Create a horizontal bar chart from a dictionary of scores.

    Args:
        data: Mapping of labels to numeric values.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        color: Bar color (hex).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Bar(
            x=list(data.values()),
            y=list(data.keys()),
            orientation="h",
            marker=dict(color=color),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=150, r=20, t=50, b=40),
        height=max(300, len(data) * 40 + 100),
    )
    return fig


def render_grouped_bar_chart(
    data: dict[str, dict[str, float]],
    title: str = "",
    x_label: str = "",
    y_label: str = "Score",
) -> go.Figure:
    """Create a grouped bar chart from nested data.

    Args:
        data: Mapping of outer labels to inner {label: value} dicts.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    inner_keys: set[str] = set()
    for inner in data.values():
        inner_keys.update(inner.keys())
    for inner_key in sorted(inner_keys):
        fig.add_trace(
            go.Bar(
                name=inner_key,
                x=list(data.keys()),
                y=[data[outer].get(inner_key, 0.0) for outer in data],
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        barmode="group",
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def render_radar_chart(
    categories: list[str],
    values: list[float],
    title: str = "",
    color: str = "#636EFA",
) -> go.Figure:
    """Create a radar / spider chart.

    Args:
        categories: List of category labels.
        values: List of numeric values (same length as categories).
        title: Chart title.
        color: Fill color (hex).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            line=dict(color=color),
            fillcolor=f"{color}33",  # 20% opacity
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def render_comparison_heatmap(
    data: dict[str, dict[str, float]],
    title: str = "",
) -> go.Figure:
    """Create a heatmap comparing multiple dimensions.

    Args:
        data: Nested dict of {row: {col: value}}.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    rows = list(data.keys())
    cols = list({k for inner in data.values() for k in inner.keys()})
    z = [[data[row].get(col, 0.0) for col in cols] for row in rows]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=cols,
            y=rows,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=150, r=20, t=50, b=40),
        height=max(300, len(rows) * 40 + 100),
    )
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def render_data_table(data: list[dict], use_container_width: bool = True) -> None:
    """Render a data table from a list of dictionaries."""
    if not data:
        st.info("No data available.")
        return
    st.dataframe(data, use_container_width=use_container_width)
