"""Evaluation page for the KnowProbe Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from knowprobe.dashboard.components import render_bar_chart, render_data_table, render_radar_chart
from knowprobe.dashboard.utils import api_get, api_post, ensure_session_state, format_score
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.pages.evaluation")


def render() -> None:
    """Render the Evaluation page."""
    st.header("Evaluation")
    st.markdown("Evaluate generated questions using BLEU-4, ROUGE, BERTScore, and LLM Judge.")

    # Load available metrics
    with st.spinner("Loading metrics..."):
        metrics_data = api_get("/evaluate/metrics")

    available_metrics = ["bleu", "rouge", "bert_score", "llm_judge"]
    if metrics_data and "metrics" in metrics_data:
        available_metrics = [m["name"] for m in metrics_data["metrics"]]

    # Evaluation input
    st.markdown("---")
    st.subheader("Evaluate a Question")

    col1, col2 = st.columns(2)
    with col1:
        question_text = st.text_area(
            "Generated Question",
            value="What is the theory of relativity developed by Albert Einstein?",
            height=80,
            key="eval_question",
        )
    with col2:
        reference_text = st.text_area(
            "Reference Question (optional)",
            value="What theory did Albert Einstein develop?",
            height=80,
            key="eval_reference",
        )

    selected_metrics = st.multiselect(
        "Select Metrics",
        options=available_metrics,
        default=["bleu", "rouge"],
        key="eval_metrics",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        evaluate_btn = st.button("Evaluate", type="primary", use_container_width=True)

    if evaluate_btn:
        _handle_evaluation(question_text, reference_text, selected_metrics)

    # Results visualization
    ensure_session_state("eval_results", [])
    if st.session_state["eval_results"]:
        st.markdown("---")
        st.subheader("Results")
        _render_results(st.session_state["eval_results"])

    # Metric info
    st.markdown("---")
    st.subheader("Metric Descriptions")
    if metrics_data and "metrics" in metrics_data:
        for m in metrics_data["metrics"]:
            with st.container(border=True):
                st.markdown(f"**{m['full_name']}** ({m['name']})")
                st.markdown(m["description"])
                st.caption(f"Range: {m['range']} · Category: {m['category']}")
    else:
        st.info("Metric descriptions unavailable. Start the API server to load them.")


def _handle_evaluation(
    question_text: str,
    reference_text: str,
    metrics: list[str],
) -> None:
    """Handle evaluation request and store results."""
    payload = {
        "question": {
            "question_text": question_text,
            "knowledge_input": {
                "source_id": "eval-source",
                "content": "Evaluation context",
            },
            "question_type": "factual",
            "prompt_strategy": "cot",
            "model_name": "eval-model",
            "model_provider": "ollama",
        },
        "reference_question": reference_text if reference_text.strip() else None,
        "metrics": metrics,
    }

    with st.spinner("Running evaluation..."):
        result = api_post("/evaluate", payload)

    if result is None:
        st.error("Failed to connect to API.")
        return

    if result.get("success") and result.get("scores"):
        scores = result["scores"]
        st.session_state["eval_results"].append(
            {
                "question": question_text,
                "reference": reference_text,
                "scores": scores,
                "latency_ms": result["latency_ms"],
            }
        )
        st.success(f"Evaluation completed in {result['latency_ms']}ms")
    else:
        st.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")


def _render_results(results: list[dict[str, Any]]) -> None:
    """Render evaluation results with charts and tables."""
    # Latest result
    latest = results[-1]
    scores = latest["scores"]

    # Score cards
    cols = st.columns(len(scores))
    for col, score in zip(cols, scores, strict=False):
        col.metric(
            label=score["metric_name"].upper(),
            value=format_score(score["score"], score["metric_name"]),
        )

    # Bar chart of latest scores
    score_data = {s["metric_name"]: s["score"] for s in scores}
    fig = render_bar_chart(score_data, title="Latest Evaluation Scores")
    st.plotly_chart(fig, use_container_width=True)

    # History table
    st.markdown("#### Evaluation History")
    table_data = []
    for r in results:
        row = {"Question": r["question"][:60] + "...", "Latency (ms)": r["latency_ms"]}
        for s in r["scores"]:
            row[s["metric_name"]] = format_score(s["score"], s["metric_name"])
        table_data.append(row)
    render_data_table(table_data)

    # Radar chart if multiple metrics
    if len(scores) >= 3:
        categories = [s["metric_name"].upper() for s in scores]
        values = [s["score"] for s in scores]
        # Normalize LLM judge to 0-1 scale for radar
        normalized = [
            v / 5.0 if cat == "LLM JUDGE" else v for v, cat in zip(values, categories, strict=False)
        ]
        fig_radar = render_radar_chart(categories, normalized, title="Score Profile")
        st.plotly_chart(fig_radar, use_container_width=True)
