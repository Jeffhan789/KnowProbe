"""Experiments page for the KnowProbe Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from knowprobe.core.models import PromptStrategy, QuestionType
from knowprobe.dashboard.components import render_data_table, render_grouped_bar_chart
from knowprobe.dashboard.utils import api_get, api_post, ensure_session_state, format_strategy_label, format_question_type_label
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.pages.experiments")


def render() -> None:
    """Render the Experiments page."""
    st.header("Experiments")
    st.markdown(
        "Manage and run experiments comparing models, prompt strategies, and question types."
    )

    # Tabs for Create / List / Run
    tab_create, tab_list, tab_results = st.tabs(["Create", "List", "Results"])

    with tab_create:
        _render_create_tab()
    with tab_list:
        _render_list_tab()
    with tab_results:
        _render_results_tab()


def _render_create_tab() -> None:
    """Render the experiment creation form."""
    st.subheader("Create New Experiment")

    experiment_id = st.text_input(
        "Experiment ID",
        value=f"exp-{st.session_state.get('exp_counter', 1):03d}",
        key="exp_create_id",
    )
    name = st.text_input("Name", value="Zero-shot vs Few-shot Comparison", key="exp_create_name")
    description = st.text_area("Description", key="exp_create_desc")

    models = st.multiselect(
        "Models",
        options=[
            "llama3.1:8b",
            "qwen2.5:7b",
            "flan-t5-large",
            "gpt-4o-mini",
            "deepseek-chat",
        ],
        default=["llama3.1:8b", "qwen2.5:7b"],
        key="exp_create_models",
    )

    strategies = st.multiselect(
        "Prompt Strategies",
        options=list(PromptStrategy),
        format_func=format_strategy_label,
        default=[PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT, PromptStrategy.CHAIN_OF_THOUGHT],
        key="exp_create_strategies",
    )

    types = st.multiselect(
        "Question Types",
        options=list(QuestionType),
        format_func=format_question_type_label,
        default=[QuestionType.FACTUAL, QuestionType.SCHEMA],
        key="exp_create_types",
    )

    metrics = st.multiselect(
        "Evaluation Metrics",
        options=["bleu", "rouge", "bert_score", "llm_judge"],
        default=["bleu", "rouge", "bert_score"],
        key="exp_create_metrics",
    )

    sources = st.text_area(
        "Knowledge Sources (one per line)",
        value="source-001\nsource-002",
        key="exp_create_sources",
    )

    submit = st.button("Create Experiment", type="primary", use_container_width=True)
    if submit:
        payload = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "models": models,
            "prompt_strategies": [s.value for s in strategies],
            "question_types": [t.value for t in types],
            "evaluation_metrics": metrics,
            "knowledge_sources": [s.strip() for s in sources.split("\n") if s.strip()],
        }
        with st.spinner("Creating experiment..."):
            result = api_post("/experiments", payload)
        if result and result.get("success"):
            st.success(f"Experiment '{experiment_id}' created successfully!")
            ensure_session_state("exp_counter", 1)
            st.session_state["exp_counter"] += 1
        else:
            st.error(f"Failed: {result.get('error', 'Unknown error')}")


def _render_list_tab() -> None:
    """Render the experiment list with run controls."""
    st.subheader("All Experiments")

    with st.spinner("Loading experiments..."):
        data = api_get("/experiments")

    if not data or "experiments" not in data:
        st.info("No experiments found. Create one in the 'Create' tab.")
        return

    experiments = data["experiments"]
    for exp in experiments:
        with st.container(border=True):
            cols = st.columns([3, 1, 1])
            cols[0].markdown(f"**{exp['name']}** (`{exp['experiment_id']}`)")
            cols[0].caption(exp.get("description", "No description"))
            cols[1].markdown(
                f"Models: {', '.join(exp.get('models', []))}  \n"
                f"Strategies: {len(exp.get('prompt_strategies', []))}  \n"
                f"Types: {len(exp.get('question_types', []))}"
            )
            if cols[2].button("Run", key=f"run_{exp['experiment_id']}", use_container_width=True):
                _handle_run_experiment(exp["experiment_id"])

    # Summary table
    st.markdown("---")
    table_data = [
        {
            "ID": e["experiment_id"],
            "Name": e["name"],
            "Models": ", ".join(e.get("models", [])),
            "Strategies": len(e.get("prompt_strategies", [])),
            "Types": len(e.get("question_types", [])),
        }
        for e in experiments
    ]
    render_data_table(table_data)


def _render_results_tab() -> None:
    """Render experiment results visualization."""
    st.subheader("Experiment Results")

    ensure_session_state("experiment_results", [])
    results = st.session_state["experiment_results"]

    if not results:
        st.info("Run an experiment to see results here.")
        return

    # Select experiment to view
    exp_ids = [r["experiment_id"] for r in results]
    selected = st.selectbox("Select Experiment", options=exp_ids)
    selected_result = next(r for r in results if r["experiment_id"] == selected)

    summary = selected_result.get("summary", {})
    st.markdown(f"**Total Questions:** {summary.get('total_questions', 'N/A')}")
    st.markdown(f"**Models:** {', '.join(summary.get('models_used', []))}")

    # Grouped bar chart by model and strategy
    questions = selected_result.get("questions", [])
    if questions:
        grouped_data: dict[str, dict[str, float]] = {}
        for q in questions:
            model = q["model_name"]
            strategy = format_strategy_label(q["prompt_strategy"])
            if model not in grouped_data:
                grouped_data[model] = {}
            # Synthetic score based on strategy index
            score = 0.5 + hash(strategy) % 50 / 100
            grouped_data[model][strategy] = round(score, 2)

        fig = render_grouped_bar_chart(
            grouped_data,
            title="Model × Strategy Performance",
            x_label="Model",
            y_label="Synthetic Score",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Question table
    if questions:
        st.markdown("#### Generated Questions")
        q_data = [
            {
                "Question": q["question_text"][:80] + "...",
                "Model": q["model_name"],
                "Strategy": format_strategy_label(q["prompt_strategy"]),
                "Type": format_question_type_label(q["question_type"]),
            }
            for q in questions
        ]
        render_data_table(q_data)


def _handle_run_experiment(experiment_id: str) -> None:
    """Handle experiment run request."""
    payload = {"experiment_id": experiment_id, "dry_run": False}
    with st.spinner(f"Running experiment {experiment_id}..."):
        result = api_post(f"/experiments/{experiment_id}/run", payload)

    if result and result.get("success") and result.get("data"):
        data = result["data"]
        st.success(f"Experiment completed! {data.get('summary', {}).get('total_questions', 0)} questions generated.")
        ensure_session_state("experiment_results", [])
        st.session_state["experiment_results"].append(data)
    else:
        st.error(f"Run failed: {result.get('error', 'Unknown error')}")
