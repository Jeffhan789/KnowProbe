"""Question Generation page for the KnowProbe Dashboard."""

from __future__ import annotations

import streamlit as st

from knowprobe.core.config import get_settings
from knowprobe.core.models import KnowledgeInput, PromptStrategy, QuestionType
from knowprobe.dashboard.components import info_card, metric_card, render_bar_chart
from knowprobe.dashboard.utils import (
    api_get,
    api_post,
    ensure_session_state,
    format_question_type_label,
    format_strategy_label,
)
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.pages.generation")
settings = get_settings()


def render() -> None:
    """Render the Question Generation page."""
    st.header("Question Generation")
    st.markdown(
        "Generate knowledge-grounded questions using different models and prompt strategies."
    )

    # Fetch available metadata from API
    with st.spinner("Loading available options..."):
        strategies_data = api_get("/generate/strategies")
        types_data = api_get("/generate/types")

    strategies = [PromptStrategy.CHAIN_OF_THOUGHT]
    if strategies_data and "strategies" in strategies_data:
        strategies = [PromptStrategy(s["value"]) for s in strategies_data["strategies"]]

    types = [QuestionType.FACTUAL]
    if types_data and "types" in types_data:
        types = [QuestionType(t["value"]) for t in types_data["types"]]

    # Sidebar controls
    col1, col2 = st.columns(2)
    with col1:
        selected_strategy = st.selectbox(
            "Prompt Strategy",
            options=strategies,
            format_func=format_strategy_label,
            key="gen_strategy",
        )
    with col2:
        selected_type = st.selectbox(
            "Question Type",
            options=types,
            format_func=format_question_type_label,
            key="gen_type",
        )

    model_name = st.text_input(
        "Model Name",
        value=settings.models.local.default_model,
        help="e.g., llama3.1:8b, qwen2.5:7b, flan-t5-large",
    )

    # Knowledge input
    st.markdown("---")
    st.subheader("Knowledge Input")
    input_type = st.selectbox(
        "Input Type",
        options=["triple", "schema", "text", "entity"],
        key="gen_input_type",
    )
    source_id = st.text_input(
        "Source ID",
        value="test-source-001",
        key="gen_source_id",
    )
    knowledge_content = st.text_area(
        "Knowledge Content",
        value="Albert Einstein was a physicist who developed the theory of relativity.",
        height=120,
        key="gen_content",
    )

    # Generation parameters
    with st.expander("Advanced Parameters"):
        temp = st.slider("Temperature", 0.0, 1.5, settings.generation.temperature, 0.05)
        top_p = st.slider("Top-P", 0.0, 1.0, settings.generation.top_p, 0.05)
        max_len = st.number_input(
            "Max Length",
            min_value=16,
            max_value=1024,
            value=settings.generation.max_length,
        )

    # Generate button
    st.markdown("---")
    generate_btn = st.button("Generate Question", type="primary", use_container_width=True)

    if generate_btn:
        _handle_generation(
            source_id=source_id,
            input_type=input_type,
            content=knowledge_content,
            question_type=selected_type,
            strategy=selected_strategy,
            model_name=model_name,
            temperature=temp,
            top_p=top_p,
            max_length=max_len,
        )

    # Batch generation section
    st.markdown("---")
    st.subheader("Batch Generation")
    batch_content = st.text_area(
        "Enter multiple knowledge items (one per line)",
        height=80,
        key="gen_batch_content",
    )
    batch_btn = st.button("Generate Batch", use_container_width=True)

    if batch_btn and batch_content.strip():
        _handle_batch_generation(
            lines=batch_content.strip().split("\n"),
            question_type=selected_type,
            strategy=selected_strategy,
            model_name=model_name,
        )

    # Display session history
    ensure_session_state("gen_history", [])
    if st.session_state["gen_history"]:
        st.markdown("---")
        st.subheader("Generation History")
        for idx, item in enumerate(reversed(st.session_state["gen_history"])):
            with st.container(border=True):
                st.markdown(f"**{item['question_text']}**")
                cols = st.columns(4)
                cols[0].caption(f"Model: {item['model_name']}")
                cols[1].caption(f"Strategy: {format_strategy_label(item['strategy'])}")
                cols[2].caption(f"Type: {format_question_type_label(item['type'])}")
                cols[3].caption(f"Latency: {item['latency_ms']}ms")
                if item.get("raw_output"):
                    with st.expander("Raw Output"):
                        st.text(item["raw_output"])


def _handle_generation(
    source_id: str,
    input_type: str,
    content: str,
    question_type: QuestionType,
    strategy: PromptStrategy,
    model_name: str,
    temperature: float,
    top_p: float,
    max_length: int,
) -> None:
    """Handle a single question generation request."""
    payload = {
        "knowledge": {
            "source_id": source_id,
            "input_type": input_type,
            "content": content,
            "structured": {},
            "metadata": {},
        },
        "question_type": question_type,
        "prompt_strategy": strategy,
        "model_name": model_name,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
        },
    }

    with st.spinner("Generating question..."):
        result = api_post("/generate", payload)

    if result is None:
        st.error("Failed to connect to API. Is the server running?")
        return

    if result.get("success") and result.get("data"):
        data = result["data"]
        st.success("Question generated successfully!")
        st.markdown(f"### {data['question_text']}")

        cols = st.columns(4)
        cols[0].metric("Model", data["model_name"])
        cols[1].metric("Strategy", format_strategy_label(data["prompt_strategy"]))
        cols[2].metric("Type", format_question_type_label(data["question_type"]))
        cols[3].metric("Latency", f"{result['latency_ms']}ms")

        if data.get("confidence"):
            st.progress(data["confidence"], text=f"Confidence: {data['confidence']:.2f}")

        # Add to history
        st.session_state["gen_history"].append(
            {
                "question_text": data["question_text"],
                "model_name": data["model_name"],
                "strategy": data["prompt_strategy"],
                "type": data["question_type"],
                "latency_ms": result["latency_ms"],
                "raw_output": data.get("raw_output", ""),
            }
        )
    else:
        st.error(f"Generation failed: {result.get('error', 'Unknown error')}")


def _handle_batch_generation(
    lines: list[str],
    question_type: QuestionType,
    strategy: PromptStrategy,
    model_name: str,
) -> None:
    """Handle a batch question generation request."""
    knowledge_items = [
        {
            "source_id": f"batch-{i:03d}",
            "input_type": "text",
            "content": line.strip(),
            "structured": {},
            "metadata": {},
        }
        for i, line in enumerate(lines)
        if line.strip()
    ]

    payload = {
        "knowledge_items": knowledge_items,
        "question_type": question_type,
        "prompt_strategy": strategy,
        "model_name": model_name,
    }

    with st.spinner(f"Generating {len(knowledge_items)} questions..."):
        result = api_post("/generate/batch", payload)

    if result is None:
        st.error("Failed to connect to API.")
        return

    if result.get("success"):
        st.success(
            f"Batch complete: {result['success_count']} succeeded, "
            f"{result['failed_count']} failed in {result['total_latency_ms']}ms"
        )
        for r in result.get("results", []):
            if r.get("success") and r.get("data"):
                d = r["data"]
                st.markdown(f"- **{d['question_text']}** ({d['model_name']})")
    else:
        st.error(f"Batch failed: {result.get('error', 'Unknown error')}")
