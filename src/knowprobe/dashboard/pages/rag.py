"""RAG Evaluation page for the KnowProbe Dashboard."""

from __future__ import annotations

from typing import Any

import streamlit as st

from knowprobe.dashboard.components import (
    render_bar_chart,
)
from knowprobe.dashboard.utils import api_post, ensure_session_state
from knowprobe.utils.logging import get_logger

logger = get_logger("dashboard.pages.rag")


def render() -> None:
    """Render the RAG Evaluation page."""
    st.header("RAG Evaluation")
    st.markdown(
        "Evaluate Retrieval-Augmented Generation pipelines: retrieve documents "
        "and assess answer quality."
    )

    # Document collection
    st.markdown("---")
    st.subheader("Document Collection")
    docs_input = st.text_area(
        "Enter documents (format: title | content, one per line)",
        height=120,
        value=(
            "Einstein Bio | Albert Einstein was a German-born theoretical physicist.\n"
            "Relativity | The theory of relativity transformed our understanding of space and time.\n"
            "Nobel Prize | Einstein received the 1921 Nobel Prize in Physics.\n"
            "Quantum Mechanics | Einstein was also a pioneer in quantum theory.\n"
            "Cosmology | His work on cosmology led to the expansion of the universe concept."
        ),
        key="rag_docs",
    )

    # Query input
    st.markdown("---")
    st.subheader("Query")
    query_text = st.text_input(
        "Query Text",
        value="What was Albert Einstein's major contribution to physics?",
        key="rag_query",
    )
    expected_answer = st.text_input(
        "Expected Answer (optional, for evaluation)",
        value="He developed the theory of relativity.",
        key="rag_expected",
    )

    top_k = st.slider("Top-K Retrieval", 1, 10, 3, key="rag_top_k")

    # RAG Query
    col1, col2 = st.columns([1, 1])
    with col1:
        query_btn = st.button("Execute RAG Query", type="primary", use_container_width=True)
    with col2:
        eval_btn = st.button("Evaluate RAG Result", use_container_width=True)

    if query_btn:
        _handle_rag_query(docs_input, query_text, expected_answer, top_k)

    if eval_btn:
        _handle_rag_evaluate(query_text)

    # Display results
    ensure_session_state("rag_results", [])
    if st.session_state["rag_results"]:
        st.markdown("---")
        st.subheader("RAG Results")
        _render_rag_results(st.session_state["rag_results"])


def _parse_documents(text: str) -> list[dict[str, Any]]:
    """Parse document input text into structured document dicts.

    Args:
        text: Raw text with documents in "title | content" format.

    Returns:
        List of document dictionaries.
    """
    documents: list[dict[str, Any]] = []
    for i, line in enumerate(text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            title, content = line.split("|", 1)
            documents.append(
                {
                    "doc_id": f"doc-{i:03d}",
                    "title": title.strip(),
                    "content": content.strip(),
                    "metadata": {},
                }
            )
        else:
            documents.append(
                {
                    "doc_id": f"doc-{i:03d}",
                    "title": f"Doc {i}",
                    "content": line,
                    "metadata": {},
                }
            )
    return documents


def _handle_rag_query(
    docs_input: str,
    query_text: str,
    expected_answer: str,
    top_k: int,
) -> None:
    """Handle RAG query execution."""
    documents = _parse_documents(docs_input)
    if not documents:
        st.error("Please provide at least one document.")
        return

    payload = {
        "query": {
            "query_id": "rag-query-001",
            "query_text": query_text,
            "expected_answer": expected_answer,
            "relevant_doc_ids": [],
        },
        "documents": documents,
        "top_k": top_k,
        "retriever_type": "dense",
    }

    with st.spinner("Executing RAG query..."):
        result = api_post("/rag/query", payload)

    if result is None:
        st.error("Failed to connect to API.")
        return

    if result.get("success") and result.get("result"):
        data = result["result"]
        st.success(f"RAG query completed in {result['latency_ms']}ms")

        st.markdown("#### Retrieved Documents")
        for doc in data.get("retrieved_docs", []):
            with st.container(border=True):
                st.markdown(f"**{doc['title']}**")
                st.markdown(doc["content"])
                st.caption(f"ID: {doc['doc_id']}")

        st.markdown("#### Generated Answer")
        st.info(data.get("generated_answer", "No answer generated."))

        st.session_state["rag_results"].append(
            {
                "query_id": data["query_id"],
                "query_text": query_text,
                "retrieved_count": len(data.get("retrieved_docs", [])),
                "generated_answer": data.get("generated_answer", ""),
                "latency_ms": result["latency_ms"],
                "eval_scores": {},
            }
        )
    else:
        st.error(f"RAG query failed: {result.get('error', 'Unknown error')}")


def _handle_rag_evaluate(query_text: str) -> None:
    """Handle RAG evaluation."""
    if not st.session_state["rag_results"]:
        st.error("Please run a RAG query first.")
        return

    latest = st.session_state["rag_results"][-1]
    payload = {
        "rag_result": {
            "query_id": latest["query_id"],
            "retrieved_docs": [],
            "generated_answer": latest["generated_answer"],
            "evaluation_scores": {},
            "latency_ms": latest["latency_ms"],
        },
        "metrics": ["retrieval_accuracy", "answer_relevance", "faithfulness"],
    }

    with st.spinner("Evaluating RAG result..."):
        result = api_post("/rag/evaluate", payload)

    if result and result.get("success") and result.get("scores"):
        scores = result["scores"]
        latest["eval_scores"] = scores
        st.success("RAG evaluation completed!")

        # Score cards
        cols = st.columns(len(scores))
        for col, (metric, score) in zip(cols, scores.items(), strict=False):
            col.metric(label=metric.replace("_", " ").title(), value=f"{score:.2f}")

        # Bar chart
        fig = render_bar_chart(scores, title="RAG Quality Scores")
        st.plotly_chart(fig, use_container_width=True)
    else:
        error = result.get("error", "Unknown error") if result else "API unavailable"
        st.error(f"Evaluation failed: {error}")


def _render_rag_results(results: list[dict[str, Any]]) -> None:
    """Render RAG result history."""
    for idx, r in enumerate(results):
        with st.expander(f"Result {idx + 1}: {r['query_text'][:50]}..."):
            st.markdown(f"**Query:** {r['query_text']}")
            st.markdown(f"**Retrieved:** {r['retrieved_count']} docs")
            st.markdown(f"**Answer:** {r['generated_answer']}")
            st.caption(f"Latency: {r['latency_ms']}ms")
            if r.get("eval_scores"):
                st.markdown("**Scores:**")
                for metric, score in r["eval_scores"].items():
                    st.markdown(f"- {metric}: {score:.2f}")
