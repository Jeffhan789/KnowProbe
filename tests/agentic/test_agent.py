"""Tests for Agentic RAG module."""

import pytest

from knowprobe.agentic.agent import ActionType, AgenticRAGState, AgentStep


class TestAgenticRAGState:
    def test_initial_state(self) -> None:
        state = AgenticRAGState(original_query="test", current_query="test")
        assert state.iteration == 0
        assert state.confidence == 0.0
        assert not state.is_done

    def test_is_done_max_iterations(self) -> None:
        state = AgenticRAGState(original_query="test", current_query="test", max_iterations=2)
        state.iteration = 2
        assert state.is_done

    def test_is_done_high_confidence(self) -> None:
        state = AgenticRAGState(original_query="test", current_query="test")
        state.confidence = 0.95
        state.iteration = 1
        assert state.is_done

    def test_to_context_string(self) -> None:
        state = AgenticRAGState(original_query="Who invented telephone?", current_query="Who invented telephone?")
        ctx = state.to_context_string()
        assert "Original Query" in ctx
        assert "Who invented telephone?" in ctx

    def test_reasoning_trace(self) -> None:
        state = AgenticRAGState(original_query="test", current_query="test")
        step = AgentStep(
            step_number=1,
            thought="Need info",
            action=ActionType.RETRIEVE,
            action_input="telephone inventor",
            observation="Found Bell",
        )
        state.reasoning_trace.append(step)
        assert len(state.reasoning_trace) == 1
        assert state.to_context_string() and "Step 1" in state.to_context_string()


class TestActionType:
    def test_action_types(self) -> None:
        assert ActionType.RETRIEVE.value == "retrieve"
        assert ActionType.REASON.value == "reason"
        assert ActionType.REFLECT.value == "reflect"
        assert ActionType.ANSWER.value == "answer"
        assert ActionType.STOP.value == "stop"
