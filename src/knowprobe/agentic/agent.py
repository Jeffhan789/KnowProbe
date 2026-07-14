"""Agentic RAG — 用 LLM Agent 动态决策检索策略。

教学要点：
- 传统 RAG 是固定的流水线：Query -> Retrieve -> Generate。无论查询多复杂，都只检索一次。
- Agentic RAG 的核心洞察：不同查询需要不同的检索策略。
  - 简单事实查询（"爱因斯坦的生日"）：一次向量检索就够了。
  - 多跳推理查询（"和爱因斯坦同时获得诺贝尔奖的人"）：需要多次检索 + 推理。
  - 比较查询（"相对论和量子力学的区别"）：需要分别检索两个主题再综合。
- 用 LLM 作为"指挥官"，根据当前状态动态决定下一步：检索、推理、还是回答。

参考架构：ReAct (Reasoning + Acting) 循环
- Thought: 分析当前状态，确定下一步目标。
- Action: 执行具体操作（如检索新信息）。
- Observation: 观察操作结果，更新状态。
- 循环直到满足终止条件。

这与 FulfillCrew 的多 Agent 系统不同：Agentic RAG 是单 Agent 的多步决策，
不是多 Agent 协作。更适合教学，因为逻辑简单清晰。
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from knowprobe.core.models import RAGChunk, RAGQuery
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class ActionType(str, Enum):
    """Agent 可以执行的动作类型。"""

    RETRIEVE = "retrieve"  # 检索新信息
    REASON = "reason"  # 基于已有信息进行推理
    REFLECT = "reflect"  # 反思已有答案是否充分
    ANSWER = "answer"  # 生成最终答案
    STOP = "stop"  # 停止循环（无法回答或已回答）


class AgentStep(BaseModel):
    """Agent 单步决策记录。"""

    step_number: int = Field(description="Step index in the reasoning loop")
    thought: str = Field(description="Agent's reasoning about what to do next")
    action: ActionType = Field(description="Chosen action type")
    action_input: str = Field(
        default="", description="Action parameters (e.g., query for retrieve)"
    )
    observation: str = Field(default="", description="Result of the action")
    latency_ms: float = Field(default=0.0, description="Step latency in milliseconds")


class AgenticRAGState(BaseModel):
    """Agentic RAG 运行状态。

    状态包含所有已收集的信息，Agent 基于当前状态做决策。
    """

    original_query: str = Field(description="User's original query")
    current_query: str = Field(description="Current sub-query or refined query")
    retrieved_chunks: list[RAGChunk] = Field(
        default_factory=list, description="All chunks retrieved so far"
    )
    reasoning_trace: list[AgentStep] = Field(
        default_factory=list, description="Full reasoning history"
    )
    current_answer: str = Field(default="", description="Draft answer so far")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Agent's confidence in current answer"
    )
    iteration: int = Field(default=0, description="Current iteration count")
    max_iterations: int = Field(default=5, description="Maximum allowed iterations")

    @property
    def is_done(self) -> bool:
        """检查是否应终止循环。"""
        return self.iteration >= self.max_iterations or self.confidence >= 0.9

    def to_context_string(self) -> str:
        """将当前状态转换为文本描述，供 LLM 决策使用。"""
        lines = [
            f"Original Query: {self.original_query}",
            f"Current Query: {self.current_query}",
            f"Retrieved {len(self.retrieved_chunks)} chunks so far.",
            f"Current Answer Draft: {self.current_answer[:200]}..."
            if self.current_answer
            else "No answer yet.",
            f"Confidence: {self.confidence:.2f}",
            f"Iteration: {self.iteration}/{self.max_iterations}",
        ]
        if self.reasoning_trace:
            lines.append("\nPrevious Actions:")
            for step in self.reasoning_trace[-3:]:  # 只展示最近 3 步
                lines.append(
                    f"  Step {step.step_number}: {step.action.value} -> {step.observation[:80]}..."
                )
        return "\n".join(lines)


class AgenticRAGEvaluator:
    """Agentic RAG 评估器。

    使用 ReAct 模式评估 RAG 系统：不是固定检索一次，而是让 Agent 动态决定
    是否需要更多检索、是否需要分解查询、是否可以回答。

    教学流程：
    1. 初始化状态（包含用户查询）。
    2. 循环（最多 max_iterations 次）：
       a. LLM 思考：基于当前状态，下一步应该做什么？
       b. 执行动作：检索 / 推理 / 反思 / 回答。
       c. 观察结果，更新状态。
    3. 返回最终答案 + 完整推理轨迹。

    这展示了为什么 Agentic RAG 在复杂查询上优于传统 RAG：
    - 可以自适应地检索多轮信息。
    - 可以检测信息缺失并主动补充检索。
    - 可以分解复杂查询为子查询。
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
        llm_client: Any | None = None,
        max_iterations: int = 5,
    ) -> None:
        """
        Args:
            retriever: 任何实现了 retrieve(query, top_k) 的检索器。
            generator: 任何实现了 generate(query, chunks) 的生成器。
            llm_client: 用于决策的 LLM 客户端。如果为 None，使用简化规则决策。
            max_iterations: 最大循环次数。
        """
        self.retriever = retriever
        self.generator = generator
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self._logger = get_logger(f"{__name__}.evaluator")

    # ------------------------------------------------------------------
    # 核心 ReAct 循环
    # ------------------------------------------------------------------
    def evaluate(self, query: RAGQuery) -> dict[str, Any]:
        """运行完整的 Agentic RAG 评估。

        Returns:
            包含最终答案、推理轨迹、迭代次数、延迟的字典。
        """
        start_time = time.perf_counter()
        state = AgenticRAGState(
            original_query=query.query_text,
            current_query=query.query_text,
            max_iterations=self.max_iterations,
        )

        self._logger.info(
            "agentic.evaluate_start",
            query=query.query_text[:50],
            max_iterations=self.max_iterations,
        )

        while not state.is_done:
            state.iteration += 1

            # Step 1: Think —— LLM 决定下一步动作
            action, action_input, thought = self._decide_next_action(state)

            step_start = time.perf_counter()

            # Step 2: Act —— 执行动作
            observation = self._execute_action(action, action_input, state)

            step_latency = (time.perf_counter() - step_start) * 1000

            # Step 3: Record —— 记录步骤
            step = AgentStep(
                step_number=state.iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                latency_ms=step_latency,
            )
            state.reasoning_trace.append(step)

            self._logger.info(
                "agentic.step_complete",
                step=state.iteration,
                action=action.value,
                observation_preview=observation[:80],
            )

        total_latency = (time.perf_counter() - start_time) * 1000

        result = {
            "query_id": query.query_id,
            "final_answer": state.current_answer,
            "confidence": state.confidence,
            "iterations_used": state.iteration,
            "reasoning_trace": [step.model_dump() for step in state.reasoning_trace],
            "total_latency_ms": total_latency,
            "retrieved_chunks_count": len(state.retrieved_chunks),
        }

        self._logger.info(
            "agentic.evaluate_complete",
            query_id=query.query_id,
            iterations=state.iteration,
            confidence=state.confidence,
            latency_ms=total_latency,
        )
        return result

    # ------------------------------------------------------------------
    # 决策引擎
    # ------------------------------------------------------------------
    def _decide_next_action(self, state: AgenticRAGState) -> tuple[ActionType, str, str]:
        """基于当前状态决定下一步动作。

        如果配置了 LLM 客户端，使用 LLM 做决策。
        否则使用简化规则（用于教学演示，无需 LLM 后端）。
        """
        if self.llm_client is not None:
            return self._llm_decide(state)
        return self._rule_decide(state)

    def _rule_decide(self, state: AgenticRAGState) -> tuple[ActionType, str, str]:
        """基于规则的简化决策器。

        教学目的：展示即使没有 LLM，Agentic 逻辑也可以工作。
        规则：
        - 第 1 轮：RETRIEVE（先检索）
        - 第 2-3 轮：如果已有信息，REASON（推理）或 REFLECT（反思）
        - 第 4-5 轮：ANSWER（回答）或 STOP（停止）
        """
        if state.iteration == 1:
            return (
                ActionType.RETRIEVE,
                state.current_query,
                "First step: retrieve relevant information for the query.",
            )
        if state.iteration == 2 and not state.retrieved_chunks:
            return (
                ActionType.RETRIEVE,
                state.current_query,
                "No chunks retrieved yet, try broader retrieval.",
            )
        if state.iteration <= 3 and len(state.retrieved_chunks) < 3:
            return (
                ActionType.RETRIEVE,
                f"more about {state.current_query}",
                "Need more information before answering.",
            )
        if state.iteration == 4 and not state.current_answer:
            return (
                ActionType.REASON,
                "Synthesize retrieved information into a draft answer.",
                "Have enough information, start synthesizing answer.",
            )
        if state.iteration >= 4 and state.current_answer:
            return (
                ActionType.REFLECT,
                "Check if the answer is complete and accurate.",
                "Review draft answer for completeness.",
            )
        return (
            ActionType.ANSWER,
            "Generate final answer.",
            "Final step: produce the answer.",
        )

    def _llm_decide(self, state: AgenticRAGState) -> tuple[ActionType, str, str]:
        """使用 LLM 做决策。

        Prompt 设计：让 LLM 分析当前状态，输出下一步动作。
        这是 ReAct 模式的核心。
        """
        from knowprobe.llm.types import GenerationRequest

        if self.llm_client is None:
            raise RuntimeError("LLM decision mode requires an llm_client")
        prompt = self._build_decision_prompt(state)
        request = GenerationRequest(prompt=prompt, model="")
        response = self.llm_client.generate(request)
        text = response.text.strip().lower()

        # 解析 LLM 输出（简单的关键词匹配）
        if "retrieve" in text or "search" in text:
            action = ActionType.RETRIEVE
        elif "reason" in text or "synthesize" in text:
            action = ActionType.REASON
        elif "reflect" in text or "check" in text:
            action = ActionType.REFLECT
        elif "answer" in text or "final" in text:
            action = ActionType.ANSWER
        else:
            action = ActionType.STOP

        # 提取 action_input（第一行或引号中的内容）
        lines = text.split("\n")
        action_input = lines[1] if len(lines) > 1 else text

        return action, action_input, text

    def _build_decision_prompt(self, state: AgenticRAGState) -> str:
        """构建决策 prompt。"""
        return (
            "You are an intelligent RAG agent. Based on the current state, decide the next action.\n\n"
            f"{state.to_context_string()}\n\n"
            "Choose one action: RETRIEVE | REASON | REFLECT | ANSWER | STOP\n"
            "Format: ACTION\nAction input or explanation."
        )

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------
    def _execute_action(self, action: ActionType, action_input: str, state: AgenticRAGState) -> str:
        """执行具体动作并返回观察结果。"""
        if action == ActionType.RETRIEVE:
            return self._action_retrieve(action_input, state)
        if action == ActionType.REASON:
            return self._action_reason(action_input, state)
        if action == ActionType.REFLECT:
            return self._action_reflect(action_input, state)
        if action == ActionType.ANSWER:
            return self._action_answer(action_input, state)
        return "Stopped."

    def _action_retrieve(self, query: str, state: AgenticRAGState) -> str:
        """执行检索动作。"""
        results = self.retriever.retrieve(query, top_k=3)
        for r in results:
            if r.chunk not in state.retrieved_chunks:  # 去重
                state.retrieved_chunks.append(r.chunk)
        return f"Retrieved {len(results)} new chunks. Total: {len(state.retrieved_chunks)}."

    def _action_reason(self, instruction: str, state: AgenticRAGState) -> str:
        """执行推理动作：基于已有 chunks 生成草稿答案。"""
        if not state.retrieved_chunks:
            state.current_answer = "Insufficient information to answer."
            state.confidence = 0.1
            return "No chunks available. Cannot reason."

        # 使用 generator 生成草稿答案
        dummy_query = RAGQuery(query_id="reason", query_text=state.current_query)
        try:
            answer, _, _ = self.generator.generate(dummy_query, state.retrieved_chunks)
            state.current_answer = answer
            state.confidence = 0.6  # 草稿阶段置信度中等
            return f"Draft answer generated. Length: {len(answer)} chars."
        except Exception as e:
            return f"Reasoning failed: {e}"

    def _action_reflect(self, instruction: str, state: AgenticRAGState) -> str:
        """执行反思动作：评估答案是否充分。"""
        if not state.current_answer:
            state.confidence = 0.0
            return "No answer to reflect on."

        # 简单启发式：检查答案长度和是否包含 "I don't know" 类短语
        answer = state.current_answer.lower()
        if "don't know" in answer or "insufficient" in answer or "no information" in answer:
            state.confidence = 0.2
            return "Answer seems incomplete. Need more information."

        # 如果答案较长且迭代次数充足，认为可信
        if len(state.current_answer) > 100 and state.iteration >= 3:
            state.confidence = 0.85
            return "Answer looks complete and well-supported."

        state.confidence = 0.5
        return "Answer is partial. May need verification."

    def _action_answer(self, instruction: str, state: AgenticRAGState) -> str:
        """生成最终答案。"""
        if not state.current_answer:
            state.current_answer = "I cannot answer this query with the available information."
        state.confidence = min(state.confidence + 0.1, 1.0)
        return f"Final answer finalized. Confidence: {state.confidence:.2f}."
