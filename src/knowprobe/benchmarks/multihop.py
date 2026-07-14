"""Multi-hop reasoning benchmarks for RAG evaluation.

教学要点：
- 多跳推理 (Multi-hop Reasoning) 是指需要跨多个文档/事实组合推理才能回答的问题。
- 例如："和爱因斯坦同时获得诺贝尔奖的人是谁？"
  - 跳 1: 爱因斯坦 -> 获得 -> 诺贝尔奖 (1921)
  - 跳 2: 1921 诺贝尔奖 -> 同时获得者 -> 其他科学家
- 传统向量检索在 Multi-hop 上表现差，因为中间事实在语义上可能不直接相关。
- 标准基准：HotpotQA (2-hop), MuSiQue (2-4 hop), 2WikiMultiHopQA。

本模块提供：
1. 多跳评估数据集加载和生成。
2. 多跳评估指标（准确率、推理链完整性、中间事实召回率）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class MultiHopQuestion(BaseModel):
    """多跳问题数据模型。

    Attributes:
        question_id: 唯一标识符。
        question_text: 问题文本。
        answer: 正确答案。
        hops: 推理跳数（2, 3, 4...）。
        reasoning_chain: 推理链，每步是一个 (fact, source) 对。
        supporting_facts: 支撑答案的所有事实列表。
        difficulty: 难度级别 (easy, medium, hard)。
    """

    question_id: str
    question_text: str
    answer: str
    hops: int = Field(ge=2, description="Number of reasoning hops required")
    reasoning_chain: list[dict[str, str]] = Field(
        default_factory=list,
        description="Ordered reasoning steps: [{fact: ..., source: ...}, ...]",
    )
    supporting_facts: list[str] = Field(
        default_factory=list, description="All facts needed to answer"
    )
    difficulty: str = Field(default="medium", description="easy | medium | hard")


class MultiHopMetrics(BaseModel):
    """多跳评估指标。"""

    question_id: str
    answer_correct: bool = Field(description="Whether the final answer is correct")
    reasoning_chain_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of reasoning chain facts recovered by the RAG system",
    )
    supporting_facts_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of all supporting facts retrieved",
    )
    num_retrieval_steps: int = Field(
        default=0, description="How many retrieval steps the system took"
    )
    latency_ms: float = Field(default=0.0)


class MultiHopBenchmark:
    """多跳推理基准测试器。

    使用方式：
    1. 加载或生成多跳问题集。
    2. 对每个问题运行 RAG 系统（传统或 Agentic）。
    3. 评估答案正确性和推理链完整性。
    4. 汇总统计（按 hops 和 difficulty 分组）。

    教学价值：
    - 对比 VectorRAG 和 GraphRAG 在多跳上的表现差异。
    - 展示 Agentic RAG 如何通过多次检索逐步构建推理链。
    """

    def __init__(self, questions: list[MultiHopQuestion] | None = None) -> None:
        self.questions = questions or []
        self._logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # 数据集加载与生成
    # ------------------------------------------------------------------
    def load_hotpotqa_style(self, data: list[dict[str, Any]]) -> None:
        """加载 HotpotQA 格式的数据。

        HotpotQA 格式示例：
        {
            "question": "...",
            "answer": "...",
            "supporting_facts": [["title", "sentence_idx"], ...],
            "level": "hard"
        }
        """
        for item in data:
            q = MultiHopQuestion(
                question_id=item.get("_id", f"q_{len(self.questions)}"),
                question_text=item["question"],
                answer=item["answer"],
                hops=2,  # HotpotQA 主要是 2-hop
                difficulty=item.get("level", "medium"),
            )
            self.questions.append(q)
        self._logger.info("benchmark.loaded_hotpotqa", count=len(data))

    def generate_synthetic(
        self,
        triples: list[tuple[str, str, str]],
        num_questions: int = 10,
        hops: int = 2,
    ) -> list[MultiHopQuestion]:
        """从三元组生成合成多跳问题。

        算法：
        1. 从三元组构建图谱。
        2. 随机选择两个相距 hops 跳的节点。
        3. 构造问题："从 A 到 B 的关系是什么？" 或 "通过 A 能到达 B 吗？"

        教学目的：不需要外部数据集，用自定义三元组就能生成本地测试用例。
        """
        from knowprobe.kg.builder import GraphBuilder

        builder = GraphBuilder(name="synthetic")
        graph = builder.build_from_triples(triples)

        questions: list[MultiHopQuestion] = []
        node_ids = list(graph.nodes.keys())
        if len(node_ids) < hops + 1:
            logger.warning("benchmark.not_enough_nodes", nodes=len(node_ids), hops=hops)
            return questions

        import random

        random.seed(42)
        attempts = 0
        while len(questions) < num_questions and attempts < num_questions * 10:
            attempts += 1
            start = random.choice(node_ids)
            end = random.choice(node_ids)
            if start == end:
                continue
            paths = graph.find_paths(start, end, max_depth=hops)
            if not paths:
                continue

            # 使用最短路径作为 reasoning chain
            shortest = min(paths, key=len)
            chain = []
            for edge in shortest:
                chain.append(
                    {
                        "fact": f"{graph.nodes[edge.source].label} {edge.relation} {graph.nodes[edge.target].label}",
                        "source": edge.evidence,
                    }
                )

            question = MultiHopQuestion(
                question_id=f"synth_{len(questions)}",
                question_text=f"How is {graph.nodes[start].label} related to {graph.nodes[end].label}?",
                answer=f" through {len(shortest)} hops: "
                + " -> ".join(
                    f"{graph.nodes[e.source].label} {e.relation} {graph.nodes[e.target].label}"
                    for e in shortest
                ),
                hops=len(shortest),
                reasoning_chain=chain,
                supporting_facts=[c["fact"] for c in chain],
            )
            questions.append(question)

        self.questions.extend(questions)
        self._logger.info("benchmark.generated_synthetic", count=len(questions), hops=hops)
        return questions

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    def evaluate(
        self,
        rag_pipeline: Any,
        questions: list[MultiHopQuestion] | None = None,
    ) -> dict[str, Any]:
        """运行多跳基准评估。

        Args:
            rag_pipeline: 任何实现了 run(query) -> RAGPipelineResult 的对象。

        Returns:
            包含每个问题的评估指标和汇总统计的字典。
        """
        qs = questions or self.questions
        if not qs:
            raise ValueError("No questions to evaluate. Load or generate questions first.")

        results: list[MultiHopMetrics] = []
        for q in qs:
            from knowprobe.core.models import RAGQuery

            query = RAGQuery(
                query_id=q.question_id, query_text=q.question_text, expected_answer=q.answer
            )
            try:
                pipeline_result = rag_pipeline.run(query)
                generated_answer = pipeline_result.generated_answer
                metrics = self._score_question(q, generated_answer, pipeline_result)
                results.append(metrics)
            except Exception as e:
                logger.error("benchmark.evaluate_error", question_id=q.question_id, error=str(e))
                results.append(
                    MultiHopMetrics(question_id=q.question_id, answer_correct=False, latency_ms=0.0)
                )

        # 汇总统计
        summary = self._summarize(results)
        return {"per_question": [r.model_dump() for r in results], "summary": summary}

    def _score_question(
        self,
        question: MultiHopQuestion,
        generated_answer: str,
        pipeline_result: Any,
    ) -> MultiHopMetrics:
        """评估单个问题。"""
        # 答案正确性：简单包含检查（精确匹配需要更复杂的指标）
        answer_correct = question.answer.lower() in generated_answer.lower()

        # 推理链召回：检查 retrieved chunks 是否包含 reasoning_chain 中的事实
        retrieved_texts = [c.content.lower() for c in pipeline_result.retrieval_results]
        chain_hits = sum(
            1
            for step in question.reasoning_chain
            if any(step["fact"].lower() in t for t in retrieved_texts)
        )
        chain_recall = (
            chain_hits / len(question.reasoning_chain) if question.reasoning_chain else 0.0
        )

        # 支撑事实召回
        fact_hits = sum(
            1
            for fact in question.supporting_facts
            if any(fact.lower() in t for t in retrieved_texts)
        )
        fact_recall = (
            fact_hits / len(question.supporting_facts) if question.supporting_facts else 0.0
        )

        return MultiHopMetrics(
            question_id=question.question_id,
            answer_correct=answer_correct,
            reasoning_chain_recall=chain_recall,
            supporting_facts_recall=fact_recall,
            num_retrieval_steps=getattr(pipeline_result, "num_retrieval_steps", 1),
            latency_ms=pipeline_result.latency_ms,
        )

    def _summarize(self, results: list[MultiHopMetrics]) -> dict[str, Any]:
        """汇总统计。"""
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r.answer_correct)
        avg_chain_recall = sum(r.reasoning_chain_recall for r in results) / total
        avg_fact_recall = sum(r.supporting_facts_recall for r in results) / total
        avg_latency = sum(r.latency_ms for r in results) / total

        return {
            "total_questions": total,
            "answer_accuracy": correct / total,
            "avg_reasoning_chain_recall": round(avg_chain_recall, 4),
            "avg_supporting_facts_recall": round(avg_fact_recall, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }
