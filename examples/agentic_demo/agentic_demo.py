"""Demo script: Agentic RAG — ReAct reasoning loop.

运行方式:
    cd knowprobe_v2
    source .venv/bin/activate
    PYTHONPATH=src python examples/agentic_demo/agentic_demo.py

教学目的:
    1. 展示 Agentic RAG 的 ReAct 循环。
    2. 对比固定 RAG 和 Agentic RAG 在复杂查询上的表现。
    3. 展示推理轨迹（reasoning trace）如何帮助理解系统行为。
"""

from knowprobe.agentic.agent import AgenticRAGEvaluator
from knowprobe.core.models import RAGQuery


class MockRetriever:
    """模拟检索器（无需真实 LLM 后端）。"""

    def __init__(self, data: dict[str, str]) -> None:
        self.data = data

    def retrieve(self, query: str, top_k: int = 3):
        from knowprobe.core.models import RAGChunk, RetrievalResult

        results = []
        for k, v in self.data.items():
            if any(word in query.lower() for word in k.split()):
                results.append(
                    RetrievalResult(
                        chunk=RAGChunk(chunk_id=k, doc_id=k, content=v),
                        score=0.9,
                        rank=len(results) + 1,
                    )
                )
        return results[:top_k]


class MockGenerator:
    """模拟生成器。"""

    def generate(self, query, chunks):
        answer = f"Based on {len(chunks)} chunks: " + " | ".join(c.content[:30] for c in chunks)
        return answer, "", 100.0


def main() -> None:
    # 模拟知识库
    knowledge = {
        "einstein": "Albert Einstein was a German theoretical physicist who developed relativity.",
        "nobel_prize": "The Nobel Prize in Physics was awarded to Einstein in 1921.",
        "sweden": "The Nobel Prize is awarded in Stockholm, Sweden.",
        "relativity": "Einstein's theory of relativity revolutionized modern physics.",
    }

    retriever = MockRetriever(knowledge)
    generator = MockGenerator()

    print("=" * 60)
    print("Agentic RAG Demo — ReAct Reasoning Loop")
    print("=" * 60)

    evaluator = AgenticRAGEvaluator(
        retriever=retriever,
        generator=generator,
        llm_client=None,  # 使用规则决策，无需 LLM
        max_iterations=4,
    )

    query = RAGQuery(
        query_id="demo_1",
        query_text="What is the connection between Einstein and Sweden?",
    )

    print(f"\n查询: {query.query_text}")
    print(f"最大迭代: {evaluator.max_iterations}")
    print("-" * 60)

    result = evaluator.evaluate(query)

    print(f"\n最终答案: {result['final_answer']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"实际迭代: {result['iterations_used']}")
    print(f"总延迟: {result['total_latency_ms']:.0f} ms")

    print("\n推理轨迹:")
    for step in result["reasoning_trace"]:
        print(f"\n  Step {step['step_number']}: {step['action']}")
        print(f"    Thought: {step['thought'][:80]}...")
        print(f"    Observation: {step['observation'][:80]}...")

    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
