# Agentic RAG 教程（中文）

> 本教程讲解 **Agentic RAG** 的核心原理：为什么固定流水线不够，以及如何用 ReAct 模式让 RAG 系统动态决策。

## 1. 传统 RAG 的问题

传统 RAG 是**固定流水线**：
```
Query -> Retrieve (Top-K) -> Generate -> Answer
```

无论查询多简单或多复杂，都只执行一次检索。这带来三个问题：

1. **简单查询**："相对论是谁提出的？" — 检索一次就够了，但系统仍然检索 Top-K 个文档，浪费算力。
2. **复杂查询**："和爱因斯坦同时获得诺贝尔奖的人还有谁？" — 需要多次检索（先找爱因斯坦的诺贝尔奖年份，再找同年获奖者）。
3. **信息缺失**：如果第一次检索没找到答案，系统不会自动补充检索，而是基于不完整信息生成错误答案。

## 2. Agentic RAG 的洞察

核心思想：**不同查询需要不同的检索策略**。用 LLM 作为"指挥官"，动态决定：
- 是否需要检索？
- 检索什么？
- 检索到足够信息了吗？
- 可以回答了吗？

## 3. ReAct 模式

ReAct = Reasoning + Acting（思考 + 行动）

循环结构：
```
Thought  ->  Action  ->  Observation  ->  (repeat)
```

**Step 1: Thought（思考）**
> "用户问的是多跳问题，我需要先找到爱因斯坦的诺贝尔奖信息，再找同年获奖者。"

**Step 2: Action（行动）**
> 执行 RETRIEVE，查询 "Einstein Nobel Prize"

**Step 3: Observation（观察）**
> "检索到 3 个文档，包含爱因斯坦 1921 年获得诺贝尔奖的信息。"

**Step 4: 重复**
> Thought: "现在我知道年份是 1921，需要查找同年的其他获奖者。"
> Action: RETRIEVE "Nobel Prize 1921 other winners"

## 4. 代码实践

### 4.1 使用 AgenticRAGEvaluator（无需 LLM 后端）

```python
from knowprobe.agentic.agent import AgenticRAGEvaluator
from knowprobe.core.models import RAGQuery

evaluator = AgenticRAGEvaluator(
    retriever=your_retriever,      # 任何实现 retrieve() 的对象
    generator=your_generator,      # 任何实现 generate() 的对象
    llm_client=None,               # None = 使用规则决策（教学演示用）
    max_iterations=5,
)

query = RAGQuery(query_id="q1", query_text="What is the connection between Einstein and Sweden?")
result = evaluator.evaluate(query)

print(f"答案: {result['final_answer']}")
print(f"迭代次数: {result['iterations_used']}")
print(f"推理轨迹:")
for step in result['reasoning_trace']:
    print(f"  Step {step['step_number']}: {step['action']}")
    print(f"    Thought: {step['thought'][:80]}")
    print(f"    Observation: {step['observation'][:80]}")
```

### 4.2 使用 LLM 做决策（生产环境）

```python
evaluator = AgenticRAGEvaluator(
    retriever=your_retriever,
    generator=your_generator,
    llm_client=your_llm_client,  # 提供 LLM 客户端
    max_iterations=5,
)
```

此时 LLM 会分析当前状态（已检索到多少信息、答案是否充分），输出下一步动作。

## 5. 架构要点

**Q: Agentic RAG 和传统 RAG 的区别？**
> 传统 RAG 是固定流水线，检索一次就生成答案。Agentic RAG 用 LLM 动态决策检索策略，可以多次检索、分解查询、检测信息缺失。适合复杂查询，但对简单查询有额外开销。

**Q: ReAct 模式是什么？**
> Reasoning + Acting 循环。Agent 先思考当前状态和目标，然后执行具体操作（如检索），观察结果，再决定下一步。这比固定的 Chain-of-Thought 更灵活，因为可以调用外部工具。

**Q: Agentic RAG 的代价？**
> 1. 延迟更高：每轮都需要 LLM 决策。
> 2. 成本更高：多轮 LLM 调用 + 多轮检索。
> 3. 对简单查询是过度设计：需要路由层判断何时用传统 RAG，何时用 Agentic RAG。

## 6. 参考

- ReAct: Synergizing Reasoning and Acting in Language Models [ICLR 2023]
- Agentic RAG Survey: [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
