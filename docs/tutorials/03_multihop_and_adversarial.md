# 多跳推理与对抗性评估教程（中文）

> 本教程讲解两个进阶主题：如何评估 RAG 的多跳推理能力，以及如何通过对抗性测试发现系统弱点。

## 1. 多跳推理 (Multi-hop Reasoning)

### 1.1 什么是多跳？

单跳问题：答案在一个文档中就能找到。
- "爱因斯坦是谁？" -> 文档："爱因斯坦是物理学家"

多跳问题：需要跨多个文档/事实组合推理。
- "爱因斯坦获得诺贝尔奖时，那个奖项在哪里颁发？"
  - 跳 1: 爱因斯坦 -> 获得 -> 诺贝尔奖
  - 跳 2: 诺贝尔奖 -> 颁发于 -> 瑞典

### 1.2 为什么向量检索在多跳上表现差？

向量检索基于语义相似度。但多跳问题中：
- 查询 "Einstein Nobel Prize Sweden" 的语义重心可能在 "Einstein"
- 返回的 Top-K 文档大多讲相对论，很少提到诺贝尔奖和瑞典
- 关键中间事实（"诺贝尔奖在瑞典颁发"）被遗漏了

**解决方案**：
- GraphRAG：用图谱显式连接这些事实
- Agentic RAG：多轮检索，逐步构建推理链

### 1.3 代码实践：多跳基准测试

```python
from knowprobe.benchmarks.multihop import MultiHopBenchmark
from knowprobe.kg.builder import GraphBuilder

# 从三元组构建图谱，生成合成多跳问题
triples = [
    ("Einstein", "won", "Nobel Prize"),
    ("Nobel Prize", "awarded_in", "Sweden"),
    ("Marie Curie", "won", "Nobel Prize"),
    ("Curie", "discovered", "Radium"),
]

benchmark = MultiHopBenchmark()
benchmark.generate_synthetic(triples, num_questions=5, hops=2)

# 评估你的 RAG 系统
results = benchmark.evaluate(your_rag_pipeline)
print(results["summary"])
# {
#   "total_questions": 5,
#   "answer_accuracy": 0.6,
#   "avg_reasoning_chain_recall": 0.75,
#   ...
# }
```

## 2. 对抗性评估 (Adversarial Evaluation)

### 2.1 为什么需要对抗性测试？

RAG 系统在各种"正常"查询上表现良好，但在边界情况、否定形式、相似实体混淆等场景下容易失败。

对抗性测试的目的：**系统性地发现弱点**，而不是让系统表现差。

### 2.2 四种对抗性策略

**策略 1: Distractor（混淆实体）**
- 原始: "Who invented the telephone?"
- 对抗: "Did Thomas Edison invent the telephone?"
- 测试点：RAG 是否能区分相似人物和事实

**策略 2: Negation（否定形式）**
- 原始: "What did Einstein win?"
- 对抗: "What did Einstein NOT win?"
- 测试点：RAG 是否正确处理否定语义（而不是忽略 NOT）

**策略 3: Multi-hop Variation（改变推理路径）**
- 原始: "Where was the inventor of the telephone born?"
- 对抗: "Which country did the telephone inventor's wife come from?"
- 测试点：RAG 是否过度依赖常见路径，对新路径不敏感

**策略 4: Edge Case（边界情况）**
- 原始: "What is the tallest mountain?"
- 对抗: "What is the SECOND tallest mountain?"
- 测试点：RAG 对 superlative 变体的处理（通常只回答最常见答案）

### 2.3 代码实践：生成对抗性问题

```python
from knowprobe.adversarial.generator import AdversarialQuestionGenerator

gen = AdversarialQuestionGenerator(seed=42)
questions = gen.generate(
    "What did Einstein discover?",
    answer="relativity",
    strategies=["distractor", "negation", "edge_case"]
)

for q in questions:
    print(f"[{q.strategy}] {q.adversarial_question}")
    print(f"  目标弱点: {q.target_weakness}")
```

### 2.4 评估对抗性测试结果

```python
from knowprobe.adversarial.generator import AdversarialEvaluator

evaluator = AdversarialEvaluator()
summary = evaluator.evaluate(
    adversarial_questions=questions,
    rag_answers=["..." for _ in questions],  # 你的 RAG 回答
    correct_answers=["..." for _ in questions],  # 正确答案
)

print(f"攻击成功率: {summary['attack_success_rate']}")
print(f"鲁棒性得分: {summary['robustness_score']}")
# 按策略分组
for strategy, metrics in summary["by_strategy"].items():
    print(f"  {strategy}: 失败率 {metrics['failure_rate']}")
```

## 3. 架构要点

**Q: 如何评估 RAG 系统的多跳推理能力？**
> 使用标准基准（HotpotQA、MuSiQue）或合成多跳数据集。关键指标不是最终答案准确率，而是**推理链召回率**（系统是否检索到了所有中间事实）。因为即使答案正确，如果中间事实缺失，系统可能是靠"猜"的。

**Q: 对抗性测试是什么？为什么重要？**
> 对抗性测试是系统性地构造让 RAG 系统失败的查询变体，发现鲁棒性弱点。例如否定形式、相似实体混淆、边界情况等。这比随机测试更有系统性，能指导改进方向。

**Q: 常见的 RAG 弱点有哪些？**
> 1. 否定处理：忽略 NOT 关键词。
> 2. 实体混淆：相似名字/概念混淆。
> 3. 多跳推理：遗漏中间事实。
> 4. Superlative 推理：只回答最常见答案，不处理变体。
> 5. 时间推理：无法处理"之前/之后"的时间关系。

## 4. 参考

- HotpotQA: [EMNLP 2018](https://hotpotqa.github.io/)
- MuSiQue: [ACL 2022](https://arxiv.org/abs/2108.00573)
- SafeRAG: [arXiv 2025](https://arxiv.org/abs/2501.09136)
