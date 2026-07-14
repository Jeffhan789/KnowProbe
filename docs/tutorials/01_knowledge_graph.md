# KnowProbe 知识图谱教程（中文）

> 本教程面向希望理解 **GraphRAG** 原理的学习者。不需要外部 LLM 后端，零成本运行。

## 1. 为什么需要知识图谱？

传统 RAG 的工作流程：
```
用户提问 -> 向量检索 Top-K 段落 -> LLM 生成答案
```

问题：如果答案需要**多个事实组合**，怎么办？

**例**："爱因斯坦和诺贝尔奖有什么关系？"
- 事实 1：爱因斯坦获得了诺贝尔物理学奖（1921）
- 事实 2：诺贝尔奖在瑞典颁发
- 结论：爱因斯坦与瑞典的关系通过诺贝尔奖建立

向量检索的问题是：
- 检索 "爱因斯坦" 可能返回很多关于相对论的内容，但不一定包含诺贝尔奖信息。
- 检索 "诺贝尔奖" 可能返回很多获奖者名单，但不一定提到瑞典。
- **两个事实在语义上可能不相似**，所以不会同时出现在 Top-K 结果中。

知识图谱的解决方案：**显式建模实体关系**。用 `(实体-关系-实体)` 三元组把知识结构化，让检索沿着关系路径进行，而不是仅依赖语义相似度。

## 2. 核心概念

### 2.1 三元组 (Triple)
```
(Albert Einstein, won, Nobel Prize in Physics)
(Nobel Prize in Physics, awarded_in, Sweden)
```

### 2.2 图检索 vs 向量检索

| 特性 | 向量检索 | 图检索 |
|------|----------|--------|
| 擅长 | 语义相似匹配 | 关系推理、多跳查询 |
| 局限 | 无法处理多跳 | 无法处理语义相似但关系未知的情况 |
| 典型查询 | "相对论的核心观点" | "和爱因斯坦同时获得诺贝尔奖的人" |

### 2.3 为什么 GraphRAG 比纯 RAG 强？

GraphRAG = 向量检索（局部语义）+ 图检索（全局关系）。
- 局部查询：用向量检索找相关段落。
- 全局查询：用图检索找实体间的关联路径。
- 两者结合：HybridGraphRetriever。

## 3. 代码实践

### 3.1 从文本构建知识图谱

```python
from knowprobe.kg.builder import RuleBasedBuilder

builder = RuleBasedBuilder()
text = """
Albert Einstein was born in Germany. Einstein won the Nobel Prize in Physics.
The Nobel Prize was established in Sweden. Marie Curie also won the Nobel Prize.
"""
graph = builder.build_from_text(text)

# 查看统计
print(graph.summary())
# {
#   "num_nodes": 4,
#   "num_edges": 3,
#   "density": 0.25,
#   "avg_degree": 1.5,
#   ...
# }

# 查看三元组
for s, r, o in graph.to_triples():
    print(f"({s}) --[{r}]--> ({o})")
```

### 3.2 Ego Graph 检索

```python
from knowprobe.kg.retriever import EgoGraphRetriever

# 提取 "爱因斯坦" 周围 2 跳内的所有信息
retriever = EgoGraphRetriever(graph, k_hops=2)
results = retriever.retrieve("Tell me about Einstein", top_k=5)
```

### 3.3 路径检索（多跳推理）

```python
from knowprobe.kg.retriever import PathRetriever

# 找 "爱因斯坦" 到 "瑞典" 的路径
retriever = PathRetriever(graph, max_depth=3)
results = retriever.retrieve("How is Einstein related to Sweden?", top_k=5)

# 查看路径
paths = graph.find_paths("albert_einstein", "sweden", max_depth=3)
for path in paths:
    print(" -> ".join(f"{e.source} [{e.relation}] {e.target}" for e in path))
# 输出: albert_einstein [won] nobel_prize_in_physics [awarded_in] sweden
```

### 3.4 混合检索（生产环境推荐）

```python
from knowprobe.kg.retriever import HybridGraphRetriever
from knowprobe.rag.retriever import DenseRetriever
from knowprobe.rag.embeddings import SentenceTransformerEmbeddings

# 向量检索 + 图检索
dense = DenseRetriever(embedding_provider=SentenceTransformerEmbeddings())
graph_r = EgoGraphRetriever(graph, k_hops=2)
hybrid = HybridGraphRetriever(dense, graph_r, dense_weight=0.6, graph_weight=0.4)
```

## 4. 架构要点

**Q: 为什么 GraphRAG 比传统 RAG 好？**
> 传统 RAG 只依赖语义相似度，丢失了实体间的显式关系。对于多跳推理问题（如"A 和 B 的关系"），相关文档在语义上可能不相似，导致向量检索漏掉关键信息。GraphRAG 用知识图谱显式建模关系，通过图遍历发现跨文档的关联路径。

**Q: GraphRAG 的代价是什么？**
> 1. 构建成本高：需要 LLM 或 NER 从文本中提取实体关系。
> 2. 维护成本高：新知识加入时需要更新图谱结构。
> 3. 对简单查询是过度设计：如果只需要查"相对论的定义"，向量检索就够了。

**Q: 你知道哪些 GraphRAG 系统？**
> 微软 GraphRAG（社区摘要）、HippoRAG（Personalized PageRank）、LightRAG（双层检索）、LinearRAG（ICLR 2026，无关系图构建）。

## 5. 延伸阅读

- Microsoft GraphRAG: [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)
- HippoRAG: [NeurIPS 2024](https://arxiv.org/abs/2405.14831)
- LightRAG: [GitHub](https://github.com/HKUDS/LightRAG)
