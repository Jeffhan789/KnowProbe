"""Demo script: Building and using a Knowledge Graph for GraphRAG.

运行方式:
    cd knowprobe_v2
    source .venv/bin/activate
    PYTHONPATH=src python examples/graphrag_demo/graphrag_demo.py

教学目的:
    1. 展示如何从文本构建知识图谱。
    2. 展示 EgoGraph 检索和路径检索的区别。
    3. 展示为什么 GraphRAG 优于纯向量检索。
"""

from knowprobe.kg.builder import RuleBasedBuilder
from knowprobe.kg.graph import KnowledgeGraph
from knowprobe.kg.retriever import EgoGraphRetriever, PathRetriever

# ----------------------------------------------------------------------
# 示例 1: 从文本构建知识图谱（RuleBasedBuilder —— 零成本）
# ----------------------------------------------------------------------
SAMPLE_TEXT = """
Albert Einstein was born in Germany. Einstein won the Nobel Prize in Physics.
The Nobel Prize was established in Sweden. Marie Curie also won the Nobel Prize.
Curie discovered radium. Radium is a radioactive element.
"""

print("=" * 60)
print("Demo 1: 从文本构建知识图谱 (RuleBasedBuilder)")
print("=" * 60)

builder = RuleBasedBuilder()
graph = builder.build_from_text(SAMPLE_TEXT, source_id="demo_text")

print(f"\n图谱统计:")
summary = graph.summary()
for key, value in summary.items():
    print(f"  {key}: {value}")

print(f"\n三元组列表:")
for s, r, o in graph.to_triples():
    print(f"  ({s}) --[{r}]--> ({o})")

# ----------------------------------------------------------------------
# 示例 2: Ego Graph 检索
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 2: Ego Graph 检索 —— '告诉我关于爱因斯坦的一切'")
print("=" * 60)

# 将节点映射到文本片段（真实场景中来自文档）
chunk_map = {
    "albert_einstein": type("Chunk", (), {"chunk_id": "c1", "content": "Einstein was a theoretical physicist."})(),
    "germany": type("Chunk", (), {"chunk_id": "c2", "content": "Germany is a country in Europe."})(),
    "nobel_prize_in_physics": type("Chunk", (), {"chunk_id": "c3", "content": "The Nobel Prize in Physics is awarded annually."})(),
    "marie_curie": type("Chunk", (), {"chunk_id": "c4", "content": "Marie Curie was a pioneering physicist."})(),
}

# 为了 chunk_map 兼容，需要转换为 RAGChunk
from knowprobe.core.models import RAGChunk
chunk_map_typed = {
    k: RAGChunk(chunk_id=f"demo_{k}", doc_id=k, content=v.content)
    for k, v in chunk_map.items()
}

ego_retriever = EgoGraphRetriever(graph, chunk_map=chunk_map_typed, k_hops=2)
results = ego_retriever.retrieve("Tell me about Einstein", top_k=5)

print(f"\n检索到 {len(results)} 个相关片段:")
for r in results:
    print(f"  [{r.score:.2f}] {r.chunk.content[:60]}...")

# ----------------------------------------------------------------------
# 示例 3: 路径检索 —— 多跳推理
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 3: 路径检索 —— '爱因斯坦和瑞典有什么关系？'")
print("=" * 60)

path_retriever = PathRetriever(graph, chunk_map=chunk_map_typed, max_depth=3)
results = path_retriever.retrieve("How is Einstein related to Sweden?", top_k=5)

print(f"\n检索到 {len(results)} 个相关片段:")
for r in results:
    print(f"  [{r.score:.2f}] {r.chunk.content[:60]}...")

# 展示路径
print("\n图谱中的路径:")
paths = graph.find_paths("albert_einstein", "sweden", max_depth=3)
if paths:
    for i, path in enumerate(paths, 1):
        path_str = " -> ".join(f"{e.source} [{e.relation}] {e.target}" for e in path)
        print(f"  路径 {i}: {path_str}")
else:
    print("  未找到路径（注意：RuleBasedBuilder 可能未提取所有关系）")

# ----------------------------------------------------------------------
# 示例 4: 图统计
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 4: 图统计与质量分析")
print("=" * 60)

print(f"\n节点数: {graph.num_nodes}")
print(f"边数: {graph.num_edges}")
print(f"图密度: {graph.density:.4f}")
print(f"平均度: {graph.avg_degree:.2f}")
print(f"关系类型: {graph.relation_types}")
print(f"\n度分布（Top 5）:")
for node_id, degree in graph.degree_distribution().items():
    print(f"  {node_id}: {degree}")

print("\n" + "=" * 60)
print("Demo 完成！")
print("=" * 60)
