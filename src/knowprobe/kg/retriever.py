"""Graph-based retriever for RAG pipeline.

教学要点：
- 传统向量检索只考虑语义相似度，忽略了实体间的显式关系。
- 图检索器 (GraphRetriever) 利用知识图谱中的关系路径进行检索，
  能回答需要多跳推理的问题（如 "和爱因斯坦同时获得诺贝尔奖的人"）。
- 本模块实现三种检索策略：
  1. EgoGraphRetriever：提取查询中实体周围的 k-hop 子图。
  2. PathRetriever：找到查询实体与答案实体之间的路径。
  3. HybridGraphRetriever：结合向量检索 + 图检索，取并集。

参考：
- HippoRAG (Gutiérrez et al., 2024) 使用 Personalized PageRank 在图上传播检索信号。
- LightRAG (Guo et al., 2024) 使用双层检索（局部 + 全局）。
"""

from __future__ import annotations

from typing import Any

from knowprobe.core.models import RAGChunk, RetrievalResult
from knowprobe.kg.graph import KnowledgeGraph
from knowprobe.rag.retriever import BaseRetriever, DenseRetriever
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class GraphRetriever(BaseRetriever):
    """基于知识图谱的检索器基类。

    所有图检索器都基于一个已构建的知识图谱进行操作。
    图谱中的每个节点对应一个实体，节点的 properties 中可存储关联的文本片段。
    """

    def __init__(self, graph: KnowledgeGraph, chunk_map: dict[str, RAGChunk] | None = None) -> None:
        """
        Args:
            graph: 已构建的知识图谱。
            chunk_map: 节点 ID -> 文本片段的映射。如果节点本身包含文本，可为 None。
        """
        self.graph = graph
        self.chunk_map = chunk_map or {}
        logger.info("retriever.graph_init", nodes=graph.num_nodes, edges=graph.num_edges)

    def index_documents(self, documents: list[Any]) -> None:
        """图检索器不直接索引文档，而是通过 GraphBuilder 构建图谱。

        如果传入文档，会尝试用默认的 RuleBasedBuilder 构建图谱。
        """
        from knowprobe.kg.builder import RuleBasedBuilder

        builder = RuleBasedBuilder()
        for doc in documents:
            text = getattr(doc, "content", str(doc))
            source_id = getattr(doc, "doc_id", "unknown")
            sub_graph = builder.build_from_text(text, source_id=source_id)
            # 合并子图到主图
            for node in sub_graph.nodes.values():
                self.graph.add_node(node)
            for edge in sub_graph.edges:
                self.graph.add_edge(edge)
        logger.info(
            "retriever.graph_indexed",
            total_nodes=self.graph.num_nodes,
            total_edges=self.graph.num_edges,
        )

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """子类必须实现具体的检索逻辑。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 辅助方法：从查询文本中提取实体
    # ------------------------------------------------------------------
    def _extract_entities_from_query(self, query_text: str) -> list[str]:
        """从查询中提取图谱中存在的实体。

        简单实现：遍历图谱中所有节点 label，看是否在查询中出现。
        生产环境可用 NER 模型提高精度。
        """
        found: list[str] = []
        for node in self.graph.nodes.values():
            if node.label.lower() in query_text.lower():
                found.append(node.id)
        return found

    def _chunks_to_results(
        self, chunks: list[RAGChunk], scores: list[float] | None = None
    ) -> list[RetrievalResult]:
        """将文本片段包装为 RetrievalResult。"""
        results: list[RetrievalResult] = []
        for i, chunk in enumerate(chunks):
            score = scores[i] if scores and i < len(scores) else 1.0
            results.append(RetrievalResult(chunk=chunk, score=score, rank=i + 1))
        return results


class EgoGraphRetriever(GraphRetriever):
    """Ego Graph 检索器。

    检索策略：
    1. 从查询中提取提及的实体。
    2. 对每个实体提取其 k-hop ego graph（子图）。
    3. 将子图中的所有节点关联的文本片段作为检索结果。

    适用场景："告诉我关于 X 的一切"类型的查询（局部信息聚合）。
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        chunk_map: dict[str, RAGChunk] | None = None,
        k_hops: int = 2,
    ) -> None:
        super().__init__(graph, chunk_map)
        self.k_hops = k_hops
        logger.info("retriever.ego_init", k_hops=k_hops)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        entity_ids = self._extract_entities_from_query(query_text)
        if not entity_ids:
            logger.warning("retriever.ego_no_entities", query=query_text[:50])
            return []

        # 收集所有相关子图中的节点
        all_node_ids: set[str] = set()
        for eid in entity_ids:
            ego = self.graph.ego_graph(eid, k_hops=self.k_hops)
            all_node_ids.update(ego.nodes.keys())

        # 将这些节点映射回文本片段
        chunks: list[RAGChunk] = []
        for nid in all_node_ids:
            if nid in self.chunk_map:
                chunks.append(self.chunk_map[nid])
            # 如果节点自身存储了文本，也可以直接使用
            elif nid in self.graph.nodes and (
                content := self.graph.nodes[nid].properties.get("content")
            ):
                chunks.append(RAGChunk(chunk_id=f"kg_{nid}", doc_id=nid, content=str(content)))

        logger.info(
            "retriever.ego_search",
            query=query_text[:50],
            entities=entity_ids,
            nodes_found=len(all_node_ids),
            chunks_found=len(chunks),
        )
        return self._chunks_to_results(chunks[:top_k])


class PathRetriever(GraphRetriever):
    """路径检索器。

    检索策略：
    1. 从查询中提取起始实体和目标实体（假设查询中有两个实体）。
    2. 在图谱中搜索它们之间的路径（最大深度可控）。
    3. 将路径上的所有节点关联的文本片段作为检索结果。

    适用场景："A 和 B 之间的关系是什么"类型的多跳推理查询。

    教学要点：
    - 这是传统向量检索无法完成的任务，因为相关文档可能在语义上不相似。
    - 例如 "爱因斯坦和波尔的关系" — 需要经过 "量子力学争论" 这个中间节点。
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        chunk_map: dict[str, RAGChunk] | None = None,
        max_depth: int = 3,
    ) -> None:
        super().__init__(graph, chunk_map)
        self.max_depth = max_depth
        logger.info("retriever.path_init", max_depth=max_depth)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        entity_ids = self._extract_entities_from_query(query_text)
        if len(entity_ids) < 2:
            logger.warning("retriever.path_need_two_entities", query=query_text[:50])
            return []

        # 尝试找到前两个实体之间的路径
        start, end = entity_ids[0], entity_ids[1]
        paths = self.graph.find_paths(start, end, max_depth=self.max_depth)

        if not paths:
            logger.info("retriever.path_no_paths", start=start, end=end)
            return []

        # 收集路径上的所有节点
        path_nodes: set[str] = set()
        for path in paths:
            for edge in path:
                path_nodes.add(edge.source)
                path_nodes.add(edge.target)

        chunks: list[RAGChunk] = []
        for nid in path_nodes:
            if nid in self.chunk_map:
                chunks.append(self.chunk_map[nid])
            elif nid in self.graph.nodes and (
                content := self.graph.nodes[nid].properties.get("content")
            ):
                chunks.append(RAGChunk(chunk_id=f"kg_path_{nid}", doc_id=nid, content=str(content)))

        logger.info(
            "retriever.path_search",
            query=query_text[:50],
            start=start,
            end=end,
            paths_found=len(paths),
            nodes_on_paths=len(path_nodes),
        )
        return self._chunks_to_results(chunks[:top_k])


class HybridGraphRetriever(BaseRetriever):
    """混合检索器：向量检索 + 图检索。

    教学要点：
    - 单独使用向量检索：擅长语义匹配，但无法处理多跳推理。
    - 单独使用图检索：擅长关系推理，但对语义相似度不敏感。
    - 混合检索：取长补短，先用向量检索找到语义相关的文档，
      再用图检索补充这些文档中实体相关的子图信息。

    这是 GraphRAG 系统的核心设计模式之一。
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        graph_retriever: GraphRetriever,
        dense_weight: float = 0.6,
        graph_weight: float = 0.4,
        top_k: int = 5,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.graph_retriever = graph_retriever
        self.dense_weight = dense_weight
        self.graph_weight = graph_weight
        self.top_k = top_k
        logger.info(
            "retriever.hybrid_graph_init",
            dense_weight=dense_weight,
            graph_weight=graph_weight,
        )

    def index_documents(self, documents: list[Any]) -> None:
        """同时索引到向量存储和知识图谱。"""
        self.dense_retriever.index_documents(documents)
        self.graph_retriever.index_documents(documents)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[RetrievalResult]:
        """混合检索流程：
        1. 向量检索获取 top_k*2 候选。
        2. 图检索获取相关子图/路径。
        3. 合并去重，按加权得分排序。
        """
        k = max(top_k, self.top_k)

        # Step 1: Dense retrieval
        dense_results = self.dense_retriever.retrieve(query_text, top_k=k * 2)
        dense_scores: dict[str, float] = {
            r.chunk.chunk_id: self.dense_weight * r.score for r in dense_results
        }

        # Step 2: Graph retrieval
        graph_results = self.graph_retriever.retrieve(query_text, top_k=k * 2)
        graph_scores: dict[str, float] = {
            r.chunk.chunk_id: self.graph_weight * r.score for r in graph_results
        }

        # Step 3: Merge
        all_ids = set(dense_scores.keys()) | set(graph_scores.keys())
        merged_scores: dict[str, float] = {}
        for cid in all_ids:
            merged_scores[cid] = dense_scores.get(cid, 0.0) + graph_scores.get(cid, 0.0)

        # Sort and build results
        sorted_ids = sorted(merged_scores.keys(), key=lambda x: merged_scores[x], reverse=True)
        chunk_map = {r.chunk.chunk_id: r.chunk for r in dense_results + graph_results}

        results: list[RetrievalResult] = []
        for rank, cid in enumerate(sorted_ids[:k], 1):
            if cid in chunk_map:
                results.append(
                    RetrievalResult(chunk=chunk_map[cid], score=merged_scores[cid], rank=rank)
                )

        logger.info(
            "retriever.hybrid_graph_search",
            query=query_text[:50],
            dense_results=len(dense_results),
            graph_results=len(graph_results),
            merged=len(results),
        )
        return results
