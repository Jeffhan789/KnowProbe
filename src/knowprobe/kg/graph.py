"""Knowledge Graph data models and structures.

知识图谱 (Knowledge Graph, KG) 是 GraphRAG 的核心数据结构。

教学要点：
- 传统 RAG 将文档切分为孤立段落，丢失了实体间的关联关系。
- 知识图谱用 (实体-关系-实体) 三元组显式建模这些关系，使模型能进行多跳推理。
- 例如："爱因斯坦→获得→诺贝尔奖"，"诺贝尔奖→颁发于→瑞典" —— 图谱让模型发现这两个事实之间的路径。

参考：Microsoft GraphRAG (Edge et al., 2024) 使用 LLM 从文本提取实体关系，
构建图谱后通过社区摘要 (Community Summaries) 回答全局查询。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KGNode(BaseModel):
    """知识图谱中的节点（实体）。

    每个节点代表一个实体，如人物、地点、组织、概念等。
    节点可以带有属性，丰富语义信息。

    Attributes:
        id: 全局唯一标识符（通常由 label + type 生成，确保同一实体不重复）。
        label: 实体名称，如 "Albert Einstein"。
        type: 实体类型，如 "Person", "Organization", "Concept"。
        properties: 附加属性，如 {"born": "1879", "nationality": "German"}。
    """

    id: str = Field(description="Unique node identifier (e.g., 'Albert_Einstein_Person')")
    label: str = Field(description="Human-readable entity label")
    type: str = Field(default="Entity", description="Entity type: Person, Organization, Concept, etc.")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional entity properties")

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KGNode):
            return NotImplemented
        return self.id == other.id


class KGEdge(BaseModel):
    """知识图谱中的边（关系）。

    边连接两个实体，描述它们之间的关系。
    边的权重表示关系强度或置信度（0.0-1.0）。

    Attributes:
        source: 源节点 ID。
        target: 目标节点 ID。
        relation: 关系类型，如 "born_in", "works_at", "invented"。
        weight: 关系权重（0.0-1.0），由 LLM 置信度或共现频率决定。
        evidence: 关系来源的证据文本，用于溯源和调试。
    """

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relation: str = Field(description="Relation type between entities")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence weight (0.0-1.0)")
    evidence: str = Field(default="", description="Source text that supports this relation")

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.relation))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KGEdge):
            return NotImplemented
        return (self.source, self.target, self.relation) == (
            other.source,
            other.target,
            other.relation,
        )


class KnowledgeGraph(BaseModel):
    """知识图谱容器。

    管理节点和边的集合，提供图的基本操作和统计信息。

    教学要点：
    - 图的密度 (density) = 实际边数 / 最大可能边数。密度高表示知识关联紧密。
    - 平均度 (avg_degree) = 2 * 边数 / 节点数。度高的节点是"知识枢纽"。
    - 这些统计指标可用于评估图谱质量：太稀疏的图谱难以支持多跳推理。

    Attributes:
        nodes: 节点字典，key 为 node.id。
        edges: 边列表（允许同一对节点间有多条不同关系）。
        metadata: 图谱元数据，如构建时间、来源文档数、LLM 模型等。
    """

    nodes: dict[str, KGNode] = Field(default_factory=dict, description="Node ID -> KGNode mapping")
    edges: list[KGEdge] = Field(default_factory=list, description="List of all relations")
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "created_at": datetime.utcnow().isoformat(),
            "source": "knowprobe_kg",
        },
        description="Graph construction metadata",
    )

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------
    def add_node(self, node: KGNode) -> None:
        """添加节点（如果已存在则更新属性）。"""
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        """添加边（自动检查源节点和目标节点是否存在）。"""
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not found in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not found in graph")
        self.edges.append(edge)

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[KGNode]:
        """获取邻居节点。

        Args:
            direction: "out" (出边), "in" (入边), "both" (双向)。
        """
        neighbor_ids: set[str] = set()
        for edge in self.edges:
            if direction in ("out", "both") and edge.source == node_id:
                neighbor_ids.add(edge.target)
            if direction in ("in", "both") and edge.target == node_id:
                neighbor_ids.add(edge.source)
        return [self.nodes[nid] for nid in neighbor_ids if nid in self.nodes]

    def get_edges_from(self, node_id: str) -> list[KGEdge]:
        """获取从某节点出发的所有边。"""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[KGEdge]:
        """获取指向某节点的所有边。"""
        return [e for e in self.edges if e.target == node_id]

    # ------------------------------------------------------------------
    # 图统计
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        """图密度 = 实际边数 / 最大可能边数 (n*(n-1) 对于无向图)。"""
        n = self.num_nodes
        if n < 2:
            return 0.0
        max_edges = n * (n - 1)
        return self.num_edges / max_edges

    @property
    def avg_degree(self) -> float:
        """平均度 = 2 * 边数 / 节点数。"""
        if self.num_nodes == 0:
            return 0.0
        return (2 * self.num_edges) / self.num_nodes

    @property
    def relation_types(self) -> set[str]:
        """返回所有关系类型。"""
        return {e.relation for e in self.edges}

    def degree_distribution(self) -> dict[str, int]:
        """返回每个节点的度。"""
        degrees: dict[str, int] = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            degrees[edge.source] = degrees.get(edge.source, 0) + 1
            degrees[edge.target] = degrees.get(edge.target, 0) + 1
        return degrees

    # ------------------------------------------------------------------
    # 路径与遍历（多跳推理的核心）
    # ------------------------------------------------------------------
    def find_paths(
        self, start_id: str, end_id: str, max_depth: int = 3
    ) -> list[list[KGEdge]]:
        """找到两个节点之间的所有路径（深度优先搜索）。

        这是多跳推理的核心算法。
        例如：从 "Albert Einstein" 到 "Sweden" 可能的路径：
        Einstein --(born_in)--> Germany --(capital)--> Berlin --(has_event)--> Nobel Prize --(awarded_in)--> Sweden

        Args:
            start_id: 起始节点 ID。
            end_id: 目标节点 ID。
            max_depth: 最大跳数（避免组合爆炸）。

        Returns:
            所有找到的路径，每条路径是一个 KGEdge 列表。
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        paths: list[list[KGEdge]] = []
        visited: set[str] = set()

        def _dfs(current: str, path: list[KGEdge], depth: int) -> None:
            if depth > max_depth:
                return
            if current == end_id and path:
                paths.append(path.copy())
                return
            for edge in self.get_edges_from(current):
                if edge.target not in visited:
                    visited.add(edge.target)
                    path.append(edge)
                    _dfs(edge.target, path, depth + 1)
                    path.pop()
                    visited.remove(edge.target)

        visited.add(start_id)
        _dfs(start_id, [], 0)
        return paths

    def ego_graph(self, node_id: str, k_hops: int = 2) -> KnowledgeGraph:
        """提取 k-hop ego graph（ ego 网络）。

        Ego graph 是围绕某个中心节点的局部子图，包含该节点 k 跳内的所有节点和边。
        GraphRAG 中的局部检索通常使用 1-hop 或 2-hop ego graph。

        Args:
            node_id: 中心节点 ID。
            k_hops: 跳数半径。

        Returns:
            子图（新的 KnowledgeGraph 实例）。
        """
        if node_id not in self.nodes:
            return KnowledgeGraph()

        included_nodes: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(k_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for edge in self.edges:
                    if edge.source == nid and edge.target not in included_nodes:
                        next_frontier.add(edge.target)
                    if edge.target == nid and edge.source not in included_nodes:
                        next_frontier.add(edge.source)
            included_nodes.update(next_frontier)
            frontier = next_frontier

        sub_edges = [
            e for e in self.edges if e.source in included_nodes and e.target in included_nodes
        ]
        return KnowledgeGraph(
            nodes={nid: self.nodes[nid] for nid in included_nodes},
            edges=sub_edges,
            metadata={**self.metadata, "ego_center": node_id, "k_hops": k_hops},
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_triples(self) -> list[tuple[str, str, str]]:
        """导出为 (subject, relation, object) 三元组列表。"""
        return [(e.source, e.relation, e.target) for e in self.edges]

    def to_networkx(self) -> dict[str, Any]:
        """导出为 NetworkX 兼容的节点边列表格式。"""
        return {
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "edges": [e.model_dump() for e in self.edges],
        }

    def summary(self) -> dict[str, Any]:
        """返回图谱摘要统计。"""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": round(self.density, 4),
            "avg_degree": round(self.avg_degree, 2),
            "relation_types": sorted(self.relation_types),
            "top_degree_nodes": sorted(
                self.degree_distribution().items(), key=lambda x: x[1], reverse=True
            )[:5],
        }
