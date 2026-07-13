"""Tests for Knowledge Graph module.

教学要点：这些测试展示了知识图谱的基本操作和验证方法。
学习者可以通过修改测试数据来理解图谱的各种属性。
"""

import pytest

from knowprobe.kg.graph import KGEdge, KGNode, KnowledgeGraph


class TestKGNode:
    def test_basic_node(self) -> None:
        node = KGNode(id="einstein_person", label="Albert Einstein", type="Person")
        assert node.id == "einstein_person"
        assert node.label == "Albert Einstein"
        assert node.type == "Person"

    def test_node_hash(self) -> None:
        n1 = KGNode(id="a", label="A", type="T")
        n2 = KGNode(id="a", label="A", type="T")
        assert n1 == n2
        assert hash(n1) == hash(n2)


class TestKGEdge:
    def test_basic_edge(self) -> None:
        edge = KGEdge(source="a", target="b", relation="invented", weight=0.9)
        assert edge.source == "a"
        assert edge.relation == "invented"
        assert edge.weight == 0.9

    def test_edge_invalid_weight(self) -> None:
        with pytest.raises(Exception):  # pydantic validation
            KGEdge(source="a", target="b", relation="r", weight=1.5)


class TestKnowledgeGraph:
    def test_empty_graph(self) -> None:
        g = KnowledgeGraph()
        assert g.num_nodes == 0
        assert g.num_edges == 0
        assert g.density == 0.0

    def test_add_nodes_and_edges(self) -> None:
        g = KnowledgeGraph()
        g.add_node(KGNode(id="a", label="A"))
        g.add_node(KGNode(id="b", label="B"))
        g.add_edge(KGEdge(source="a", target="b", relation="link"))
        assert g.num_nodes == 2
        assert g.num_edges == 1
        assert g.density == 0.5  # 1 / (2*1)

    def test_add_edge_missing_node_raises(self) -> None:
        g = KnowledgeGraph()
        g.add_node(KGNode(id="a", label="A"))
        with pytest.raises(ValueError, match="Target node"):
            g.add_edge(KGEdge(source="a", target="b", relation="link"))

    def test_neighbors(self) -> None:
        g = KnowledgeGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(KGNode(id=nid, label=nid.upper()))
        g.add_edge(KGEdge(source="a", target="b", relation="r1"))
        g.add_edge(KGEdge(source="a", target="c", relation="r2"))
        neighbors = g.get_neighbors("a")
        assert len(neighbors) == 2

    def test_find_paths(self) -> None:
        g = KnowledgeGraph()
        for nid in ["a", "b", "c", "d"]:
            g.add_node(KGNode(id=nid, label=nid.upper()))
        g.add_edge(KGEdge(source="a", target="b", relation="r1"))
        g.add_edge(KGEdge(source="b", target="c", relation="r2"))
        g.add_edge(KGEdge(source="c", target="d", relation="r3"))

        paths = g.find_paths("a", "d", max_depth=3)
        assert len(paths) == 1
        assert len(paths[0]) == 3  # a->b, b->c, c->d

    def test_find_paths_no_path(self) -> None:
        g = KnowledgeGraph()
        for nid in ["a", "b"]:
            g.add_node(KGNode(id=nid, label=nid.upper()))
        paths = g.find_paths("a", "b", max_depth=3)
        assert len(paths) == 0

    def test_ego_graph(self) -> None:
        g = KnowledgeGraph()
        for nid in ["center", "n1", "n2", "far"]:
            g.add_node(KGNode(id=nid, label=nid))
        g.add_edge(KGEdge(source="center", target="n1", relation="r"))
        g.add_edge(KGEdge(source="n1", target="n2", relation="r"))
        g.add_edge(KGEdge(source="n2", target="far", relation="r"))

        ego = g.ego_graph("center", k_hops=2)
        assert "center" in ego.nodes
        assert "n1" in ego.nodes
        assert "n2" in ego.nodes
        assert "far" not in ego.nodes  # 3 hops away

    def test_summary(self) -> None:
        g = KnowledgeGraph()
        g.add_node(KGNode(id="a", label="A"))
        g.add_node(KGNode(id="b", label="B"))
        g.add_edge(KGEdge(source="a", target="b", relation="link"))
        s = g.summary()
        assert s["num_nodes"] == 2
        assert s["num_edges"] == 1
        assert s["relation_types"] == ["link"]


class TestGraphStatistics:
    def test_avg_degree(self) -> None:
        g = KnowledgeGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(KGNode(id=nid, label=nid))
        g.add_edge(KGEdge(source="a", target="b", relation="r"))
        g.add_edge(KGEdge(source="b", target="c", relation="r"))
        # 2 edges, 3 nodes -> avg_degree = 4/3 ≈ 1.33
        assert round(g.avg_degree, 2) == 1.33

    def test_degree_distribution(self) -> None:
        g = KnowledgeGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(KGNode(id=nid, label=nid))
        g.add_edge(KGEdge(source="a", target="b", relation="r"))
        g.add_edge(KGEdge(source="a", target="c", relation="r"))
        dist = g.degree_distribution()
        assert dist["a"] == 2
        assert dist["b"] == 1
        assert dist["c"] == 1
