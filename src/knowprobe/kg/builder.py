"""Knowledge Graph Builder — 从文本或三元组构建知识图谱。

教学要点：
- 构建知识图谱有两种主要方式：
  1. 基于 LLM 的提取：用 prompt 让 LLM 从文本中识别实体和关系。精度高但成本高。
  2. 基于规则的提取：用正则表达式、词典匹配等。成本低但覆盖率低。
- 本模块实现了两种策略，供学习者对比：
  - RuleBasedBuilder：零成本，适合教学演示。
  - LLMBasedBuilder：需要 LLM 后端，但精度更高。

工程要点：
- 使用 Builder 模式，将复杂的图谱构建过程封装为可配置、可扩展的流水线。
- 每一步都有日志记录，便于调试和审计。
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from knowprobe.kg.graph import KGEdge, KGNode, KnowledgeGraph
from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """知识图谱构建器的抽象基类。

    所有具体构建器都继承此类，实现统一的 build_from_text 接口。
    """

    def __init__(self, name: str = "builder") -> None:
        self.name = name
        self._logger = get_logger(f"{__name__}.{name}")

    def build_from_text(self, text: str, source_id: str = "") -> KnowledgeGraph:
        """从文本构建知识图谱。子类必须实现。"""
        raise NotImplementedError

    def build_from_triples(
        self, triples: list[tuple[str, str, str]], source_id: str = ""
    ) -> KnowledgeGraph:
        """从 (subject, relation, object) 三元组直接构建图谱。

        这是最确定性的构建方式，没有 LLM 提取的不确定性。
        适合处理结构化数据源（如 WikiData、FreeBase）。
        """
        graph = KnowledgeGraph(
            metadata={
                "source_id": source_id or f"triples_{uuid.uuid4().hex[:6]}",
                "builder": self.name,
                "method": "direct_triples",
            }
        )
        for subj, rel, obj in triples:
            # 为每个实体创建唯一 ID（避免同名不同类型实体冲突）
            subj_id = self._make_id(subj)
            obj_id = self._make_id(obj)

            graph.add_node(KGNode(id=subj_id, label=subj, type="Entity"))
            graph.add_node(KGNode(id=obj_id, label=obj, type="Entity"))
            graph.add_edge(
                KGEdge(source=subj_id, relation=rel, target=obj_id, evidence=f"{subj} {rel} {obj}")
            )

        self._logger.info(
            "kg.built_from_triples",
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            source=source_id,
        )
        return graph

    @staticmethod
    def _make_id(label: str) -> str:
        """将标签转换为安全 ID（去除空格，统一小写）。"""
        return re.sub(r"\W+", "_", label.strip()).lower()


class RuleBasedBuilder(GraphBuilder):
    """基于规则的知识图谱构建器。

    教学演示用。使用简单的正则规则和命名实体识别模式提取三元组。
    优点：零成本、可解释、确定性。
    缺点：覆盖率低，只能处理非常规范的文本。

    适用场景：
    - 教学演示，展示图谱构建流程。
    - 处理高度结构化或半结构化的文本（如 Wikipedia 信息框）。
    """

    # 预定义的关系模式：正则表达式 -> 关系名
    DEFAULT_PATTERNS: list[tuple[str, str]] = [
        # "A is the capital of B"
        (r"([A-Z][a-zA-Z\s]+)\s+is\s+(?:the\s+)?([a-z]+)\s+of\s+([A-Z][a-zA-Z\s]+)", "{1}_of"),
        # "A was born in B"
        (r"([A-Z][a-zA-Z\s]+)\s+was\s+born\s+in\s+([A-Z][a-zA-Z\s]+)", "born_in"),
        # "A invented B"
        (r"([A-Z][a-zA-Z\s]+)\s+invented\s+([A-Z][a-zA-Z\s]+)", "invented"),
        # "A discovered B"
        (r"([A-Z][a-zA-Z\s]+)\s+discovered\s+([A-Z][a-zA-Z\s]+)", "discovered"),
        # "A works at B"
        (r"([A-Z][a-zA-Z\s]+)\s+works\s+(?:at|for)\s+([A-Z][a-zA-Z\s]+)", "works_at"),
        # "A won the B"
        (r"([A-Z][a-zA-Z\s]+)\s+won\s+(?:the\s+)?([A-Z][a-zA-Z\s]+)", "won"),
    ]

    def __init__(self, patterns: list[tuple[str, str]] | None = None) -> None:
        super().__init__(name="rule_based")
        self.patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self._logger.info("kg.rule_builder_init", patterns=len(self.patterns))

    def build_from_text(self, text: str, source_id: str = "") -> KnowledgeGraph:
        """从文本中提取三元组并构建图谱。

        处理流程：
        1. 将文本按句子切分。
        2. 对每个句子匹配预定义的正则模式。
        3. 将匹配结果转换为 (subject, relation, object) 三元组。
        4. 用三元组构建图谱。
        """
        sentences = self._split_sentences(text)
        triples: list[tuple[str, str, str]] = []

        for sentence in sentences:
            for pattern, rel_template in self.patterns:
                match = re.search(pattern, sentence)
                if match:
                    groups = match.groups()
                    # 根据模板确定关系名
                    if "{1}" in rel_template:
                        rel = rel_template.format(groups[1].strip().lower())
                    else:
                        rel = rel_template
                    subj = groups[0].strip()
                    obj = groups[1].strip() if len(groups) == 2 else groups[2].strip()
                    triples.append((subj, rel, obj))
                    self._logger.debug("kg.rule_match", sentence=sentence[:60], triple=(subj, rel, obj))
                    break  # 一个句子只匹配一个模式

        graph = self.build_from_triples(triples, source_id=source_id or "rule_extraction")
        graph.metadata["sentences_parsed"] = len(sentences)
        graph.metadata["triples_extracted"] = len(triples)
        return graph

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """简单句子切分。"""
        return [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]


class LLMBasedBuilder(GraphBuilder):
    """基于 LLM 的知识图谱构建器。

    使用 LLM 从非结构化文本中提取实体和关系。
    这是生产环境推荐的方式，但需要 LLM 后端支持。

    教学要点：
    - LLM 提取的优势：能处理复杂句式、隐含关系。
    - LLM 提取的局限：成本高、可能产生幻觉、需要后处理去重。
    - 实践中常结合两者：先用 LLM 提取，再用规则过滤和验证。

    本实现使用 prompt 模板让 LLM 输出结构化的 JSON 格式三元组。
    """

    EXTRACTION_PROMPT = """Extract all entity-relation-entity triples from the following text.
Output ONLY a JSON array of objects with keys: "subject", "relation", "object".
Be concise. Include only explicit facts stated in the text.

Text:
{text}

JSON:"""

    def __init__(self, llm_client: Any | None = None) -> None:
        """
        Args:
            llm_client: 符合 knowprobe.llm 接口的 LLM 客户端。
                        如果为 None，则只能使用 build_from_triples。
        """
        super().__init__(name="llm_based")
        self.llm_client = llm_client

    def build_from_text(self, text: str, source_id: str = "") -> KnowledgeGraph:
        """使用 LLM 从文本提取三元组并构建图谱。

        流程：
        1. 构造 extraction prompt。
        2. 调用 LLM 生成 JSON 格式的三元组列表。
        3. 解析 JSON，转换为三元组。
        4. 用 build_from_triples 构建图谱。
        """
        if self.llm_client is None:
            raise RuntimeError(
                "LLMBasedBuilder requires an LLM client. "
                "Pass llm_client to __init__ or use RuleBasedBuilder instead."
            )

        from knowprobe.llm.types import GenerationRequest

        prompt = self.EXTRACTION_PROMPT.format(text=text[:4000])  # 限制长度
        request = GenerationRequest(prompt=prompt, model="")
        response = self.llm_client.generate(request)
        raw_text = response.text.strip()

        # 解析 JSON 三元组
        triples = self._parse_llm_output(raw_text)
        graph = self.build_from_triples(triples, source_id=source_id or "llm_extraction")
        graph.metadata["llm_model"] = getattr(response, "model", "unknown")
        graph.metadata["raw_output_length"] = len(raw_text)
        return graph

    @staticmethod
    def _parse_llm_output(text: str) -> list[tuple[str, str, str]]:
        """解析 LLM 输出的 JSON 三元组。

        容错处理：
        - 提取 JSON 数组（即使被 markdown 代码块包裹）。
        - 如果解析失败，返回空列表（不 panic）。
        """
        import json

        # 尝试从 markdown 代码块中提取 JSON
        code_block = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
        if code_block:
            text = code_block.group(1)
        else:
            # 尝试直接提取 JSON 数组
            arr_match = re.search(r"(\[.*?\])", text, re.DOTALL)
            if arr_match:
                text = arr_match.group(1)

        try:
            data = json.loads(text)
            if not isinstance(data, list):
                logger.warning("kg.llm_parse_not_list", type=type(data).__name__)
                return []
            triples = []
            for item in data:
                if isinstance(item, dict) and "subject" in item and "relation" in item and "object" in item:
                    triples.append((item["subject"], item["relation"], item["object"]))
            return triples
        except json.JSONDecodeError as e:
            logger.warning("kg.llm_parse_failed", error=str(e))
            return []
