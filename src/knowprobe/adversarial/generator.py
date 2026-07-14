"""Adversarial question generation and evaluation for RAG robustness testing.

教学要点：
- RAG 系统不是万能的。它们有很多弱点：
  1. 对否定敏感（"爱因斯坦没有获得什么奖？" -> 检索到 "爱因斯坦获得诺贝尔奖"）。
  2. 对相似实体混淆（"牛顿" vs "牛顿第二定律" vs "牛顿公司"）。
  3. 对多跳变体不鲁棒（改变推理路径）。
  4. 对边界情况处理差（"世界上最高的山" vs "世界上第二高的山"）。
- 对抗性测试不是"攻击"，而是系统性地发现系统弱点，帮助改进。

参考：
- PoisonedRAG (Zou et al., 2025): 知识注入攻击，测试 RAG 对恶意信息的脆弱性。
- SafeRAG (Liang et al., 2025): 系统评估 RAG 对噪声、冲突、对抗输入的鲁棒性。

本模块实现四种对抗性生成策略：
1. Distractor: 在相似实体中构造混淆。
2. Negation: 否定形式，测试语义理解。
3. MultiHopVariation: 改变多跳路径，测试推理鲁棒性。
4. EdgeCase: 边界情况（超级lative变体、时间范围等）。
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, Field

from knowprobe.utils.logging import get_logger

logger = get_logger(__name__)


class AdversarialQuestion(BaseModel):
    """对抗性问题数据模型。"""

    original_question: str = Field(description="Original benign question")
    adversarial_question: str = Field(description="Modified adversarial question")
    strategy: str = Field(description="Which strategy was used to generate this")
    expected_behavior: str = Field(
        default="",
        description="What the RAG system should ideally do (e.g., 'reject', 'answer correctly')",
    )
    target_weakness: str = Field(
        default="",
        description="Which RAG weakness this targets (e.g., 'negation_handling')",
    )


class AdversarialQuestionGenerator:
    """对抗性问题生成器。

    输入一个正常问题，输出多种对抗性变体，用于测试 RAG 系统的鲁棒性。

    使用方式：
    1. 准备一组正常问答对（种子数据）。
    2. 用本生成器创建对抗性变体。
    3. 运行 RAG 系统回答这些对抗性问题。
    4. 分析哪些变体让系统失败，定位弱点。
    """

    STRATEGIES = ["distractor", "negation", "multihop_variation", "edge_case"]

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self._logger = get_logger(__name__)

    def generate(
        self,
        question: str,
        answer: str = "",
        strategies: list[str] | None = None,
    ) -> list[AdversarialQuestion]:
        """为一个问题生成所有对抗性变体。

        Args:
            question: 原始问题。
            answer: 原始答案（部分策略需要）。
            strategies: 要使用的策略列表。默认使用全部。

        Returns:
            对抗性问题列表。
        """
        to_use = strategies or self.STRATEGIES
        results: list[AdversarialQuestion] = []

        if "distractor" in to_use:
            results.extend(self._distractor(question, answer))
        if "negation" in to_use:
            results.extend(self._negation(question, answer))
        if "multihop_variation" in to_use:
            results.extend(self._multihop_variation(question, answer))
        if "edge_case" in to_use:
            results.extend(self._edge_case(question, answer))

        self._logger.info(
            "adversarial.generated",
            original=question[:50],
            variants=len(results),
            strategies=to_use,
        )
        return results

    # ------------------------------------------------------------------
    # 策略 1: Distractor（混淆实体）
    # ------------------------------------------------------------------
    def _distractor(self, question: str, answer: str) -> list[AdversarialQuestion]:
        """在问题中引入相似但错误的实体，测试 RAG 是否能区分。

        示例：
        原始: "Who invented the telephone?"
        对抗: "Did Thomas Edison invent the telephone?" (实际发明者是 Bell)
        """
        variants: list[AdversarialQuestion] = []

        # 简单启发式：如果问题包含人名，替换为另一个名人
        distractor_map = {
            "einstein": "newton",
            "newton": "einstein",
            "tesla": "edison",
            "edison": "tesla",
            "shakespeare": "dickens",
            "dickens": "shakespeare",
        }

        lower_q = question.lower()
        for name, distractor in distractor_map.items():
            if name in lower_q:
                new_q = lower_q.replace(name, distractor)
                variants.append(
                    AdversarialQuestion(
                        original_question=question,
                        adversarial_question=new_q,
                        strategy="distractor",
                        expected_behavior="answer correctly or clarify the confusion",
                        target_weakness="entity_disambiguation",
                    )
                )

        return variants

    # ------------------------------------------------------------------
    # 策略 2: Negation（否定形式）
    # ------------------------------------------------------------------
    def _negation(self, question: str, answer: str) -> list[AdversarialQuestion]:
        """将问题转换为否定形式，测试 RAG 的语义理解。

        示例：
        原始: "What did Einstein win?"
        对抗: "What award did Einstein NOT win?"

        为什么 RAG 容易失败：
        - 向量检索匹配 "Einstein win"，返回 "Einstein won Nobel Prize"。
        - 生成模型没有意识到问题是否定形式，直接回答 "Nobel Prize"。
        - 正确行为应该是回答其他奖项或说明无法确定。
        """
        variants: list[AdversarialQuestion] = []

        # 简单转换：在 what/which/who 后插入 NOT
        if question.lower().startswith(("what", "which", "who")):
            words = question.split()
            if len(words) > 1:
                words.insert(1, "NOT")
                new_q = " ".join(words)
                variants.append(
                    AdversarialQuestion(
                        original_question=question,
                        adversarial_question=new_q,
                        strategy="negation",
                        expected_behavior="answer correctly considering negation or reject",
                        target_weakness="negation_handling",
                    )
                )

        return variants

    # ------------------------------------------------------------------
    # 策略 3: MultiHop Variation（改变多跳路径）
    # ------------------------------------------------------------------
    def _multihop_variation(self, question: str, answer: str) -> list[AdversarialQuestion]:
        """改变多跳问题的推理路径，测试推理鲁棒性。

        示例：
        原始: "Where was the inventor of the telephone born?"
        路径: Telephone -> Bell -> born -> Scotland
        对抗: "Which country did the telephone inventor's wife come from?"
        路径: Telephone -> Bell -> wife -> Mabel -> born -> USA

        为什么测试：RAG 系统可能过度依赖训练数据中的常见路径，
        对路径变化不敏感。
        """
        # 简单启发式：将 "born" 替换为其他关系词
        variants: list[AdversarialQuestion] = []
        if "born" in question.lower():
            for alt in ["die", "study", "work", "live"]:
                new_q = question.lower().replace("born", alt)
                variants.append(
                    AdversarialQuestion(
                        original_question=question,
                        adversarial_question=new_q,
                        strategy="multihop_variation",
                        expected_behavior="answer correctly with new relation",
                        target_weakness="relation_path_robustness",
                    )
                )
        return variants

    # ------------------------------------------------------------------
    # 策略 4: Edge Case（边界情况）
    # ------------------------------------------------------------------
    def _edge_case(self, question: str, answer: str) -> list[AdversarialQuestion]:
        """构造边界情况问题。

        示例：
        原始: "What is the tallest mountain?" -> "Mount Everest"
        对抗: "What is the SECOND tallest mountain?" -> "K2"

        为什么测试：RAG 系统对 Superlative 问题的回答通常是训练数据中的最常见答案，
        对变体（第二、第三、最矮等）处理不佳。
        """
        variants: list[AdversarialQuestion] = []
        lower_q = question.lower()

        if "tallest" in lower_q or "highest" in lower_q or "largest" in lower_q:
            for adj in ["second tallest", "third tallest", "shortest", "smallest"]:
                new_q = (
                    lower_q.replace("tallest", adj).replace("highest", adj).replace("largest", adj)
                )
                variants.append(
                    AdversarialQuestion(
                        original_question=question,
                        adversarial_question=new_q,
                        strategy="edge_case",
                        expected_behavior="answer correctly with modified superlative",
                        target_weakness="superlative_reasoning",
                    )
                )

        if "when" in lower_q:
            variants.append(
                AdversarialQuestion(
                    original_question=question,
                    adversarial_question=lower_q.replace("when", "before what year"),
                    strategy="edge_case",
                    expected_behavior="answer correctly with temporal constraint",
                    target_weakness="temporal_reasoning",
                )
            )

        return variants


class AdversarialEvaluator:
    """对抗性评估器。

    评估 RAG 系统在面对对抗性问题时的表现。

    指标：
    - 攻击成功率 (Attack Success Rate): 对抗性问题导致错误回答的比例。
    - 鲁棒性得分 (Robustness Score): 1 - 攻击成功率。
    - 按策略分组的失败率：帮助定位具体弱点。
    """

    def __init__(self) -> None:
        self._logger = get_logger(__name__)

    def evaluate(
        self,
        adversarial_questions: list[AdversarialQuestion],
        rag_answers: list[str],
        correct_answers: list[str],
    ) -> dict[str, Any]:
        """评估对抗性测试结果。

        Args:
            adversarial_questions: 对抗性问题列表。
            rag_answers: RAG 系统对每个问题的回答。
            correct_answers: 正确答案列表。

        Returns:
            包含总体和按策略分组的评估结果。
        """
        if len(adversarial_questions) != len(rag_answers) != len(correct_answers):
            raise ValueError("All input lists must have the same length")

        total = len(adversarial_questions)
        failures = 0
        strategy_counts: dict[str, dict[str, int]] = {}

        for aq, rag_ans, correct in zip(
            adversarial_questions, rag_answers, correct_answers, strict=False
        ):
            # 简单判断：RAG 答案是否包含正确答案（或相反）
            # 对抗性问题的"正确"行为可能不是简单匹配，而是拒绝或修正
            # 这里用简化规则：如果 RAG 答案包含正确关键词，算成功
            is_correct = correct.lower() in rag_ans.lower() if correct else False
            is_attack_successful = not is_correct

            if is_attack_successful:
                failures += 1

            strategy = aq.strategy
            if strategy not in strategy_counts:
                strategy_counts[strategy] = {"total": 0, "failures": 0}
            strategy_counts[strategy]["total"] += 1
            if is_attack_successful:
                strategy_counts[strategy]["failures"] += 1

        asr = failures / total if total else 0.0
        summary = {
            "total_adversarial_questions": total,
            "attack_successes": failures,
            "attack_success_rate": round(asr, 4),
            "robustness_score": round(1 - asr, 4),
            "by_strategy": {
                s: {
                    "total": c["total"],
                    "failures": c["failures"],
                    "failure_rate": round(c["failures"] / c["total"], 4) if c["total"] else 0.0,
                }
                for s, c in strategy_counts.items()
            },
        }

        self._logger.info(
            "adversarial.evaluation_complete",
            total=total,
            asr=round(asr, 4),
            robustness=round(1 - asr, 4),
        )
        return summary
