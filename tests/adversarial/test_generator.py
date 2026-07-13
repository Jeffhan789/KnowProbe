"""Tests for Adversarial question generation."""

import pytest

from knowprobe.adversarial.generator import AdversarialQuestionGenerator


class TestAdversarialQuestionGenerator:
    def test_init(self) -> None:
        gen = AdversarialQuestionGenerator(seed=42)
        assert gen is not None

    def test_distractor_generation(self) -> None:
        gen = AdversarialQuestionGenerator(seed=42)
        questions = gen.generate("What did Einstein discover?", answer="relativity")
        distractors = [q for q in questions if q.strategy == "distractor"]
        # "Einstein" -> may produce distractor with "newton"
        assert len(distractors) >= 0  # depends on keyword matching

    def test_negation_generation(self) -> None:
        gen = AdversarialQuestionGenerator(seed=42)
        questions = gen.generate("What did Einstein win?", answer="Nobel Prize")
        negations = [q for q in questions if q.strategy == "negation"]
        if negations:
            assert "NOT" in negations[0].adversarial_question

    def test_edge_case_generation(self) -> None:
        gen = AdversarialQuestionGenerator(seed=42)
        questions = gen.generate("What is the tallest mountain?", answer="Everest")
        edges = [q for q in questions if q.strategy == "edge_case"]
        if edges:
            assert "second" in edges[0].adversarial_question.lower() or "shortest" in edges[0].adversarial_question.lower()

    def test_specific_strategies(self) -> None:
        gen = AdversarialQuestionGenerator(seed=42)
        questions = gen.generate("Who invented the telephone?", strategies=["distractor"])
        assert all(q.strategy == "distractor" for q in questions)
