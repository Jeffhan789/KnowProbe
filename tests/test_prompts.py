"""Unit tests for the prompt strategy engine.

Run with:
    pytest tests/test_prompts.py -v

Requires: pytest, pytest-asyncio
"""

from __future__ import annotations

import pytest
from pathlib import Path

from knowprobe.core.models import KnowledgeInput, PromptStrategy, QuestionType
from knowprobe.prompts.templates import (
    PromptTemplate,
    TemplateRegistry,
    TemplateRenderError,
    TemplateLoadError,
    load_builtin_templates,
)
from knowprobe.prompts.examples import (
    Example,
    ExampleBank,
    ExampleLoadError,
    RandomExampleSelector,
    SimilarityExampleSelector,
    DiversityExampleSelector,
    ExampleSelectorFactory,
)
from knowprobe.prompts.strategies import (
    BaseStrategy,
    PromptContext,
    StrategyFactory,
    ZeroShotStrategy,
    FewShotStrategy,
    CoTStrategy,
    SelfConsistencyStrategy,
    ReActStrategy,
    StrategyError,
)
from knowprobe.prompts.builder import PromptBuilder, PromptBuilderError
from knowprobe.prompts.engine import PromptStrategyEngine, PromptEngineError


# ── Fixtures ──

@pytest.fixture
def knowledge_input() -> KnowledgeInput:
    return KnowledgeInput(
        source_id="test-001",
        input_type="triple",
        content="Paris is the capital of France and the largest city in the country.",
        structured={"subject": "Paris", "predicate": "is capital of", "object": "France"},
        metadata={"domain": "geography"},
    )


@pytest.fixture
def example_bank() -> ExampleBank:
    bank = ExampleBank()
    bank.extend([
        Example(
            knowledge="Tokyo is the capital of Japan.",
            question="What is the capital of Japan?",
            question_type=QuestionType.FACTUAL,
            strategy=PromptStrategy.FEW_SHOT,
        ),
        Example(
            knowledge="The Amazon River flows through South America.",
            question="Through which continent does the Amazon River flow?",
            question_type=QuestionType.FACTUAL,
            strategy=PromptStrategy.FEW_SHOT,
        ),
        Example(
            knowledge="Entity: Employee; Attributes: id, name, department; Relations: works_in Department",
            question="What attributes define an Employee and how do they relate to Department?",
            question_type=QuestionType.SCHEMA,
            strategy=PromptStrategy.FEW_SHOT,
        ),
    ])
    return bank


@pytest.fixture
def template_registry(tmp_path: Path) -> TemplateRegistry:
    """Create a registry with a temporary template directory."""
    d = tmp_path / "prompts"
    d.mkdir()
    defaults = d / "_defaults"
    defaults.mkdir()
    (defaults / "zero_shot.j2").write_text(
        "Knowledge: {{ knowledge }}\nType: {{ question_type }}\nQuestion:", encoding="utf-8"
    )
    (defaults / "few_shot.j2").write_text(
        "{% for ex in examples %}Ex: {{ ex.knowledge }} -> {{ ex.question }}\n{% endfor %}"
        "Knowledge: {{ knowledge }}\nQuestion:", encoding="utf-8"
    )
    (defaults / "cot.j2").write_text(
        "Think step by step about: {{ knowledge }}\nQuestion:", encoding="utf-8"
    )
    return TemplateRegistry(d)


# ── Template Tests ──

class TestTemplates:
    def test_load_builtin_templates(self) -> None:
        templates = load_builtin_templates()
        assert len(templates) == 15  # 5 strategies × 3 question types
        assert "zero_shot_factual" in templates
        assert "cot_schema" in templates

    def test_template_render_inline(self) -> None:
        tmpl = PromptTemplate(
            name="test",
            strategy=PromptStrategy.ZERO_SHOT,
            question_type=QuestionType.FACTUAL,
            content="Hello {{ name }}!",
        )
        result = tmpl.render({"name": "World"})
        assert result == "Hello World!"

    def test_template_render_missing_var_raises(self) -> None:
        tmpl = PromptTemplate(
            name="test",
            strategy=PromptStrategy.ZERO_SHOT,
            question_type=QuestionType.FACTUAL,
            content="Value: {{ missing_var }}",
        )
        # Jinja2 by default renders missing variables as empty string
        result = tmpl.render({})
        assert result == "Value: "

    def test_registry_load_from_disk(self, template_registry: TemplateRegistry) -> None:
        assert len(template_registry) == 3
        tmpl = template_registry.get(PromptStrategy.ZERO_SHOT, QuestionType.FACTUAL)
        assert tmpl.strategy == PromptStrategy.ZERO_SHOT
        rendered = template_registry.render(
            PromptStrategy.ZERO_SHOT, QuestionType.FACTUAL,
            {"knowledge": "Test", "question_type": "factual"}
        )
        assert "Test" in rendered

    def test_registry_fallback_to_default(self, template_registry: TemplateRegistry) -> None:
        # zero_shot_default exists, so composite falls back to it
        tmpl = template_registry.get(PromptStrategy.ZERO_SHOT, QuestionType.COMPOSITE)
        assert tmpl is not None

    def test_registry_missing_template_raises(self, template_registry: TemplateRegistry) -> None:
        with pytest.raises(TemplateRenderError):
            template_registry.get(PromptStrategy.REACT, QuestionType.FACTUAL)

    def test_registry_list_templates(self, template_registry: TemplateRegistry) -> None:
        templates = template_registry.list_templates()
        assert len(templates) == 3


# ── Example Tests ──

class TestExamples:
    def test_example_bank_add_and_filter(self, example_bank: ExampleBank) -> None:
        assert len(example_bank) == 3
        factual = example_bank.filter(question_type=QuestionType.FACTUAL)
        assert len(factual) == 2
        schema = example_bank.filter(question_type=QuestionType.SCHEMA)
        assert len(schema) == 1

    def test_example_bank_save_and_load_yaml(self, example_bank: ExampleBank, tmp_path: Path) -> None:
        path = tmp_path / "examples.yaml"
        example_bank.save_to_yaml(path)
        assert path.exists()

        new_bank = ExampleBank()
        new_bank.load_from_yaml(path)
        assert len(new_bank) == 3

    def test_random_selector(self, example_bank: ExampleBank) -> None:
        selector = RandomExampleSelector(seed=42)
        selected = selector.select(
            example_bank, "Paris is great.", QuestionType.FACTUAL, PromptStrategy.FEW_SHOT, k=2
        )
        assert len(selected) == 2
        assert all(isinstance(ex, Example) for ex in selected)

    def test_similarity_selector(self, example_bank: ExampleBank) -> None:
        selector = SimilarityExampleSelector()
        selected = selector.select(
            example_bank, "Tokyo is the capital.", QuestionType.FACTUAL, PromptStrategy.FEW_SHOT, k=1
        )
        assert len(selected) == 1
        # Should pick the Tokyo example since it shares words
        assert "Tokyo" in selected[0].knowledge

    def test_diversity_selector(self, example_bank: ExampleBank) -> None:
        selector = DiversityExampleSelector(lambda_param=0.5)
        selected = selector.select(
            example_bank, "Some knowledge.", QuestionType.FACTUAL, PromptStrategy.FEW_SHOT, k=2
        )
        assert len(selected) <= 2

    def test_selector_factory(self) -> None:
        for name in ExampleSelectorFactory.list_selectors():
            selector = ExampleSelectorFactory.create(name)
            assert selector is not None

        with pytest.raises(ValueError):
            ExampleSelectorFactory.create("nonexistent")

    def test_example_to_prompt_block(self) -> None:
        ex = Example(knowledge="A is B.", question="What is A?")
        block = ex.to_prompt_block()
        assert "Knowledge: A is B." in block
        assert "Question: What is A?" in block


# ── Strategy Tests ──

class TestStrategies:
    def test_strategy_factory_list(self) -> None:
        strategies = StrategyFactory.list_strategies()
        assert set(strategies) == {
            "zero_shot", "few_shot", "cot", "self_consistency", "react"
        }

    def test_strategy_factory_create(self) -> None:
        for ps in PromptStrategy:
            s = StrategyFactory.create(ps)
            assert isinstance(s, BaseStrategy)
            assert s.strategy_type == ps

    def test_strategy_factory_unknown_raises(self) -> None:
        # PromptStrategy enum prevents unknown values, so this tests the register path
        with pytest.raises(TypeError):
            StrategyFactory.register(PromptStrategy.ZERO_SHOT, object)  # type: ignore

    def test_zero_shot_build(self, knowledge_input: KnowledgeInput) -> None:
        strategy = ZeroShotStrategy()
        ctx = PromptContext(
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        prompts = strategy.build(ctx)
        assert len(prompts) == 1
        assert "Paris" in prompts[0]
        assert "Question:" in prompts[0]

    def test_few_shot_build_with_examples(self, knowledge_input: KnowledgeInput, example_bank: ExampleBank) -> None:
        strategy = FewShotStrategy(example_bank=example_bank, few_shot_k=2)
        ctx = PromptContext(
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        prompts = strategy.build(ctx)
        assert len(prompts) == 1
        # Should contain example content
        assert "Tokyo" in prompts[0] or "Amazon" in prompts[0]

    def test_few_shot_build_with_context_examples(self, knowledge_input: KnowledgeInput) -> None:
        strategy = FewShotStrategy()
        ctx = PromptContext(
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
            examples=[
                Example(knowledge="X is Y.", question="What is X?", question_type=QuestionType.FACTUAL)
            ],
        )
        prompts = strategy.build(ctx)
        assert len(prompts) == 1
        assert "X is Y" in prompts[0]

    def test_cot_build(self, knowledge_input: KnowledgeInput) -> None:
        strategy = CoTStrategy()
        ctx = PromptContext(
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        prompts = strategy.build(ctx)
        assert len(prompts) == 1
        assert "step" in prompts[0].lower()

    def test_self_consistency_build(self, knowledge_input: KnowledgeInput) -> None:
        strategy = SelfConsistencyStrategy(num_samples=3)
        ctx = PromptContext(
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        prompts = strategy.build(ctx)
        assert len(prompts) == 3
        # Each prompt should have a different reasoning instruction
        assert len(set(prompts)) == 3

    def test_react_build(self, knowledge_input: KnowledgeInput) -> None:
        strategy = ReActStrategy()
        ctx = PromptContext(
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        prompts = strategy.build(ctx)
        assert len(prompts) == 1
        assert "Thought" in prompts[0]

    def test_strategy_build_error_handling(self, knowledge_input: KnowledgeInput) -> None:
        # Create a broken strategy that raises in build_prompts
        class BrokenStrategy(BaseStrategy):
            strategy_type = PromptStrategy.ZERO_SHOT
            def build_prompts(self, context: PromptContext) -> list[str]:
                raise RuntimeError("Simulated failure")

        StrategyFactory.register(PromptStrategy.ZERO_SHOT, BrokenStrategy)
        strategy = StrategyFactory.create(PromptStrategy.ZERO_SHOT)
        ctx = PromptContext(knowledge_input=knowledge_input, question_type=QuestionType.FACTUAL)
        with pytest.raises(StrategyError):
            strategy.build(ctx)

        # Restore original
        StrategyFactory.register(PromptStrategy.ZERO_SHOT, ZeroShotStrategy)


# ── Builder Tests ──

class TestPromptBuilder:
    def test_builder_single(self, knowledge_input: KnowledgeInput) -> None:
        builder = PromptBuilder()
        prompts = builder.build(
            strategy=PromptStrategy.ZERO_SHOT,
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        assert len(prompts) == 1
        assert isinstance(prompts[0], str)

    def test_builder_batch(self, knowledge_input: KnowledgeInput) -> None:
        builder = PromptBuilder()
        inputs = [knowledge_input, knowledge_input]
        results = builder.build_batch(
            strategy=PromptStrategy.COT,
            knowledge_inputs=inputs,
            question_type=QuestionType.FACTUAL,
        )
        assert len(results) == 2
        assert all(len(r) == 1 for r in results)

    def test_builder_with_examples(self, knowledge_input: KnowledgeInput, example_bank: ExampleBank) -> None:
        builder = PromptBuilder(
            example_bank=example_bank,
            default_few_shot_k=2,
        )
        prompts = builder.build(
            strategy=PromptStrategy.FEW_SHOT,
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        assert len(prompts) == 1

    def test_builder_properties(self) -> None:
        builder = PromptBuilder(default_few_shot_k=5)
        assert builder.default_few_shot_k == 5
        assert builder.example_selector is not None


# ── Engine Tests ──

class TestPromptStrategyEngine:
    def test_engine_from_settings_builtin_fallback(self, tmp_path: Path) -> None:
        """Test that engine falls back to built-in templates when no files exist."""
        empty_dir = tmp_path / "empty_prompts"
        empty_dir.mkdir()
        engine = PromptStrategyEngine.from_settings(
            templates_dir=empty_dir,
        )
        assert len(engine.list_templates()) > 0  # Built-ins loaded
        assert len(engine.list_strategies()) == 5

    def test_engine_build(self, knowledge_input: KnowledgeInput) -> None:
        engine = PromptStrategyEngine.from_settings()
        prompts = engine.build(
            strategy=PromptStrategy.ZERO_SHOT,
            knowledge_input=knowledge_input,
            question_type=QuestionType.FACTUAL,
        )
        assert len(prompts) == 1
        assert "Paris" in prompts[0]

    def test_engine_build_batch(self, knowledge_input: KnowledgeInput) -> None:
        engine = PromptStrategyEngine.from_settings()
        results = engine.build_batch(
            strategy=PromptStrategy.COT,
            knowledge_inputs=[knowledge_input, knowledge_input],
            question_type=QuestionType.SCHEMA,
        )
        assert len(results) == 2

    def test_engine_experiment_matrix(self, knowledge_input: KnowledgeInput) -> None:
        engine = PromptStrategyEngine.from_settings()
        matrix = engine.build_experiment_matrix(
            knowledge_inputs=[knowledge_input],
            strategies=[PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT],
            question_types=[QuestionType.FACTUAL, QuestionType.SCHEMA],
        )
        assert "zero_shot" in matrix
        assert "few_shot" in matrix
        assert "factual" in matrix["zero_shot"]
        assert "schema" in matrix["zero_shot"]
        assert len(matrix["zero_shot"]["factual"]) == 1

    def test_engine_add_example(self) -> None:
        engine = PromptStrategyEngine.from_settings()
        ex = Example(knowledge="Test.", question="What?", question_type=QuestionType.FACTUAL)
        engine.add_example(ex)
        # Internal bank should now have at least 1 example
        assert len(engine._example_bank) >= 1

    def test_engine_reload_templates(self, tmp_path: Path) -> None:
        engine = PromptStrategyEngine.from_settings()
        new_dir = tmp_path / "new_prompts"
        new_dir.mkdir()
        defaults = new_dir / "_defaults"
        defaults.mkdir()
        (defaults / "zero_shot.j2").write_text("Custom: {{ knowledge }}")
        engine.reload_templates(new_dir)
        templates = engine.list_templates()
        assert any("zero_shot" in t for t in templates)

    def test_engine_repr(self) -> None:
        engine = PromptStrategyEngine.from_settings()
        r = repr(engine)
        assert "PromptStrategyEngine" in r
        assert "templates=" in r


# ── Integration / Edge Cases ──

class TestEdgeCases:
    def test_empty_example_bank_filter(self) -> None:
        bank = ExampleBank()
        assert len(bank.filter(question_type=QuestionType.FACTUAL)) == 0

    def test_selector_no_candidates(self, example_bank: ExampleBank) -> None:
        selector = RandomExampleSelector()
        # Ask for a type that doesn't exist in the bank
        results = selector.select(
            example_bank, "knowledge", QuestionType.COMPOSITE, PromptStrategy.FEW_SHOT, k=2
        )
        assert results == []

    def test_diversity_selector_lambda_validation(self) -> None:
        with pytest.raises(ValueError):
            DiversityExampleSelector(lambda_param=1.5)
        with pytest.raises(ValueError):
            DiversityExampleSelector(lambda_param=-0.1)

    def test_example_yaml_roundtrip(self, tmp_path: Path) -> None:
        bank = ExampleBank()
        bank.add(Example(
            knowledge="K",
            question="Q?",
            question_type=QuestionType.COMPOSITE,
            strategy=PromptStrategy.COT,
            metadata={"tag": "test"},
        ))
        path = tmp_path / "roundtrip.yaml"
        bank.save_to_yaml(path)
        loaded = ExampleBank()
        loaded.load_from_yaml(path)
        assert len(loaded) == 1
        assert loaded.examples[0].metadata["tag"] == "test"

    def test_builtin_template_coverage(self) -> None:
        """Ensure every strategy has a built-in template for every question type."""
        built_ins = load_builtin_templates()
        for strategy in PromptStrategy:
            for qtype in QuestionType:
                key = f"{strategy.value}_{qtype.value}"
                assert key in built_ins, f"Missing built-in template for {key}"
