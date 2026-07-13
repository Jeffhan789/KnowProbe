"""Question generation CLI commands for KnowProbe."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from knowprobe.core.config import get_settings, load_settings
from knowprobe.core.models import (
    GeneratedQuestion,
    KnowledgeInput,
    ModelProvider,
    PromptStrategy,
    QuestionType,
)
from knowprobe.utils.logging import configure_logging, get_logger

from ..utils import (
    CLIError,
    ConfigurationError,
    GenerationError,
    confirm_action,
    console,
    create_progress_bar,
    format_duration,
    print_header,
    print_info,
    print_success,
    print_warning,
    questions_to_table,
    save_json_output,
    validate_input_file,
)

logger = get_logger("knowprobe.cli.generate")
app = typer.Typer(help="Generate questions from knowledge sources")


@app.command("single")
def generate_single(
    content: Annotated[str, typer.Argument(help="Knowledge content to generate question from")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model name to use")] = "",
    provider: Annotated[
        ModelProvider,
        typer.Option("--provider", "-p", help="Model provider", case_sensitive=False),
    ] = ModelProvider.OLLAMA,
    strategy: Annotated[
        PromptStrategy,
        typer.Option("--strategy", "-s", help="Prompt strategy", case_sensitive=False),
    ] = PromptStrategy.CHAIN_OF_THOUGHT,
    question_type: Annotated[
        QuestionType,
        typer.Option("--type", "-t", help="Question type", case_sensitive=False),
    ] = QuestionType.FACTUAL,
    source_id: Annotated[
        str, typer.Option("--source-id", help="Knowledge source ID")
    ] = "cli-input",
    input_type: Annotated[
        str, typer.Option("--input-type", help="Input type (triple|schema|text|entity)")
    ] = "text",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path (JSON)")
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", help="Generation temperature")
    ] = None,
    max_length: Annotated[
        int | None, typer.Option("--max-length", help="Maximum generation length")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Generate a single question from knowledge content.

    Examples:
        kp generate single "Albert Einstein developed the theory of relativity."
        kp generate single "Paris is the capital of France" -m "qwen2.5:7b" -s few_shot -t factual
    """
    try:
        _configure_environment(config, verbose)
        settings = get_settings()

        # Resolve model name
        model_name = model or settings.models.local.default_model

        print_header(f"Question Generation — {model_name}")
        print_info(
            f"Strategy: {strategy.value} | Type: {question_type.value} | Provider: {provider.value}"
        )

        # Create knowledge input
        knowledge = KnowledgeInput(
            source_id=source_id,
            input_type=input_type,
            content=content,
            metadata={
                "temperature": temperature or settings.generation.temperature,
                "max_length": max_length or settings.generation.max_length,
            },
        )

        # Build generation parameters
        gen_params = {
            "temperature": temperature or settings.generation.temperature,
            "max_length": max_length or settings.generation.max_length,
            "top_p": settings.generation.top_p,
            "top_k": settings.generation.top_k,
        }

        # Generate question (placeholder for actual generator integration)
        start_time = datetime.utcnow()
        question = _generate_question_placeholder(
            knowledge=knowledge,
            model_name=model_name,
            provider=provider,
            strategy=strategy,
            question_type=question_type,
            gen_params=gen_params,
        )
        duration = format_duration(start_time)

        # Display result
        print_success(f"Generated in {duration}")
        console.print(questions_to_table([question]))

        # Display the actual question text prominently
        console.print()
        console.print("[bold green]Question:[/bold green]")
        console.print(f"[italic]{question.question_text}[/italic]")

        if question.confidence is not None:
            console.print(f"[dim]Confidence: {question.confidence:.4f}[/dim]")

        # Save output if requested
        if output:
            save_json_output(question, output)

    except CLIError:
        raise
    except Exception as e:
        logger.exception("Question generation failed")
        raise GenerationError(str(e)) from e


@app.command("batch")
def generate_batch(
    input_file: Annotated[
        Path, typer.Argument(help="Path to input file (JSON/JSONL with knowledge entries)")
    ],
    model: Annotated[str, typer.Option("--model", "-m", help="Model name to use")] = "",
    provider: Annotated[
        ModelProvider,
        typer.Option("--provider", "-p", help="Model provider", case_sensitive=False),
    ] = ModelProvider.OLLAMA,
    strategy: Annotated[
        PromptStrategy,
        typer.Option("--strategy", "-s", help="Prompt strategy", case_sensitive=False),
    ] = PromptStrategy.CHAIN_OF_THOUGHT,
    question_type: Annotated[
        QuestionType,
        typer.Option("--type", "-t", help="Question type", case_sensitive=False),
    ] = QuestionType.FACTUAL,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path (JSON)")] = Path(
        "generated_questions.json"
    ),
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", help="Generation temperature")
    ] = None,
    max_length: Annotated[
        int | None, typer.Option("--max-length", help="Maximum generation length")
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Batch size for processing")
    ] = 8,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be generated without running")
    ] = False,
) -> None:
    """Generate questions in batch from a knowledge file.

    The input file should contain JSON or JSONL entries with knowledge content.
    Each entry should have at minimum a 'content' field.

    Examples:
        kp generate batch knowledge.jsonl -m llama3.1:8b -s cot -o questions.json
        kp generate batch data/schema.json --type schema --strategy few_shot
    """
    try:
        _configure_environment(config, verbose)
        settings = get_settings()

        # Validate input file
        input_path = validate_input_file(input_file)

        # Parse input file
        knowledge_entries = _parse_knowledge_file(input_path)
        if not knowledge_entries:
            raise ConfigurationError(f"No valid knowledge entries found in {input_path}")

        print_header(f"Batch Question Generation — {len(knowledge_entries)} entries")
        print_info(f"Model: {model or settings.models.local.default_model}")
        print_info(f"Strategy: {strategy.value} | Type: {question_type.value}")
        print_info(f"Batch size: {batch_size}")

        if dry_run:
            print_warning("DRY RUN — no questions will be generated")
            console.print(f"Would process {len(knowledge_entries)} entries")
            return

        # Confirm large batches
        if len(knowledge_entries) > 100 and not confirm_action(
            f"Generate questions for {len(knowledge_entries)} entries? This may take a while."
        ):
            print_info("Operation cancelled")
            raise typer.Exit(0)

        # Generate questions with progress bar
        start_time = datetime.utcnow()
        questions: list[GeneratedQuestion] = []

        with create_progress_bar("Generating questions...") as progress:
            task = progress.add_task("Generating", total=len(knowledge_entries))

            for i, entry in enumerate(knowledge_entries):
                try:
                    knowledge = KnowledgeInput(
                        source_id=entry.get("source_id", f"batch-{i}"),
                        input_type=entry.get("input_type", "text"),
                        content=entry["content"],
                        structured=entry.get("structured", {}),
                        metadata=entry.get("metadata", {}),
                    )

                    gen_params = {
                        "temperature": temperature or settings.generation.temperature,
                        "max_length": max_length or settings.generation.max_length,
                    }

                    question = _generate_question_placeholder(
                        knowledge=knowledge,
                        model_name=model or settings.models.local.default_model,
                        provider=provider,
                        strategy=strategy,
                        question_type=question_type,
                        gen_params=gen_params,
                    )
                    questions.append(question)
                    progress.advance(task)

                except Exception as e:
                    logger.warning(f"Failed to generate question for entry {i}: {e}")
                    print_warning(f"Skipping entry {i}: {e}")

        duration = format_duration(start_time)

        # Summary
        print_success(
            f"Generated {len(questions)}/{len(knowledge_entries)} questions in {duration}"
        )

        if questions:
            console.print(questions_to_table(questions[:10]))  # Show first 10
            if len(questions) > 10:
                print_info(f"... and {len(questions) - 10} more")

        # Save output
        save_json_output(questions, output)

    except CLIError:
        raise
    except Exception as e:
        logger.exception("Batch generation failed")
        raise GenerationError(str(e)) from e


@app.command("preview")
def preview_prompt(
    content: Annotated[str, typer.Argument(help="Knowledge content to preview prompt for")],
    strategy: Annotated[
        PromptStrategy,
        typer.Option("--strategy", "-s", help="Prompt strategy", case_sensitive=False),
    ] = PromptStrategy.CHAIN_OF_THOUGHT,
    question_type: Annotated[
        QuestionType,
        typer.Option("--type", "-t", help="Question type", case_sensitive=False),
    ] = QuestionType.FACTUAL,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Preview the prompt that would be sent to the model without generating.

    Examples:
        kp generate preview "The Eiffel Tower is in Paris" -s cot
        kp generate preview "Schema: Person(name, age)" -t schema -s few_shot
    """
    try:
        _configure_environment(config, verbose)

        print_header(f"Prompt Preview — {strategy.value}")

        # Build the prompt (placeholder for actual prompt builder)
        prompt = _build_prompt_placeholder(
            content=content,
            strategy=strategy,
            question_type=question_type,
        )

        console.print("[bold]System Prompt:[/bold]")
        console.print(prompt["system"], style="dim")
        console.print()
        console.print("[bold]User Prompt:[/bold]")
        console.print(prompt["user"])

        if strategy == PromptStrategy.FEW_SHOT:
            console.print()
            print_info(
                f"Would include {get_settings().prompts.few_shot_examples} few-shot examples"
            )
        elif strategy == PromptStrategy.SELF_CONSISTENCY:
            console.print()
            print_info(
                f"Would sample {get_settings().prompts.self_consistency_samples} times for self-consistency"
            )

    except Exception as e:
        logger.exception("Prompt preview failed")
        raise GenerationError(str(e)) from e


def _configure_environment(config_path: Path | None, verbose: bool) -> None:
    """Configure logging and settings from CLI options.

    Args:
        config_path: Optional path to config file.
        verbose: Whether to enable verbose/debug logging.
    """
    if config_path:
        load_settings(config_path)

    log_level = "DEBUG" if verbose else get_settings().app.log_level
    configure_logging(level=log_level, debug=verbose)
    logger.debug(
        "Logging configured", level=log_level, config=str(config_path) if config_path else None
    )


def _parse_knowledge_file(path: Path) -> list[dict]:
    """Parse a JSON or JSONL knowledge file.

    Args:
        path: Path to the input file.

    Returns:
        List of knowledge entry dictionaries.

    Raises:
        ConfigurationError: If the file cannot be parsed.
    """
    try:
        content = path.read_text(encoding="utf-8")

        # Try JSONL first
        if path.suffix in (".jsonl", ".ndjson") or content.strip().count("\n") > 0:
            entries = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "content" in entry:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
            if entries:
                return entries

        # Try JSON array
        data = json.loads(content)
        if isinstance(data, list):
            return [entry for entry in data if isinstance(entry, dict) and "content" in entry]
        elif isinstance(data, dict) and "content" in data:
            return [data]

        raise ConfigurationError(f"Could not parse knowledge entries from {path}")

    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to read {path}: {e}") from e


def _generate_question_placeholder(
    knowledge: KnowledgeInput,
    model_name: str,
    provider: ModelProvider,
    strategy: PromptStrategy,
    question_type: QuestionType,
    gen_params: dict,
) -> GeneratedQuestion:
    """Placeholder for actual question generation logic.

    This function simulates question generation until the actual generator
    module is integrated. It returns a realistic GeneratedQuestion object.

    Args:
        knowledge: The knowledge input.
        model_name: Name of the model.
        provider: Model provider.
        strategy: Prompt strategy used.
        question_type: Type of question to generate.
        gen_params: Generation parameters.

    Returns:
        A GeneratedQuestion instance.
    """
    # Simulate generation based on question type
    if question_type == QuestionType.FACTUAL:
        question_text = f"What is the key fact about: {knowledge.content[:50]}...?"
    elif question_type == QuestionType.SCHEMA:
        question_text = f"How does the schema relate to: {knowledge.content[:50]}...?"
    else:
        question_text = f"Explain the relationship involving: {knowledge.content[:50]}...?"

    return GeneratedQuestion(
        id=f"q-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hash(knowledge.content) % 10000:04d}",
        question_text=question_text,
        knowledge_input=knowledge,
        question_type=question_type,
        prompt_strategy=strategy,
        model_name=model_name,
        model_provider=provider,
        generation_params=gen_params,
        raw_output=question_text,
        confidence=0.85 + (hash(knowledge.content) % 100) / 500,  # Simulated confidence
    )


def _build_prompt_placeholder(
    content: str,
    strategy: PromptStrategy,
    question_type: QuestionType,
) -> dict[str, str]:
    """Build a prompt preview for the given strategy and question type.

    Args:
        content: Knowledge content.
        strategy: Prompt strategy.
        question_type: Type of question.

    Returns:
        Dictionary with 'system' and 'user' prompt strings.
    """
    system_prompt = (
        "You are an expert at generating high-quality questions from knowledge sources. "
        "Generate questions that are clear, specific, and grounded in the provided knowledge."
    )

    strategy_instructions = {
        PromptStrategy.ZERO_SHOT: "Generate a question directly from the knowledge provided.",
        PromptStrategy.FEW_SHOT: (
            "Generate a question from the knowledge provided. "
            "Use the following examples as guidance for the expected format and quality."
        ),
        PromptStrategy.CHAIN_OF_THOUGHT: (
            "Generate a question from the knowledge provided. "
            "Think step-by-step: identify key entities, relationships, and facts, "
            "then formulate a precise question."
        ),
        PromptStrategy.SELF_CONSISTENCY: (
            "Generate a question from the knowledge provided. "
            "Consider multiple approaches and provide the most consistent result."
        ),
        PromptStrategy.REACT: (
            "Generate a question from the knowledge provided using reasoning and action steps."
        ),
    }

    user_prompt = f"""Knowledge Source:
{content}

Instructions:
{strategy_instructions.get(strategy, strategy_instructions[PromptStrategy.ZERO_SHOT])}

Question Type: {question_type.value}

Generate a question based on the knowledge above."""

    return {"system": system_prompt, "user": user_prompt}
