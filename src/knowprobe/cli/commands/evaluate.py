"""Evaluation CLI commands for KnowProbe."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from knowprobe.core.config import get_settings, load_settings
from knowprobe.core.models import EvaluationResult, GeneratedQuestion
from knowprobe.utils.logging import configure_logging, get_logger

from ..utils import (
    CLIError,
    ConfigurationError,
    EvaluationError,
    console,
    create_progress_bar,
    evaluations_to_table,
    format_duration,
    print_header,
    print_info,
    print_success,
    print_warning,
    save_json_output,
    validate_input_file,
)

logger = get_logger("knowprobe.cli.evaluate")
app = typer.Typer(help="Evaluate generated questions with automatic metrics")


@app.command("single")
def evaluate_single(
    question_file: Annotated[Path, typer.Argument(help="Path to generated question JSON file")],
    reference: Annotated[
        str | None, typer.Option("--reference", "-r", help="Reference answer or ground truth")
    ] = None,
    metrics: Annotated[
        list[str],
        typer.Option(
            "--metric",
            help="Evaluation metrics to use (can specify multiple)",
            case_sensitive=False,
        ),
    ] = ["bleu"],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file path (JSON)")
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Evaluate a single generated question against a reference.

    Examples:
        kp evaluate single question.json -r "Paris is the capital of France" --metric bleu --metric rouge
        kp evaluate single question.json --reference reference.txt -o eval_result.json
    """
    try:
        _configure_environment(config, verbose)

        # Load question
        question_path = validate_input_file(question_file)
        question_data = json.loads(question_path.read_text(encoding="utf-8"))

        if isinstance(question_data, list):
            if not question_data:
                raise ConfigurationError("Empty question list in input file")
            question_data = question_data[0]

        question = GeneratedQuestion(**question_data)

        print_header(f"Single Question Evaluation — {question.id or 'N/A'}")
        print_info(f"Question: {question.question_text[:80]}...")
        print_info(f"Metrics: {', '.join(metrics)}")

        if reference:
            print_info(f"Reference: {reference[:80]}...")
        else:
            print_warning("No reference provided — metrics may be limited")

        # Run evaluation
        start_time = datetime.utcnow()
        results = _evaluate_question_placeholder(question, reference, metrics)
        duration = format_duration(start_time)

        # Display results
        print_success(f"Evaluation completed in {duration}")
        console.print(evaluations_to_table(results))

        # Summary
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            console.print(f"\n[bold]Average Score:[/bold] {avg_score:.4f}")

        # Save output
        if output:
            save_json_output(results, output)

    except CLIError:
        raise
    except Exception as e:
        logger.exception("Single evaluation failed")
        raise EvaluationError(str(e)) from e


@app.command("batch")
def evaluate_batch(
    questions_file: Annotated[Path, typer.Argument(help="Path to questions JSON/JSONL file")],
    references_file: Annotated[
        Path | None, typer.Option("--references", "-r", help="Path to references JSON/JSONL file")
    ] = None,
    metrics: Annotated[
        list[str],
        typer.Option(
            "--metric",
            help="Evaluation metrics to use (can specify multiple)",
            case_sensitive=False,
        ),
    ] = ["bleu", "rouge"],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path (JSON)")] = Path(
        "evaluation_results.json"
    ),
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Evaluate a batch of generated questions against references.

    The questions file should contain GeneratedQuestion objects in JSON or JSONL format.
    The references file should contain corresponding reference texts.

    Examples:
        kp evaluate batch questions.json -r references.jsonl --metric bleu -o results.json
        kp evaluate batch generated.json --metric bleu --metric rouge --metric bert_score
    """
    try:
        _configure_environment(config, verbose)

        # Load questions
        questions_path = validate_input_file(questions_file)
        questions = _load_questions(questions_path)

        if not questions:
            raise ConfigurationError(f"No valid questions found in {questions_path}")

        # Load references
        references: list[str | None] = []
        if references_file:
            refs_path = validate_input_file(references_file)
            references = _load_references(refs_path, len(questions))
        else:
            references = [None] * len(questions)

        print_header(f"Batch Evaluation — {len(questions)} questions")
        print_info(f"Metrics: {', '.join(metrics)}")
        if references_file:
            print_info(f"References: {references_file}")
        else:
            print_warning("No references file — using self-evaluation mode")

        # Run batch evaluation
        start_time = datetime.utcnow()
        all_results: list[EvaluationResult] = []

        with create_progress_bar("Evaluating questions...") as progress:
            task = progress.add_task("Evaluating", total=len(questions))

            for i, (question, reference) in enumerate(zip(questions, references, strict=False)):
                try:
                    results = _evaluate_question_placeholder(question, reference, metrics)
                    all_results.extend(results)
                    progress.advance(task)
                except Exception as e:
                    logger.warning(f"Failed to evaluate question {i}: {e}")
                    print_warning(f"Skipping question {i}: {e}")

        duration = format_duration(start_time)

        # Summary statistics
        print_success(f"Evaluated {len(questions)} questions in {duration}")

        if all_results:
            _display_evaluation_summary(all_results, metrics)
            console.print(evaluations_to_table(all_results[:20]))  # Show first 20

        # Save output
        save_json_output(all_results, output)

    except CLIError:
        raise
    except Exception as e:
        logger.exception("Batch evaluation failed")
        raise EvaluationError(str(e)) from e


@app.command("compare")
def compare_strategies(
    results_dir: Annotated[
        Path, typer.Argument(help="Directory containing evaluation result files")
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output comparison report path")
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Compare evaluation results across different strategies or models.

    The results directory should contain JSON files with evaluation results,
    where filenames indicate the strategy/model (e.g., 'cot_results.json').

    Examples:
        kp evaluate compare results/ -o comparison.md
        kp evaluate compare experiments/ --verbose
    """
    try:
        _configure_environment(config, verbose)

        if not results_dir.exists() or not results_dir.is_dir():
            raise ConfigurationError(f"Results directory not found: {results_dir}")

        print_header(f"Strategy Comparison — {results_dir}")

        # Load all result files
        result_files = list(results_dir.glob("*.json"))
        if not result_files:
            raise ConfigurationError(f"No JSON result files found in {results_dir}")

        print_info(f"Found {len(result_files)} result files")

        # Parse and aggregate results
        comparison_data = _aggregate_comparison_results(result_files)

        # Display comparison table
        _display_comparison_table(comparison_data)

        # Save comparison report
        if output:
            report = _generate_comparison_report(comparison_data)
            output.write_text(report, encoding="utf-8")
            print_success(f"Comparison report saved to {output}")

    except CLIError:
        raise
    except Exception as e:
        logger.exception("Comparison failed")
        raise EvaluationError(str(e)) from e


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
    logger.debug("Evaluation logging configured", level=log_level)


def _load_questions(path: Path) -> list[GeneratedQuestion]:
    """Load questions from a JSON or JSONL file.

    Args:
        path: Path to the questions file.

    Returns:
        List of GeneratedQuestion instances.

    Raises:
        ConfigurationError: If the file cannot be parsed.
    """
    try:
        content = path.read_text(encoding="utf-8")

        # Try JSONL first
        questions = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    questions.extend([GeneratedQuestion(**item) for item in data])
                else:
                    questions.append(GeneratedQuestion(**data))
            except (json.JSONDecodeError, TypeError):
                continue

        if questions:
            return questions

        # Try JSON array/object
        data = json.loads(content)
        if isinstance(data, list):
            return [GeneratedQuestion(**item) for item in data]
        else:
            return [GeneratedQuestion(**data)]

    except Exception as e:
        raise ConfigurationError(f"Failed to load questions from {path}: {e}") from e


def _load_references(path: Path, expected_count: int) -> list[str | None]:
    """Load reference texts from a JSON or JSONL file.

    Args:
        path: Path to the references file.
        expected_count: Expected number of references.

    Returns:
        List of reference strings.

    Raises:
        ConfigurationError: If the file cannot be parsed.
    """
    try:
        content = path.read_text(encoding="utf-8")
        references: list[str | None] = []

        # Try JSONL
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "reference" in data:
                    references.append(data["reference"])
                elif isinstance(data, str):
                    references.append(data)
                else:
                    references.append(str(data))
            except json.JSONDecodeError:
                references.append(line)

        if not references:
            # Try JSON array
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "reference" in item:
                        references.append(item["reference"])
                    elif isinstance(item, str):
                        references.append(item)
                    else:
                        references.append(str(item))

        if len(references) != expected_count:
            print_warning(
                f"Reference count mismatch: {len(references)} refs vs {expected_count} questions"
            )

        return references

    except Exception as e:
        raise ConfigurationError(f"Failed to load references from {path}: {e}") from e


def _evaluate_question_placeholder(
    question: GeneratedQuestion,
    reference: str | None,
    metrics: list[str],
) -> list[EvaluationResult]:
    """Placeholder for actual evaluation logic.

    Simulates metric computation until actual evaluator module is integrated.

    Args:
        question: The generated question to evaluate.
        reference: Optional reference text.
        metrics: List of metric names to compute.

    Returns:
        List of EvaluationResult instances.
    """
    results = []
    base_score = 0.3 + (hash(question.question_text) % 100) / 200

    for metric in metrics:
        # Simulate different metrics with varying scores
        metric_weights = {
            "bleu": 1.0,
            "rouge": 0.95,
            "bert_score": 0.98,
            "llm_judge": 0.85,
            "meteor": 0.92,
            "perplexity": 0.7,
        }

        weight = metric_weights.get(metric.lower(), 0.8)
        score = min(base_score * weight, 1.0)

        results.append(
            EvaluationResult(
                question_id=question.id or "unknown",
                metric_name=metric.lower(),
                score=round(score, 4),
                details={
                    "reference_length": len(reference) if reference else 0,
                    "question_length": len(question.question_text),
                    "has_reference": reference is not None,
                },
            )
        )

    return results


def _display_evaluation_summary(results: list[EvaluationResult], metrics: list[str]) -> None:
    """Display summary statistics for evaluation results.

    Args:
        results: List of evaluation results.
        metrics: List of metric names.
    """
    from rich.table import Table

    table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", width=15)
    table.add_column("Count", width=8, justify="right")
    table.add_column("Mean", width=10, justify="right")
    table.add_column("Std", width=10, justify="right")
    table.add_column("Min", width=10, justify="right")
    table.add_column("Max", width=10, justify="right")

    for metric in metrics:
        metric_results = [r.score for r in results if r.metric_name == metric.lower()]
        if metric_results:
            import statistics

            table.add_row(
                metric,
                str(len(metric_results)),
                f"{statistics.mean(metric_results):.4f}",
                f"{statistics.stdev(metric_results) if len(metric_results) > 1 else 0:.4f}",
                f"{min(metric_results):.4f}",
                f"{max(metric_results):.4f}",
            )

    console.print(table)
    console.print()


def _aggregate_comparison_results(result_files: list[Path]) -> dict[str, dict[str, float]]:
    """Aggregate evaluation results from multiple files for comparison.

    Args:
        result_files: List of paths to result JSON files.

    Returns:
        Dictionary mapping strategy names to metric averages.
    """
    comparison: dict[str, dict[str, list[float]]] = {}

    for file_path in result_files:
        strategy_name = file_path.stem.replace("_results", "").replace("_eval", "")

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    metric = item.get("metric_name", "unknown")
                    score = item.get("score", 0.0)

                    if strategy_name not in comparison:
                        comparison[strategy_name] = {}
                    if metric not in comparison[strategy_name]:
                        comparison[strategy_name][metric] = []
                    comparison[strategy_name][metric].append(score)
        except Exception as e:
            print_warning(f"Could not parse {file_path}: {e}")
            continue

    # Compute averages
    import statistics

    averaged: dict[str, dict[str, float]] = {}
    for strategy, metrics in comparison.items():
        averaged[strategy] = {}
        for metric, scores in metrics.items():
            averaged[strategy][metric] = round(statistics.mean(scores), 4) if scores else 0.0

    return averaged


def _display_comparison_table(comparison_data: dict[str, dict[str, float]]) -> None:
    """Display a comparison table across strategies.

    Args:
        comparison_data: Dictionary of strategy -> metric -> score.
    """
    from rich.table import Table

    if not comparison_data:
        print_warning("No comparison data to display")
        return

    # Collect all unique metrics
    all_metrics: set[str] = set()
    for metrics in comparison_data.values():
        all_metrics.update(metrics.keys())

    table = Table(title="Strategy Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Strategy", width=20)
    for metric in sorted(all_metrics):
        table.add_column(metric.upper(), width=12, justify="right")

    # Determine best per metric
    best_scores: dict[str, float] = {}
    for metric in all_metrics:
        scores = [metrics.get(metric, 0.0) for metrics in comparison_data.values()]
        best_scores[metric] = max(scores) if scores else 0.0

    for strategy in sorted(comparison_data.keys()):
        row = [strategy]
        for metric in sorted(all_metrics):
            score = comparison_data[strategy].get(metric, 0.0)
            cell = f"{score:.4f}"
            # Highlight best scores
            if best_scores.get(metric) == score and score > 0:
                cell = f"[bold green]{cell}[/bold green]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def _generate_comparison_report(comparison_data: dict[str, dict[str, float]]) -> str:
    """Generate a markdown comparison report.

    Args:
        comparison_data: Dictionary of strategy -> metric -> score.

    Returns:
        Markdown formatted comparison report.
    """
    lines = [
        "# KnowProbe Strategy Comparison Report\n",
        f"Generated: {datetime.utcnow().isoformat()}\n",
        "## Results Summary\n",
    ]

    for strategy, metrics in sorted(comparison_data.items()):
        lines.append(f"### {strategy}\n")
        for metric, score in sorted(metrics.items()):
            lines.append(f"- **{metric}**: {score:.4f}")
        lines.append("")

    return "\n".join(lines)
