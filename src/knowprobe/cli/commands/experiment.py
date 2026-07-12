"""Experiment management CLI commands for KnowProbe."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from knowprobe.core.config import get_settings, load_settings
from knowprobe.core.models import (
    EvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    GeneratedQuestion,
    ModelProvider,
    PromptStrategy,
    QuestionType,
)
from knowprobe.utils.logging import configure_logging, get_logger

from ..utils import (
    CLIError,
    ConfigurationError,
    ExperimentError,
    console,
    create_progress_bar,
    experiment_summary_table,
    format_duration,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
    save_json_output,
    validate_input_file,
)

logger = get_logger("knowprobe.cli.experiment")
app = typer.Typer(help="Run and manage experiments")


@app.command("run")
def run_experiment(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Models to evaluate (can specify multiple)", case_sensitive=False),
    ] = ["llama3.1:8b"],
    strategies: Annotated[
        list[PromptStrategy],
        typer.Option("--strategy", "-s", help="Prompt strategies to evaluate", case_sensitive=False),
    ] = [PromptStrategy.ZERO_SHOT, PromptStrategy.FEW_SHOT, PromptStrategy.CHAIN_OF_THOUGHT],
    question_types: Annotated[
        list[QuestionType],
        typer.Option("--type", "-t", help="Question types to evaluate", case_sensitive=False),
    ] = [QuestionType.FACTUAL, QuestionType.SCHEMA],
    metrics: Annotated[
        list[str],
        typer.Option("--metric", help="Evaluation metrics", case_sensitive=False),
    ] = ["bleu", "rouge", "bert_score"],
    knowledge_sources: Annotated[
        list[Path],
        typer.Option("--source", help="Knowledge source files", case_sensitive=False),
    ] = [],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o", help="Output directory for results")] = Path("results"),
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    description: Annotated[str, typer.Option("--description", "-d", help="Experiment description")] = "",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
    skip_generation: Annotated[bool, typer.Option("--skip-generation", help="Skip generation, use existing results")] = False,
    skip_evaluation: Annotated[bool, typer.Option("--skip-evaluation", help="Skip evaluation, only generate questions")] = False,
) -> None:
    """Run a full experiment comparing models and strategies.

    This command orchestrates the complete pipeline: generation + evaluation
    across multiple configurations, producing a comprehensive comparison report.

    Examples:
        kp experiment run "baseline" -m llama3.1:8b -m qwen2.5:7b -s zero_shot -s cot -t factual
        kp experiment run "full_comparison" --type factual --type schema -o experiments/
    """
    try:
        _configure_environment(config, verbose)
        
        experiment_id = f"exp-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{name.lower().replace(' ', '_')}"
        
        print_header(f"Experiment: {name}")
        print_info(f"ID: {experiment_id}")
        print_info(f"Models: {', '.join(models)}")
        print_info(f"Strategies: {', '.join(s.value for s in strategies)}")
        print_info(f"Question Types: {', '.join(t.value for t in question_types)}")
        print_info(f"Metrics: {', '.join(metrics)}")
        print_info(f"Output: {output_dir}")
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            models=models,
            prompt_strategies=strategies,
            question_types=question_types,
            evaluation_metrics=metrics,
            knowledge_sources=[str(s) for s in knowledge_sources] if knowledge_sources else ["default"],
        )
        
        # Validate knowledge sources
        if knowledge_sources:
            for source in knowledge_sources:
                validate_input_file(source)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load knowledge entries
        knowledge_entries = _load_knowledge_entries(knowledge_sources)
        if not knowledge_entries:
            raise ConfigurationError("No knowledge entries available for the experiment")
        
        print_info(f"Loaded {len(knowledge_entries)} knowledge entries")
        
        if skip_generation and skip_evaluation:
            print_warning("Both generation and evaluation skipped — nothing to do")
            return
        
        # Run experiment
        start_time = datetime.utcnow()
        
        all_questions: list[GeneratedQuestion] = []
        all_evaluations: list[EvaluationResult] = []
        
        if not skip_generation:
            print_header("Phase 1: Question Generation")
            all_questions = _run_generation_phase(
                knowledge_entries=knowledge_entries,
                models=models,
                strategies=strategies,
                question_types=question_types,
            )
            
            # Save generated questions
            questions_file = output_dir / f"{experiment_id}_questions.json"
            save_json_output(all_questions, questions_file)
            print_info(f"Saved {len(all_questions)} questions to {questions_file}")
        else:
            print_info("Skipping generation phase")
            # Load existing questions if available
            existing_questions = output_dir / f"{experiment_id}_questions.json"
            if existing_questions.exists():
                data = json.loads(existing_questions.read_text(encoding="utf-8"))
                all_questions = [GeneratedQuestion(**q) for q in data]
                print_info(f"Loaded {len(all_questions)} existing questions")
        
        if not skip_evaluation and all_questions:
            print_header("Phase 2: Evaluation")
            all_evaluations = _run_evaluation_phase(
                questions=all_questions,
                metrics=metrics,
            )
            
            # Save evaluation results
            eval_file = output_dir / f"{experiment_id}_evaluations.json"
            save_json_output(all_evaluations, eval_file)
            print_info(f"Saved {len(all_evaluations)} evaluations to {eval_file}")
        else:
            print_info("Skipping evaluation phase")
        
        # Build experiment result
        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            config=experiment_config,
            questions=all_questions,
            evaluations=all_evaluations,
            summary=_build_experiment_summary(all_questions, all_evaluations, metrics),
        )
        
        duration = format_duration(start_time)
        
        # Final output
        print_header("Experiment Complete")
        print_success(f"Completed in {duration}")
        console.print(experiment_summary_table(experiment_result))
        
        # Save final result
        result_file = output_dir / f"{experiment_id}_result.json"
        save_json_output(experiment_result, result_file)
        
        # Generate comparison by strategy
        if all_evaluations:
            _generate_strategy_comparison(all_evaluations, output_dir, experiment_id)
        
    except CLIError:
        raise
    except Exception as e:
        logger.exception("Experiment failed")
        raise ExperimentError(str(e)) from e


@app.command("list")
def list_experiments(
    results_dir: Annotated[Path, typer.Argument(help="Directory containing experiment results")] = Path("results"),
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """List all experiments in the results directory.

    Examples:
        kp experiment list
        kp experiment list experiments/ -v
    """
    try:
        _configure_environment(config, verbose)
        
        if not results_dir.exists():
            print_info(f"Results directory not found: {results_dir}")
            return
        
        print_header(f"Experiments in {results_dir}")
        
        # Find all experiment result files
        result_files = sorted(results_dir.glob("*_result.json"))
        
        if not result_files:
            print_info("No experiment results found")
            return
        
        from rich.table import Table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Experiment ID", width=25)
        table.add_column("Name", width=20)
        table.add_column("Models", width=25)
        table.add_column("Strategies", width=20)
        table.add_column("Questions", width=10, justify="right")
        table.add_column("Evaluations", width=10, justify="right")
        table.add_column("Completed", width=20)
        
        for result_file in result_files:
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                table.add_row(
                    data.get("experiment_id", "N/A")[:25],
                    data.get("config", {}).get("name", "N/A")[:20],
                    ", ".join(data.get("config", {}).get("models", []))[:25],
                    ", ".join(data.get("config", {}).get("prompt_strategies", []))[:20],
                    str(len(data.get("questions", []))),
                    str(len(data.get("evaluations", []))),
                    data.get("completed_at", "N/A")[:20],
                )
            except Exception as e:
                logger.warning(f"Failed to parse {result_file}: {e}")
        
        console.print(table)
        print_success(f"Found {len(result_files)} experiments")
        
    except Exception as e:
        logger.exception("Failed to list experiments")
        raise ExperimentError(str(e)) from e


@app.command("show")
def show_experiment(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID or result file path")],
    results_dir: Annotated[Path, typer.Option("--results-dir", "-r", help="Results directory")] = Path("results"),
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Show detailed information about a specific experiment.

    Examples:
        kp experiment show exp-20240701-120000-baseline
        kp experiment show results/exp-xxx_result.json
    """
    try:
        _configure_environment(config, verbose)
        
        # Resolve experiment file
        if experiment_id.endswith(".json"):
            result_file = Path(experiment_id)
        else:
            result_file = results_dir / f"{experiment_id}_result.json"
        
        if not result_file.exists():
            # Try partial match
            matches = list(results_dir.glob(f"*{experiment_id}*"))
            if matches:
                result_file = matches[0]
            else:
                raise ConfigurationError(f"Experiment not found: {experiment_id}")
        
        data = json.loads(result_file.read_text(encoding="utf-8"))
        experiment = ExperimentResult(**data)
        
        print_header(f"Experiment: {experiment.config.name}")
        console.print(experiment_summary_table(experiment))
        
        if verbose and experiment.questions:
            from ..utils import questions_to_table
            console.print(questions_to_table(experiment.questions[:5]))
        
        if verbose and experiment.evaluations:
            from ..utils import evaluations_to_table
            console.print(evaluations_to_table(experiment.evaluations[:10]))
        
    except CLIError:
        raise
    except Exception as e:
        logger.exception("Failed to show experiment")
        raise ExperimentError(str(e)) from e


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
    logger.debug("Experiment logging configured", level=log_level)


def _load_knowledge_entries(sources: list[Path]) -> list[dict]:
    """Load knowledge entries from source files or default.

    Args:
        sources: List of source file paths.

    Returns:
        List of knowledge entry dictionaries.
    """
    if not sources:
        # Return default sample entries for demo
        return [
            {
                "source_id": "sample-1",
                "content": "The Eiffel Tower was constructed in 1889 by Gustave Eiffel.",
                "input_type": "text",
            },
            {
                "source_id": "sample-2",
                "content": "Python is a high-level programming language created by Guido van Rossum.",
                "input_type": "text",
            },
            {
                "source_id": "sample-3",
                "content": "Schema: Person(name: String, age: Integer, occupation: String)",
                "input_type": "schema",
            },
        ]
    
    entries = []
    for source in sources:
        validate_input_file(source)
        content = source.read_text(encoding="utf-8")
        
        try:
            data = json.loads(content)
            if isinstance(data, list):
                entries.extend(data)
            else:
                entries.append(data)
        except json.JSONDecodeError:
            # Try JSONL
            for line in content.strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        entries.append({"content": line, "source_id": str(source), "input_type": "text"})
    
    return [e for e in entries if "content" in e]


def _run_generation_phase(
    knowledge_entries: list[dict],
    models: list[str],
    strategies: list[PromptStrategy],
    question_types: list[QuestionType],
) -> list[GeneratedQuestion]:
    """Run the question generation phase of an experiment.

    Args:
        knowledge_entries: List of knowledge entries.
        models: List of model names.
        strategies: List of prompt strategies.
        question_types: List of question types.

    Returns:
        List of all generated questions.
    """
    from knowprobe.core.models import KnowledgeInput
    
    total_tasks = len(knowledge_entries) * len(models) * len(strategies) * len(question_types)
    questions: list[GeneratedQuestion] = []
    
    with create_progress_bar("Generating questions...") as progress:
        task = progress.add_task("Generating", total=total_tasks)
        
        for entry in knowledge_entries:
            knowledge = KnowledgeInput(
                source_id=entry.get("source_id", "unknown"),
                input_type=entry.get("input_type", "text"),
                content=entry["content"],
                structured=entry.get("structured", {}),
                metadata=entry.get("metadata", {}),
            )
            
            for model in models:
                for strategy in strategies:
                    for q_type in question_types:
                        # Placeholder generation (integrates with actual generator)
                        from ..commands.generate import _generate_question_placeholder
                        
                        question = _generate_question_placeholder(
                            knowledge=knowledge,
                            model_name=model,
                            provider=ModelProvider.OLLAMA,
                            strategy=strategy,
                            question_type=q_type,
                            gen_params={"temperature": 0.7, "max_length": 256},
                        )
                        questions.append(question)
                        progress.advance(task)
    
    return questions


def _run_evaluation_phase(
    questions: list[GeneratedQuestion],
    metrics: list[str],
) -> list[EvaluationResult]:
    """Run the evaluation phase of an experiment.

    Args:
        questions: List of generated questions.
        metrics: List of evaluation metrics.

    Returns:
        List of all evaluation results.
    """
    from ..commands.evaluate import _evaluate_question_placeholder
    
    evaluations: list[EvaluationResult] = []
    
    with create_progress_bar("Evaluating questions...") as progress:
        task = progress.add_task("Evaluating", total=len(questions))
        
        for question in questions:
            results = _evaluate_question_placeholder(
                question=question,
                reference=None,  # Self-evaluation in experiment mode
                metrics=metrics,
            )
            evaluations.extend(results)
            progress.advance(task)
    
    return evaluations


def _build_experiment_summary(
    questions: list[GeneratedQuestion],
    evaluations: list[EvaluationResult],
    metrics: list[str],
) -> dict:
    """Build summary statistics for an experiment.

    Args:
        questions: List of generated questions.
        evaluations: List of evaluation results.
        metrics: List of metric names.

    Returns:
        Dictionary of summary statistics.
    """
    import statistics
    
    summary: dict = {
        "total_questions": len(questions),
        "total_evaluations": len(evaluations),
        "models_used": list(set(q.model_name for q in questions)),
        "strategies_used": list(set(q.prompt_strategy.value for q in questions)),
        "question_types": list(set(q.question_type.value for q in questions)),
    }
    
    for metric in metrics:
        metric_scores = [e.score for e in evaluations if e.metric_name == metric.lower()]
        if metric_scores:
            summary[f"{metric}_mean"] = round(statistics.mean(metric_scores), 4)
            summary[f"{metric}_std"] = round(statistics.stdev(metric_scores), 4) if len(metric_scores) > 1 else 0.0
            summary[f"{metric}_min"] = round(min(metric_scores), 4)
            summary[f"{metric}_max"] = round(max(metric_scores), 4)
    
    # Strategy-wise breakdown
    strategy_breakdown = {}
    for strategy in summary.get("strategies_used", []):
        strategy_scores = {
            metric: [
                e.score for e in evaluations
                if e.metric_name == metric.lower()
                and e.question_id in [q.id for q in questions if q.prompt_strategy.value == strategy]
            ]
            for metric in metrics
        }
        strategy_breakdown[strategy] = {
            metric: round(statistics.mean(scores), 4) if scores else 0.0
            for metric, scores in strategy_scores.items()
        }
    
    summary["strategy_breakdown"] = strategy_breakdown
    return summary


def _generate_strategy_comparison(
    evaluations: list[EvaluationResult],
    output_dir: Path,
    experiment_id: str,
) -> None:
    """Generate a strategy comparison report from evaluation results.

    Args:
        evaluations: List of evaluation results.
        output_dir: Output directory.
        experiment_id: Experiment ID for filename.
    """
    import statistics
    from rich.table import Table
    
    # Group by metric and strategy (inferred from question IDs)
    comparison_file = output_dir / f"{experiment_id}_comparison.md"
    
    lines = [
        f"# Strategy Comparison Report: {experiment_id}",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]
    
    # Metric-wise summary
    metrics = set(e.metric_name for e in evaluations)
    for metric in sorted(metrics):
        scores = [e.score for e in evaluations if e.metric_name == metric]
        if scores:
            lines.append(f"## {metric.upper()}")
            lines.append(f"- Mean: {statistics.mean(scores):.4f}")
            lines.append(f"- Std: {statistics.stdev(scores) if len(scores) > 1 else 0:.4f}")
            lines.append(f"- Min: {min(scores):.4f}")
            lines.append(f"- Max: {max(scores):.4f}")
            lines.append("")
    
    comparison_file.write_text("\n".join(lines), encoding="utf-8")
    print_info(f"Strategy comparison saved to {comparison_file}")
