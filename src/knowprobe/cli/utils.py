"""CLI utilities and shared helpers for KnowProbe."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from knowprobe.core.models import EvaluationResult, ExperimentResult, GeneratedQuestion
from knowprobe.utils.logging import get_logger

logger = get_logger("knowprobe.cli")

T = TypeVar("T")

# Shared console instance for consistent output styling
console = Console(stderr=True)


class CLIError(Exception):
    """Base exception for CLI errors with user-friendly messages."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class ConfigurationError(CLIError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Configuration Error: {message}", exit_code=2)


class GenerationError(CLIError):
    """Raised when question generation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Generation Error: {message}", exit_code=3)


class EvaluationError(CLIError):
    """Raised when evaluation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Evaluation Error: {message}", exit_code=4)


class ExperimentError(CLIError):
    """Raised when experiment execution fails."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Experiment Error: {message}", exit_code=5)


def print_success(message: str) -> None:
    """Print a success message with green styling."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message with red styling."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow styling."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_header(title: str) -> None:
    """Print a styled header panel."""
    console.print(Panel(title, style="bold cyan", expand=False))


def create_progress_bar(description: str = "Processing...") -> Progress:
    """Create a Rich progress bar with standard styling.

    Args:
        description: Initial description text for the progress bar.

    Returns:
        A configured Rich Progress instance.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def questions_to_table(questions: list[GeneratedQuestion]) -> Table:
    """Convert a list of generated questions to a Rich Table.

    Args:
        questions: List of GeneratedQuestion instances.

    Returns:
        A Rich Table with formatted question data.
    """
    table = Table(
        title="Generated Questions",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("ID", style="dim", width=12)
    table.add_column("Question", width=50, no_wrap=False)
    table.add_column("Type", width=10)
    table.add_column("Strategy", width=12)
    table.add_column("Model", width=15)
    table.add_column("Confidence", width=10, justify="right")

    for q in questions:
        conf_str = f"{q.confidence:.3f}" if q.confidence is not None else "N/A"
        table.add_row(
            q.id or "N/A",
            Text(q.question_text, overflow="ellipsis"),
            q.question_type.value,
            q.prompt_strategy.value,
            q.model_name,
            conf_str,
        )
    return table


def evaluations_to_table(evaluations: list[EvaluationResult]) -> Table:
    """Convert a list of evaluation results to a Rich Table.

    Args:
        evaluations: List of EvaluationResult instances.

    Returns:
        A Rich Table with formatted evaluation data.
    """
    table = Table(
        title="Evaluation Results",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Question ID", style="dim", width=12)
    table.add_column("Metric", width=15)
    table.add_column("Score", width=10, justify="right")
    table.add_column("Details", width=40)

    for ev in evaluations:
        details = json.dumps(ev.details, indent=2) if ev.details else "{}"
        table.add_row(
            ev.question_id,
            ev.metric_name,
            f"{ev.score:.4f}",
            Text(details, overflow="ellipsis"),
        )
    return table


def experiment_summary_table(result: ExperimentResult) -> Table:
    """Create a summary table for an experiment result.

    Args:
        result: An ExperimentResult instance.

    Returns:
        A Rich Table with experiment summary statistics.
    """
    table = Table(
        title=f"Experiment Summary: {result.config.name}",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Metric", width=25)
    table.add_column("Value", width=40)

    summary_data: dict[str, Any] = {
        "Experiment ID": result.experiment_id,
        "Description": result.config.description or "N/A",
        "Models": ", ".join(result.config.models),
        "Strategies": ", ".join(s.value for s in result.config.prompt_strategies),
        "Question Types": ", ".join(t.value for t in result.config.question_types),
        "Total Questions": len(result.questions),
        "Total Evaluations": len(result.evaluations),
        "Completed At": result.completed_at.isoformat(),
    }

    for key, value in summary_data.items():
        table.add_row(key, str(value))

    return table


def save_json_output(data: Any, output_path: Path | str) -> None:
    """Save data as formatted JSON to a file.

    Args:
        data: The data to serialize (must be JSON-compatible or Pydantic model).
        output_path: Path to the output file.

    Raises:
        CLIError: If the file cannot be written.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(data, "model_dump"):
            json_data = json.dumps(data.model_dump(mode="json"), indent=2, ensure_ascii=False)
        elif hasattr(data, "__iter__") and not isinstance(data, str):
            json_data = json.dumps(
                [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in data
                ],
                indent=2,
                ensure_ascii=False,
            )
        else:
            json_data = json.dumps(data, indent=2, ensure_ascii=False)

        path.write_text(json_data, encoding="utf-8")
        print_success(f"Saved output to {path}")
    except OSError as e:
        raise CLIError(f"Failed to write output file {path}: {e}") from e


def validate_input_file(path: Path | str) -> Path:
    """Validate that an input file exists and is readable.

    Args:
        path: Path to the input file.

    Returns:
        The validated Path object.

    Raises:
        ConfigurationError: If the file does not exist or is not readable.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(f"Input file not found: {p}")
    if not p.is_file():
        raise ConfigurationError(f"Path is not a file: {p}")
    try:
        with p.open("rb"):
            pass
    except OSError as exc:
        raise ConfigurationError(f"File is not readable: {p}") from exc
    return p


def format_duration(start_time: datetime, end_time: datetime | None = None) -> str:
    """Format a duration between two timestamps.

    Args:
        start_time: The start timestamp.
        end_time: The end timestamp (defaults to now).

    Returns:
        A human-readable duration string.
    """
    end = end_time or datetime.utcnow()
    delta = end - start_time
    total_seconds = delta.total_seconds()

    if total_seconds < 1:
        return f"{total_seconds * 1000:.0f}ms"
    if total_seconds < 60:
        return f"{total_seconds:.2f}s"
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}m {seconds:.2f}s"


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt the user for confirmation.

    Args:
        message: The confirmation message to display.
        default: The default value if user just presses Enter.

    Returns:
        True if confirmed, False otherwise.
    """
    suffix = " [Y/n]" if default else " [y/N]"
    response = console.input(f"{message}{suffix}: ")
    if not response.strip():
        return default
    return response.strip().lower() in ("y", "yes")
