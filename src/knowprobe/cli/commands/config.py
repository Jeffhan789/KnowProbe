"""Configuration management CLI commands for KnowProbe."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.syntax import Syntax
from rich.table import Table

from knowprobe.core.config import get_settings, load_settings
from knowprobe.utils.logging import configure_logging, get_logger

from ..utils import (
    CLIError,
    ConfigurationError,
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)

logger = get_logger("knowprobe.cli.config")
app = typer.Typer(help="View and manage configuration")


@app.command("show")
def show_config(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    format: Annotated[
        str,
        typer.Option(
            "--format", "-f", help="Output format (table|yaml|json)", case_sensitive=False
        ),
    ] = "table",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Display the current configuration.

    Shows all configuration values including defaults, config file settings,
    and environment variable overrides.

    Examples:
        kp config show
        kp config show -c configs/local.yaml --format yaml
        kp config show --format json
    """
    try:
        settings = load_settings(config) if config else get_settings()

        if verbose:
            configure_logging(level="DEBUG", debug=True)

        print_header("KnowProbe Configuration")

        if format.lower() == "yaml":
            _show_yaml_config(settings)
        elif format.lower() == "json":
            _show_json_config(settings)
        else:
            _show_table_config(settings)

        print_info(f"Configuration loaded from: {config or 'defaults'}")

    except Exception as e:
        logger.exception("Failed to show configuration")
        raise ConfigurationError(str(e)) from e


@app.command("validate")
def validate_config(
    config: Annotated[Path, typer.Argument(help="Config file path to validate")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Validate a configuration file.

    Checks that the config file is valid YAML, contains required fields,
    and has no conflicting settings.

    Examples:
        kp config validate configs/default.yaml
        kp config validate configs/production.yaml -v
    """
    try:
        if verbose:
            configure_logging(level="DEBUG", debug=True)

        print_header(f"Validating Configuration: {config}")

        if not config.exists():
            raise ConfigurationError(f"Config file not found: {config}")

        # Parse YAML
        try:
            config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config}: {e}") from e

        if config_data is None:
            config_data = {}

        print_info("Parsed YAML successfully")

        # Validate via Pydantic
        try:
            settings = load_settings(config)
            print_success("Configuration is valid")
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

        # Check environment variables
        env_vars = _check_environment_variables()
        if env_vars:
            print_info("Environment variables detected:")
            for var, value in env_vars.items():
                # Mask sensitive values
                display_value = (
                    "***"
                    if "key" in var.lower() or "secret" in var.lower() or "password" in var.lower()
                    else value
                )
                console.print(f"  {var}: {display_value}")

        # Display warnings for common issues
        _display_config_warnings(settings)

    except CLIError:
        raise
    except Exception as e:
        logger.exception("Config validation failed")
        raise ConfigurationError(str(e)) from e


@app.command("env")
def show_env(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show all env vars, not just KNOWPROBE_")
    ] = False,
) -> None:
    """Show environment variables affecting KnowProbe.

    Displays all relevant environment variables including KNOWPROBE_ prefixed
    ones and common API key variables.

    Examples:
        kp config env
        kp config env -v
    """
    try:
        print_header("Environment Variables")

        # KNOWPROBE_ prefixed variables
        knowprobe_vars = {k: v for k, v in os.environ.items() if k.startswith("KNOWPROBE_")}

        # API key variables
        api_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"]
        api_vars = {k: v for k, v in os.environ.items() if k in api_keys}

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Variable", width=30)
        table.add_column("Value", width=50)
        table.add_column("Source", width=15)

        for var, value in sorted(knowprobe_vars.items()):
            display_value = "***" if "key" in var.lower() else value
            table.add_row(var, display_value, "KNOWPROBE")

        for var, value in sorted(api_vars.items()):
            display_value = "***" if value else "(not set)"
            table.add_row(var, display_value, "API Key")

        if verbose:
            # Show all relevant vars
            relevant_prefixes = ("PYTHON", "PATH", "HOME", "USER")
            other_vars = {
                k: v
                for k, v in os.environ.items()
                if k.startswith(relevant_prefixes) and k not in knowprobe_vars and k not in api_vars
            }
            for var, value in sorted(other_vars.items())[:20]:  # Limit output
                table.add_row(var, value[:50], "System")

        if table.row_count == 0:
            print_info("No KNOWPROBE_ environment variables set")
        else:
            console.print(table)

        if not api_vars or not any(v for v in api_vars.values()):
            print_warning("No API keys configured — API models will not be available")

    except Exception as e:
        logger.exception("Failed to show environment variables")
        raise ConfigurationError(str(e)) from e


@app.command("init")
def init_config(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output config file path")] = Path(
        "configs/local.yaml"
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing file")] = False,
) -> None:
    """Create a new local configuration file from defaults.

    Generates a starter config file that can be customized for your environment.

    Examples:
        kp config init
        kp config init -o configs/myconfig.yaml
        kp config init --force
    """
    try:
        print_header("Initialize Configuration")

        if output.exists() and not force:
            print_error(f"Config file already exists: {output}")
            print_info("Use --force to overwrite")
            raise typer.Exit(1)

        # Create default config content
        default_config = """# KnowProbe Local Configuration
# Copy this file and customize for your environment

app:
  name: "KnowProbe"
  version: "2.0.0"
  environment: "development"
  debug: false
  log_level: "INFO"

database:
  url: "sqlite:///data/knowprobe.db"
  echo: false

models:
  local:
    provider: "ollama"
    base_url: "http://localhost:11434"
    default_model: "llama3.1:8b"
    timeout: 300
  api:
    openai:
      api_key: ""  # Set via env: OPENAI_API_KEY
      base_url: "https://api.openai.com/v1"
      default_model: "gpt-4o-mini"
    deepseek:
      api_key: ""  # Set via env: DEEPSEEK_API_KEY
      base_url: "https://api.deepseek.com/v1"
      default_model: "deepseek-chat"

generation:
  max_length: 256
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  batch_size: 8

evaluation:
  metrics:
    - bleu
    - rouge
    - bert_score

rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5

prompts:
  strategy: "cot"
  templates_dir: "configs/prompts"
  few_shot_examples: 3
"""

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(default_config, encoding="utf-8")
        print_success(f"Created configuration file: {output}")
        print_info("Edit this file to customize your settings")

    except Exception as e:
        logger.exception("Failed to initialize config")
        raise ConfigurationError(str(e)) from e


def _show_table_config(settings) -> None:
    """Display configuration as a formatted table.

    Args:
        settings: Settings instance to display.
    """
    # App config
    app_table = Table(title="Application", show_header=True, header_style="bold cyan")
    app_table.add_column("Setting", width=20)
    app_table.add_column("Value", width=40)

    for field, value in settings.app.model_dump().items():
        app_table.add_row(field, str(value))
    console.print(app_table)
    console.print()

    # Models config
    models_table = Table(title="Models", show_header=True, header_style="bold cyan")
    models_table.add_column("Setting", width=20)
    models_table.add_column("Value", width=40)

    models_table.add_row("Local Provider", settings.models.local.provider)
    models_table.add_row("Local Base URL", settings.models.local.base_url)
    models_table.add_row("Local Default Model", settings.models.local.default_model)
    models_table.add_row("Local Timeout", str(settings.models.local.timeout))

    for provider_name, provider_config in settings.models.api.items():
        models_table.add_row(f"API: {provider_name}", provider_config.default_model or "N/A")

    console.print(models_table)
    console.print()

    # Generation config
    gen_table = Table(title="Generation", show_header=True, header_style="bold cyan")
    gen_table.add_column("Setting", width=20)
    gen_table.add_column("Value", width=40)

    for field, value in settings.generation.model_dump().items():
        gen_table.add_row(field, str(value))
    console.print(gen_table)
    console.print()

    # Evaluation config
    eval_table = Table(title="Evaluation", show_header=True, header_style="bold cyan")
    eval_table.add_column("Setting", width=20)
    eval_table.add_column("Value", width=40)

    eval_table.add_row("Metrics", ", ".join(settings.evaluation.metrics))
    console.print(eval_table)


def _show_yaml_config(settings) -> None:
    """Display configuration as YAML.

    Args:
        settings: Settings instance to display.
    """
    config_dict = settings.model_dump(mode="json")
    yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=True, allow_unicode=True)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)


def _show_json_config(settings) -> None:
    """Display configuration as JSON.

    Args:
        settings: Settings instance to display.
    """
    import json

    json_str = json.dumps(settings.model_dump(mode="json"), indent=2, ensure_ascii=False)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(syntax)


def _check_environment_variables() -> dict[str, str]:
    """Check for relevant environment variables.

    Returns:
        Dictionary of environment variable names to values.
    """
    relevant = {}
    for key, value in os.environ.items():
        if key.startswith("KNOWPROBE_") or key in (
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            relevant[key] = value
    return relevant


def _display_config_warnings(settings) -> None:
    """Display warnings for common configuration issues.

    Args:
        settings: Current settings.
    """
    warnings = []

    # Check for debug mode in production
    if settings.app.environment == "production" and settings.app.debug:
        warnings.append("Debug mode is enabled in production environment")

    # Check model timeout
    if settings.models.local.timeout < 30:
        warnings.append(
            f"Model timeout ({settings.models.local.timeout}s) may be too short for large models"
        )

    # Check database URL
    if "sqlite" in settings.database.url and settings.app.environment == "production":
        warnings.append("Using SQLite in production — consider a production database")

    # Check API keys
    if not any(p.api_key for p in settings.models.api.values()):
        warnings.append("No API keys configured — only local models will be available")

    if warnings:
        print_warning("Configuration warnings:")
        for warning in warnings:
            console.print(f"  • {warning}")
    else:
        print_success("No configuration warnings")
