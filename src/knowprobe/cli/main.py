"""Main CLI entry point for KnowProbe.

This module registers all CLI commands and provides the top-level Typer app.
The CLI is registered as console scripts 'knowprobe' and 'kp' in pyproject.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from knowprobe import __version__
from knowprobe.core.config import get_settings, load_settings
from knowprobe.utils.logging import configure_logging, get_logger

from .utils import CLIError, console, print_error, print_info, print_success

# Import command submodules
from .commands import config, evaluate, experiment, generate

logger = get_logger("knowprobe.cli")

# Create the main Typer app
app = typer.Typer(
    name="knowprobe",
    help="KnowProbe: Knowledge-Grounded Question Generation and RAG Evaluation Platform",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)

# Register subcommands
app.add_typer(generate.app, name="generate", help="Generate questions from knowledge sources")
app.add_typer(evaluate.app, name="evaluate", help="Evaluate generated questions")
app.add_typer(experiment.app, name="experiment", help="Run and manage experiments")
app.add_typer(config.app, name="config", help="View and manage configuration")


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", "-V", help="Show version and exit")] = False,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Global config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output globally")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-essential output")] = False,
) -> None:
    """KnowProbe CLI — Research-grade question generation and evaluation.

    Use the subcommands to generate questions, evaluate them, run experiments,
    or manage configuration. Each subcommand has its own help:
        kp generate --help
        kp evaluate --help
        kp experiment --help
        kp config --help
    """
    # Handle version flag
    if version:
        console.print(f"KnowProbe [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit(0)
    
    # Load global configuration
    if config:
        load_settings(config)
        logger.debug("Loaded configuration from file", config_path=str(config))
    
    # Configure logging based on verbosity
    if verbose:
        configure_logging(level="DEBUG", debug=True)
        logger.debug("Verbose mode enabled")
    elif quiet:
        configure_logging(level="WARNING", debug=False)
    
    # Store options in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config"] = config


@app.command("serve")
def serve_api(
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind to")] = "",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 0,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Number of worker processes")] = 0,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload for development")] = False,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Start the KnowProbe API server.

    Launches the FastAPI application with uvicorn. Configuration is loaded
    from the config file or environment variables.

    Examples:
        kp serve
        kp serve --port 8080 --host 0.0.0.0
        kp serve --reload  # Development mode
    """
    try:
        if config:
            load_settings(config)
        if verbose:
            configure_logging(level="DEBUG", debug=True)
        
        settings = get_settings()
        
        # Resolve parameters from config or defaults
        host = host or settings.api.host
        port = port or settings.api.port
        workers = workers or settings.api.workers
        
        print_info(f"Starting API server on {host}:{port}")
        print_info(f"Workers: {workers}")
        if reload:
            print_info("Auto-reload enabled")
        
        # Import here to avoid heavy imports at CLI startup
        try:
            import uvicorn
        except ImportError as e:
            print_error("uvicorn is not installed. Install with: pip install uvicorn[standard]")
            raise typer.Exit(1) from e
        
        uvicorn.run(
            "knowprobe.api.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="debug" if verbose else "info",
        )
        
    except CLIError:
        raise
    except Exception as e:
        logger.exception("API server failed to start")
        print_error(f"Failed to start server: {e}")
        raise typer.Exit(1) from e


@app.command("dashboard")
def serve_dashboard(
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind to")] = 0,
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Start the KnowProbe Streamlit dashboard.

    Launches the interactive web dashboard for visualizing results and
    managing experiments.

    Examples:
        kp dashboard
        kp dashboard --port 8502
    """
    try:
        if config:
            load_settings(config)
        if verbose:
            configure_logging(level="DEBUG", debug=True)
        
        settings = get_settings()
        port = port or settings.dashboard.port
        
        print_info(f"Starting dashboard on port {port}")
        
        try:
            import streamlit.web.bootstrap as bootstrap
            from streamlit.web.bootstrap import RuntimeConfig
        except ImportError as e:
            print_error("Streamlit is not installed. Install with: pip install streamlit")
            raise typer.Exit(1) from e
        
        # Streamlit launch
        import os
        os.environ["STREAMLIT_SERVER_PORT"] = str(port)
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        
        # This is a placeholder — actual dashboard module would be implemented separately
        dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        if not dashboard_path.exists():
            print_warning(f"Dashboard not found at {dashboard_path}")
            print_info("Dashboard module is not yet implemented")
            return
        
        print_info(f"Launching dashboard from {dashboard_path}")
        # bootstrap.run(dashboard_path, [], flag_options={})
        
    except CLIError:
        raise
    except Exception as e:
        logger.exception("Dashboard failed to start")
        print_error(f"Failed to start dashboard: {e}")
        raise typer.Exit(1) from e


@app.command("status")
def show_status(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Show system status and health check.

    Checks connectivity to configured model providers, database, and displays
    system information.

    Examples:
        kp status
        kp status -v
    """
    try:
        if config:
            load_settings(config)
        if verbose:
            configure_logging(level="DEBUG", debug=True)
        
        settings = get_settings()
        
        console.print(Panel.fit("KnowProbe System Status", style="bold cyan"))
        
        # Version info
        console.print(f"[bold]Version:[/bold] {__version__}")
        console.print(f"[bold]Environment:[/bold] {settings.app.environment}")
        console.print(f"[bold]Debug Mode:[/bold] {'Yes' if settings.app.debug else 'No'}")
        console.print()
        
        # Check local model connectivity
        console.print("[bold]Model Providers:[/bold]")
        import httpx
        
        try:
            response = httpx.get(
                f"{settings.models.local.base_url}/api/tags",
                timeout=5.0,
            )
            if response.status_code == 200:
                print_success(f"Local provider ({settings.models.local.provider}): Connected")
                if verbose:
                    models = response.json().get("models", [])
                    console.print(f"  Available models: {len(models)}")
                    for model in models[:5]:
                        console.print(f"    - {model.get('name', 'unknown')}")
            else:
                print_error(f"Local provider: Status {response.status_code}")
        except Exception as e:
            print_error(f"Local provider: Unreachable ({e})")
        
        # Check API providers (just key presence, not actual connectivity)
        for provider_name, provider_config in settings.models.api.items():
            if provider_config.api_key:
                print_success(f"API provider ({provider_name}): Configured")
            else:
                print_info(f"API provider ({provider_name}): Not configured")
        
        console.print()
        
        # Check database
        console.print("[bold]Database:[/bold]")
        console.print(f"  URL: {settings.database.url}")
        console.print(f"  Echo: {settings.database.echo}")
        
        # Check directories
        console.print()
        console.print("[bold]Directories:[/bold]")
        dirs_to_check = ["configs", "data", "results", "configs/prompts"]
        for dir_name in dirs_to_check:
            p = Path(dir_name)
            status = "[green]✓[/green]" if p.exists() else "[yellow]✗[/yellow]"
            console.print(f"  {status} {dir_name}")
        
        print_success("Status check complete")
        
    except Exception as e:
        logger.exception("Status check failed")
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1) from e


# Error handling wrapper
def _handle_cli_errors() -> None:
    """Register global error handling for CLI exceptions."""
    pass


# Override Typer's exception handling to provide Rich-formatted errors
_original_error_handler = typer.main.except_hook


def _rich_except_handler(exc_type, exc_value, tb) -> None:
    """Custom exception handler with Rich formatting."""
    if isinstance(exc_value, CLIError):
        print_error(exc_value.message)
        raise typer.Exit(exc_value.exit_code)
    elif isinstance(exc_value, typer.Exit):
        raise exc_value
    else:
        # For unexpected errors, show a user-friendly message in non-verbose mode
        logger.exception("Unhandled CLI exception")
        console.print("[bold red]An unexpected error occurred.[/bold red]")
        console.print("[dim]Use --verbose for full traceback.[/dim]")
        raise typer.Exit(1)


# Apply the custom exception handler
import typer.main
typer.main.except_hook = _rich_except_handler


if __name__ == "__main__":
    app()
