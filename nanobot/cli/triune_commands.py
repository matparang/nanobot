"""CLI commands for Triune Memory System verification."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nanobot.cli.toggle_utils import toggle_feature
from nanobot.runtime.state import state
from nanobot.triune.verifier import TriuneVerifier

triune_app = typer.Typer(
    name="triune",
    help="Triune Memory System sync verification and management",
)

console = Console()


@triune_app.callback(invoke_without_command=True)
def triune_callback(ctx: typer.Context, action: str | None = typer.Argument(None)) -> None:
    if ctx.invoked_subcommand is None:
        if not toggle_feature("triune", state, "triune_memory_enabled", action or "interactive"):
            raise typer.Exit(code=1)


@triune_app.command("verify")
def verify_command(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Auto-regenerate drifted or missing YAML files",
    ),
    report: bool = typer.Option(
        False,
        "--report",
        help="Show detailed report",
    ),
    path: str = typer.Option(
        ".",
        "--path",
        "-p",
        help="Root path to verify (default: current directory)",
    ),
) -> None:
    """
    Verify Triune Memory sync integrity.

    Examples:
        nanobot triune verify          # Check all files
        nanobot triune verify --fix    # Auto-regenerate drifted YAML
        nanobot triune verify --report # Detailed report
    """
    root_path = Path(path).resolve()
    checksums_file = root_path / ".triune" / "checksums.json"

    console.print(f"[bold blue]Verifying Triune sync in:[/bold blue] {root_path}")

    verifier = TriuneVerifier(root_path, checksums_file)
    result = verifier.verify_all(fix=fix)

    if report:
        # Detailed report
        report_text = verifier.get_detailed_report(result)
        console.print(report_text)
    else:
        # Summary table
        _display_summary(result, fix)


def _display_summary(result, fixed: bool) -> None:
    """Display summary table."""
    table = Table(title="Triune Sync Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta", justify="right")

    table.add_row("Total MD files", str(result.total_md_files))
    table.add_row("Synced YAML files", f"✓ {result.synced_yaml_files}")

    if result.missing_yaml:
        status = "✗ (fixed)" if fixed else "✗"
        table.add_row("Missing YAML", f"{status} {len(result.missing_yaml)}")

    if result.drifted_files:
        status = "⚠ (fixed)" if fixed else "⚠"
        table.add_row("Drifted YAML", f"{status} {len(result.drifted_files)}")

    if result.orphaned_yaml:
        table.add_row("Orphaned YAML", f"○ {len(result.orphaned_yaml)}")

    if result.invalid_yaml:
        table.add_row("Invalid YAML", f"✗ {len(result.invalid_yaml)}")

    table.add_row("Sync Status", result.sync_status)

    console.print(table)

    # Show details if there are issues
    if result.missing_yaml or result.drifted_files or result.orphaned_yaml:
        console.print("\n[yellow]Run with --report for detailed information[/yellow]")

    if not fixed and (result.missing_yaml or result.drifted_files):
        console.print("[yellow]Run with --fix to auto-regenerate files[/yellow]")
