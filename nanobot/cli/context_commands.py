"""CLI commands for context stage control."""

import typer
from rich.console import Console
from rich.table import Table

from nanobot.runtime.state import state

context_app = typer.Typer(
    name="context-stage",
    help="Control individual context construction stages",
)

console = Console()

# All available context stages
CONTEXT_STAGES = [
    "identity",
    "bootstrap",
    "latent_state",
    "ALS",
    "core_memory",
    "fractal_memory",
    "entangled_memory",
    "skills",
    "skills_summary",
    "conversation_history",
    "user_message",
]


@context_app.command("show")
def context_show() -> None:
    """Show current state of all context stages."""
    stages = state.get_context_stages()
    
    table = Table(title="Context Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", justify="center")
    
    for stage_name in CONTEXT_STAGES:
        enabled = stages.get(stage_name, False)
        status = "[green]enabled[/green]" if enabled else "[red]disabled[/red]"
        table.add_row(stage_name, status)
    
    console.print(table)


@context_app.command("enable")
def context_enable(
    stage: str = typer.Argument(..., help="Stage name to enable"),
) -> None:
    """Enable a specific context stage."""
    try:
        state.enable_context_stage(stage)
        console.print(f"[green]✓[/green] Enabled context stage: {stage}")
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print(f"\nAvailable stages: {', '.join(CONTEXT_STAGES)}")
        raise typer.Exit(1)


@context_app.command("disable")
def context_disable(
    stage: str = typer.Argument(..., help="Stage name to disable"),
) -> None:
    """Disable a specific context stage."""
    try:
        state.disable_context_stage(stage)
        console.print(f"[green]✓[/green] Disabled context stage: {stage}")
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print(f"\nAvailable stages: {', '.join(CONTEXT_STAGES)}")
        raise typer.Exit(1)


@context_app.command("toggle")
def context_toggle(
    stage: str = typer.Argument(..., help="Stage name to toggle"),
) -> None:
    """Toggle a context stage (enable if disabled, disable if enabled)."""
    try:
        new_state = state.toggle_context_stage(stage)
        status = "enabled" if new_state else "disabled"
        console.print(f"[green]✓[/green] Toggled context stage '{stage}' to: {status}")
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}")
        console.print(f"\nAvailable stages: {', '.join(CONTEXT_STAGES)}")
        raise typer.Exit(1)


@context_app.command("enable-all")
def context_enable_all() -> None:
    """Enable all context stages."""
    state.enable_all_context_stages()
    console.print("[green]✓[/green] Enabled all context stages")


@context_app.command("disable-all")
def context_disable_all() -> None:
    """Disable all context stages."""
    confirm = typer.confirm(
        "This will disable ALL context stages including identity and user_message. Continue?"
    )
    if not confirm:
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)
    
    state.disable_all_context_stages()
    console.print("[green]✓[/green] Disabled all context stages")
