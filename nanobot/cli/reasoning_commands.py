"""CLI commands for dual-layer reasoning controls."""

import json
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from nanobot.cli.audit import AuditAction, audit_log
from nanobot.cli.toggle_utils import toggle_feature
from nanobot.runtime.state import state

reasoning_app = typer.Typer(name="reasoning", help="Dual-layer reasoning controls")
console = Console()


@reasoning_app.command("status")
def reasoning_status(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show reasoning toggles."""
    toggles = state.get_all_toggles()
    data = {
        "light_reasoner": toggles.get("light_reasoner", False),
        "latent_reasoning": toggles.get("latent_reasoning", False),
        "dual_layer": toggles.get("dual_layer", False),
        "chi_tracking": toggles.get("chi_tracking", False),
        "reasoning_audit": toggles.get("reasoning_audit", False),
    }
    if json_output:
        console.print_json(json.dumps(data))
        return

    table = Table(title="Reasoning Status")
    table.add_column("Feature")
    table.add_column("State")
    for key, enabled in data.items():
        table.add_row(key, "on" if enabled else "off")
    console.print(table)


@reasoning_app.command("mode")
def set_mode(mode: Literal["system1_only", "system2_only", "hybrid", "baseline"]) -> None:
    """Set reasoning mode preset."""
    state.set_reasoning_mode(mode)
    audit_log(AuditAction.TOGGLE_CHANGE, {"mode": mode}, source="reasoning")
    console.print(f"Reasoning mode set to [bold]{mode}[/bold]")


@reasoning_app.command("system1")
def toggle_system1(action: str | None = typer.Argument(None)) -> None:
    """Toggle LightReasoner on/off."""
    if not toggle_feature("light_reasoner", state, "light_reasoner_enabled", action or "interactive"):
        raise typer.Exit(code=1)


@reasoning_app.command("system2")
def toggle_system2(action: str | None = typer.Argument(None)) -> None:
    """Toggle LatentReasoner on/off."""
    if not toggle_feature("latent", state, "latent_reasoning_enabled", action or "interactive"):
        raise typer.Exit(code=1)


@reasoning_app.command("hybrid")
def toggle_hybrid(action: str | None = typer.Argument(None)) -> None:
    """Toggle dual-layer orchestration on/off."""
    if not toggle_feature("dual_layer", state, "dual_layer_enabled", action or "interactive"):
        raise typer.Exit(code=1)


@reasoning_app.command("chi")
def toggle_chi(action: str | None = typer.Argument(None)) -> None:
    """Toggle chi tracking on/off."""
    if not toggle_feature("chi_tracking", state, "chi_tracking_enabled", action or "interactive"):
        raise typer.Exit(code=1)
