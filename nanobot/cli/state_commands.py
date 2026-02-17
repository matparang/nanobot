"""CLI commands for persistent state management."""

import json

import typer
from rich.console import Console
from rich.table import Table

from nanobot.cli.audit import AuditAction, audit_log, read_audit_log
from nanobot.cli.state_manager import (
    _CATEGORY_MAPPING,
    check_persistent_state,
    clear_persistent_state,
    reload_persistent_state,
)

state_app = typer.Typer(
    name="state",
    help="Inspect, clear, and manage persistent agent state",
)

console = Console()


@state_app.command("inspect")
def state_inspect(
    category: str | None = typer.Argument(
        None,
        help="Category: als, memory, sessions, cron, heartbeat, triune, soul, or 'all'",
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
) -> None:
    target = category or "all"
    if target not in _CATEGORY_MAPPING:
        raise typer.BadParameter(f"Invalid category: {target}")
    result = check_persistent_state(None if target == "all" else target)
    audit_log(AuditAction.STATE_INSPECT, {"category": target}, source="state_cli")

    if json_output:
        console.print_json(json.dumps(result))
        return

    table = Table(title="Persistent State")
    table.add_column("Name")
    table.add_column("Exists")
    table.add_column("Items", justify="right")
    table.add_column("Size", justify="right")
    for name, details in result.items():
        table.add_row(
            name,
            "yes" if details["exists"] else "no",
            str(details["item_count"]),
            str(details["size_bytes"]),
        )
    console.print(table)


@state_app.command("clear")
def state_clear(
    category: str = typer.Argument(..., help="Category to clear"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Preview without deleting"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompts"),
) -> None:
    if category not in _CATEGORY_MAPPING or category == "all":
        raise typer.BadParameter(f"Invalid category: {category}")
    results = clear_persistent_state(category, dry_run=dry_run, force=force)
    total = len(results)
    ok = sum(1 for success in results.values() if success)
    verb = "Would clear" if dry_run else "Cleared"
    console.print(f"{verb} {ok}/{total} items in '{category}'")


@state_app.command("reload")
def state_reload(
    category: str = typer.Argument(..., help="Category to reinitialize"),
) -> None:
    if category not in {"als", "memory", "triune"}:
        raise typer.BadParameter(f"Reload not supported for category: {category}")
    if not reload_persistent_state(category):
        raise typer.Exit(1)
    console.print(f"Reloaded state for '{category}'")


@state_app.command("audit")
def state_audit(
    limit: int = typer.Option(50, "--limit", "-l", help="Number of entries"),
    action: str | None = typer.Option(None, "--action", "-a", help="Filter by action type"),
) -> None:
    if action:
        try:
            action_filter = AuditAction(action)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid action: {action}") from exc
    else:
        action_filter = None
    entries = read_audit_log(action_filter=action_filter, limit=limit)
    for entry in entries:
        console.print(json.dumps(entry))
