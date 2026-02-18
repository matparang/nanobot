"""CLI commands for memory inspection and management."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ActiveLearningState
from nanobot.config.loader import load_config

app = typer.Typer(help="Memory inspection and management commands")
console = Console()


def _get_memory_store() -> MemoryStore:
    """Get MemoryStore instance with workspace from config."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    # Get memory config if it exists
    memory_config = {}
    if hasattr(config, 'memory'):
        memory_config = config.memory if isinstance(config.memory, dict) else {}
    return MemoryStore(workspace, memory_config)


@app.command()
def status():
    """Display Active Learning State (ALS) status."""
    memory = _get_memory_store()

    if not memory.als_file.exists():
        console.print("[yellow]No ALS file found[/yellow]")
        return

    try:
        als = ActiveLearningState.model_validate_json(
            memory.als_file.read_text(encoding="utf-8")
        )

        console.print("\n[bold cyan]Active Learning State[/bold cyan]")
        console.print(f"[bold]Current Focus:[/bold] {als.current_focus}")
        console.print(f"[bold]Evolution Stage:[/bold] {als.evolution_stage}")
        console.print(f"[bold]Last Updated:[/bold] {als.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")

        if als.sparring_partners:
            console.print("\n[bold]Sparring Partners:[/bold]")
            for partner in als.sparring_partners:
                console.print(f"  • {partner}")

        if als.recent_reflections:
            console.print("\n[bold]Recent Reflections:[/bold]")
            for i, reflection in enumerate(als.recent_reflections[-5:], 1):
                console.print(f"  {i}. {reflection[:100]}{'...' if len(reflection) > 100 else ''}")

        console.print()

    except Exception as e:
        console.print(f"[red]Error reading ALS: {e}[/red]")


@app.command()
def list_hypotheses(
    limit: int = typer.Option(10, help="Number of entries to show"),
    high_entropy: bool = typer.Option(False, "--high-entropy", help="Show only high entropy entries"),
):
    """List hypotheses from pattern cache with entropy information."""
    memory = _get_memory_store()

    if not memory.pattern_cache_file.exists():
        console.print("[yellow]No pattern cache found[/yellow]")
        return

    entries = memory.get_pattern_cache_entries(limit=100)  # Get more to filter

    # Get threshold for highlighting
    threshold = float(memory.config.get("clarify_entropy_threshold", 0.8))

    if high_entropy:
        entries = [e for e in entries if e.get("entropy", 0.0) >= threshold]

    entries = entries[:limit]

    if not entries:
        console.print("[yellow]No hypothesis entries found[/yellow]")
        return

    console.print(f"\n[bold cyan]Hypothesis Entries[/bold cyan] (entropy threshold: {threshold})")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Time", style="dim")
    table.add_column("Entropy", justify="right")
    table.add_column("Top Intent", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Destination")

    for entry in entries:
        timestamp = entry.get("timestamp", "")[:16]  # YYYY-MM-DD HH:MM
        entropy = entry.get("entropy", 0.0)
        hypotheses = entry.get("hypotheses", [])

        # Get top hypothesis
        if hypotheses:
            top = hypotheses[0]
            intent = top.get("intent", "unknown")[:40]
            confidence = f"{top.get('confidence', 0.0):.2f}"
        else:
            intent = "no hypotheses"
            confidence = "N/A"

        # Determine destination based on entropy
        destination = "pattern_cache" if entropy >= threshold else "fractal_memory"

        # Color code entropy
        if entropy >= threshold:
            entropy_str = f"[red]{entropy:.3f}[/red]"
        else:
            entropy_str = f"[green]{entropy:.3f}[/green]"

        table.add_row(timestamp, entropy_str, intent, confidence, destination)

    console.print(table)
    console.print()


@app.command()
def fractal(
    limit: int = typer.Option(10, help="Number of nodes to show"),
    tag: str = typer.Option(None, help="Filter by tag"),
):
    """Inspect fractal memory nodes."""
    memory = _get_memory_store()

    if not memory.index_file.exists():
        console.print("[yellow]No fractal index found[/yellow]")
        return

    try:
        index_data = json.loads(memory.index_file.read_text(encoding="utf-8"))

        # Filter by tag if specified
        if tag:
            index_data = [e for e in index_data if tag in e.get("tags", [])]

        if not index_data:
            console.print("[yellow]No fractal nodes found[/yellow]")
            return

        # Sort by timestamp (most recent first)
        index_data = sorted(
            index_data,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:limit]

        console.print(f"\n[bold cyan]Fractal Memory Nodes[/bold cyan] ({len(index_data)} shown)")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Time", style="dim")
        table.add_column("Tags", style="cyan")
        table.add_column("Summary")
        table.add_column("Type", style="yellow")
        table.add_column("Depth", justify="right")

        for entry in index_data:
            timestamp = entry.get("timestamp", "")[:10]  # YYYY-MM-DD
            tags = ", ".join(entry.get("tags", [])[:3])
            summary = entry.get("summary", "")[:50]
            content_type = entry.get("content_type", "text")
            depth = str(entry.get("depth", 0))

            table.add_row(timestamp, tags, summary, content_type, depth)

        console.print(table)
        console.print(f"\nTotal nodes in index: {len(json.loads(memory.index_file.read_text()))}")
        console.print()

    except Exception as e:
        console.print(f"[red]Error reading fractal index: {e}[/red]")


@app.command()
def cache(
    limit: int = typer.Option(10, help="Number of entries to show"),
):
    """Inspect pattern cache entries."""
    memory = _get_memory_store()

    if not memory.pattern_cache_file.exists():
        console.print("[yellow]No pattern cache found[/yellow]")
        return

    entries = memory.get_pattern_cache_entries(limit=limit)

    if not entries:
        console.print("[yellow]No cache entries found[/yellow]")
        return

    console.print("\n[bold cyan]Pattern Cache Entries[/bold cyan]")

    for i, entry in enumerate(entries, 1):
        timestamp = entry.get("timestamp", "")
        entropy = entry.get("entropy", 0.0)
        user_msg = entry.get("user_message", "")[:80]
        hypotheses = entry.get("hypotheses", [])
        strategic = entry.get("strategic_direction", "")[:60]

        console.print(f"\n[bold]{i}. [{timestamp[:16]}][/bold] (entropy={entropy:.3f})")
        console.print(f"   User: {user_msg}{'...' if len(entry.get('user_message', '')) > 80 else ''}")
        console.print(f"   Strategic: {strategic}{'...' if len(entry.get('strategic_direction', '')) > 60 else ''}")
        console.print(f"   Hypotheses: {len(hypotheses)}")

        for j, hyp in enumerate(hypotheses[:3], 1):
            intent = hyp.get("intent", "unknown")[:50]
            confidence = hyp.get("confidence", 0.0)
            console.print(f"     {j}. {intent} ({confidence:.2f})")

    console.print()


@app.command()
def history(
    limit: int = typer.Option(20, help="Number of lines to show"),
):
    """Show recent HISTORY.md entries."""
    memory = _get_memory_store()

    if not memory.history_file.exists():
        console.print("[yellow]No history file found[/yellow]")
        return

    try:
        content = memory.history_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # Get last N lines
        recent = lines[-limit:] if len(lines) > limit else lines

        console.print(f"\n[bold cyan]Recent History[/bold cyan] ({len(recent)} entries)")
        console.print()

        for line in recent:
            if line.strip():
                # Highlight timestamps
                if line.startswith("["):
                    console.print(f"[green]{line}[/green]")
                else:
                    console.print(line)

        console.print()
        console.print(f"Total history lines: {len(lines)}")
        console.print()

    except Exception as e:
        console.print(f"[red]Error reading history: {e}[/red]")


if __name__ == "__main__":
    app()
