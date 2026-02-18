"""CLI commands for memory inspection and management."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ActiveLearningState
from nanobot.config.loader import load_config
from nanobot.memory.consolidation import ConsolidationPipeline
from nanobot.memory.relational_cache import RelationalCache
from nanobot.memory.session_store import SessionStore

app = typer.Typer(help="Memory inspection and management commands")
sessions_app = typer.Typer(help="Session management commands")
relational_app = typer.Typer(help="Relational cache commands")
console = Console()

# Register sub-apps
app.add_typer(sessions_app, name="sessions")
app.add_typer(relational_app, name="relational")


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


# ===== Session Management Commands =====

@sessions_app.command("list")
def sessions_list(
    include_archived: bool = typer.Option(False, "--archived", help="Include archived sessions"),
):
    """List all sessions."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    session_store = SessionStore(workspace)

    sessions = session_store.list_sessions(include_archived=include_archived)

    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return

    console.print(f"\n[bold cyan]Sessions[/bold cyan] ({len(sessions)} total)")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Events", justify="right")
    table.add_column("Status")

    for session in sessions:
        session_id = session["session_id"]
        created = session["created_at"]
        created_str = created[:16] if created and len(created) >= 16 else (created or "unknown")
        event_count = str(session["event_count"])
        status = "[yellow]archived[/yellow]" if session["archived"] else "[green]active[/green]"

        table.add_row(session_id, created_str, event_count, status)

    console.print(table)
    console.print()


@sessions_app.command("inspect")
def sessions_inspect(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
    limit: int = typer.Option(50, help="Number of events to show"),
    event_type: str = typer.Option(None, "--type", help="Filter by event type"),
):
    """Inspect session events."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    session_store = SessionStore(workspace)

    try:
        # Get session data
        session_data = session_store.load(session_id)

        # Get filtered events
        types = [event_type] if event_type else None
        events = session_store.inspect(session_id, limit=limit, types=types)

        console.print(f"\n[bold cyan]Session: {session_id}[/bold cyan]")
        console.print(f"[dim]Created: {session_data.get('created_at', 'unknown')}[/dim]")

        if session_data.get("metadata"):
            console.print(f"[dim]Metadata: {session_data['metadata']}[/dim]")

        console.print(f"\n[bold]Events[/bold] ({len(events)} shown)")

        for i, event in enumerate(events, 1):
            timestamp = event.get("timestamp", "")
            timestamp_str = timestamp[:19] if timestamp and len(timestamp) >= 19 else timestamp
            event_type_str = event.get("type", "unknown")
            payload = event.get("payload", {})

            console.print(f"\n[bold]{i}. [{timestamp_str}] {event_type_str}[/bold]")

            # Show payload preview
            if isinstance(payload, dict):
                for key, value in list(payload.items())[:3]:  # Show first 3 items
                    value_str = str(value)[:80]
                    console.print(f"   {key}: {value_str}{'...' if len(str(value)) > 80 else ''}")

        console.print()

    except FileNotFoundError:
        console.print(f"[red]Session '{session_id}' not found[/red]")
    except Exception as e:
        console.print(f"[red]Error loading session: {e}[/red]")


@sessions_app.command("archive")
def sessions_archive(
    session_id: str = typer.Argument(..., help="Session ID to archive"),
):
    """Archive a session."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    session_store = SessionStore(workspace)

    try:
        archive_path = session_store.archive(session_id)
        console.print(f"[green]✓[/green] Session '{session_id}' archived to: {archive_path}")
    except FileNotFoundError:
        console.print(f"[red]Session '{session_id}' not found[/red]")
    except Exception as e:
        console.print(f"[red]Error archiving session: {e}[/red]")


# ===== Relational Cache Commands =====

@relational_app.command("show")
def cache_show(
    limit: int = typer.Option(20, help="Number of patterns to show"),
    pattern_type: str = typer.Option(None, "--type", help="Filter by pattern type"),
):
    """Display relational cache patterns."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    cache = RelationalCache(workspace)

    patterns = cache.get_patterns(pattern_type=pattern_type, limit=limit)
    statistics = cache.get_statistics()

    console.print("\n[bold cyan]Relational Cache Statistics[/bold cyan]")
    console.print(f"Total Patterns: {statistics.get('total_patterns', 0)}")
    console.print(f"Total Relationships: {statistics.get('total_relationships', 0)}")

    tallest = statistics.get("tallest")
    if tallest:
        console.print(f"Tallest: {tallest.get('entity')} ({tallest.get('height')})")

    shortest = statistics.get("shortest")
    if shortest:
        console.print(f"Shortest: {shortest.get('entity')} ({shortest.get('height')})")

    if not patterns:
        console.print("\n[yellow]No patterns found[/yellow]")
        return

    console.print(f"\n[bold]Patterns[/bold] ({len(patterns)} shown)")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", justify="right")
    table.add_column("Type", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Entities")

    for pattern in patterns:
        pattern_id = str(pattern.get("id", ""))
        ptype = pattern.get("type", "unknown")
        timestamp = pattern.get("timestamp", "")
        timestamp_str = timestamp[:16] if timestamp and len(timestamp) >= 16 else timestamp
        entities = ", ".join(pattern.get("entities", [])[:3])

        table.add_row(pattern_id, ptype, timestamp_str, entities)

    console.print(table)
    console.print()


@relational_app.command("entities")
def cache_entities():
    """Display entities tracked in relational cache."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    cache = RelationalCache(workspace)

    entities = cache.get_entities()

    if not entities:
        console.print("[yellow]No entities found[/yellow]")
        return

    console.print(f"\n[bold cyan]Entities[/bold cyan] ({len(entities)} total)")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Entity", style="cyan")
    table.add_column("First Seen", style="dim")
    table.add_column("Patterns", justify="right")
    table.add_column("Attributes")

    for entity_name, entity_data in entities.items():
        first_seen = entity_data.get("first_seen", "")
        first_seen_str = first_seen[:10] if first_seen and len(first_seen) >= 10 else first_seen
        pattern_count = str(entity_data.get("pattern_count", 0))
        attributes = ", ".join(f"{k}={v}" for k, v in entity_data.get("attributes", {}).items())

        table.add_row(entity_name, first_seen_str, pattern_count, attributes[:40])

    console.print(table)
    console.print()


@relational_app.command("cycles")
def cache_cycles():
    """Detect and display relationship cycles."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()
    cache = RelationalCache(workspace)

    cycles = cache.detect_cycles()

    if not cycles:
        console.print("[green]No cycles detected[/green]")
        return

    console.print(f"\n[bold yellow]Cycles Detected[/bold yellow] ({len(cycles)} total)")

    for i, cycle in enumerate(cycles, 1):
        cycle_str = " → ".join(cycle)
        console.print(f"{i}. {cycle_str}")

    console.print()


@relational_app.command("query")
def cache_query(
    query: str = typer.Argument(..., help="Natural language query to execute"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed hypotheses"),
):
    """Query the relational cache using natural language."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()

    # Get entropy threshold from config
    memory_config = {}
    if hasattr(config, 'memory'):
        memory_config = config.memory if isinstance(config.memory, dict) else {}
    entropy_threshold = float(memory_config.get("clarify_entropy_threshold", 0.8))

    # Import here to avoid circular dependency
    from nanobot.memory.hypothesis_engine import HypothesisEngine

    engine = HypothesisEngine(workspace, entropy_threshold=entropy_threshold)

    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    result = engine.query(query, verbose=verbose)

    # Display answer
    confidence_color = "green" if result["confidence"] > 0.7 else "yellow" if result["confidence"] > 0.4 else "red"
    console.print(f"[bold]Answer:[/bold] {result['answer']}")
    console.print(f"[bold]Confidence:[/bold] [{confidence_color}]{result['confidence']:.2f}[/{confidence_color}]")
    console.print(f"[bold]Entropy:[/bold] {result['entropy']:.3f}")

    if result["requires_llm"]:
        console.print("[yellow]⚠ High entropy - LLM invocation recommended[/yellow]")
    else:
        console.print("[green]✓ Low entropy - answer from cache[/green]")

    console.print(f"[dim]Query type: {result['query_type']}[/dim]")

    # Show hypotheses in verbose mode
    if verbose and "hypotheses" in result:
        console.print("\n[bold]Hypotheses:[/bold]")
        for i, hyp in enumerate(result["hypotheses"], 1):
            console.print(f"\n{i}. [cyan]{hyp['intent']}[/cyan] (confidence: {hyp['confidence']:.2f})")
            console.print(f"   Reasoning: {hyp['reasoning']}")
            console.print(f"   Result: {hyp['result']}")

            if "evidence" in hyp:
                evidence = hyp["evidence"]
                console.print(f"   Evidence type: {evidence.get('type', 'unknown')}")

        if "cache_stats" in result:
            stats = result["cache_stats"]
            console.print(f"\n[dim]Cache: {stats['entities']} entities, {stats['relationships']} relationships[/dim]")

    console.print()


# ===== Consolidation Commands =====

@app.command()
def consolidate(
    session_ids: str = typer.Option(None, "--sessions", help="Comma-separated session IDs to consolidate"),
    archive: bool = typer.Option(True, "--archive/--no-archive", help="Archive sessions after consolidation"),
):
    """Run memory consolidation pipeline."""
    config = load_config()
    workspace = Path(config.agents.defaults.workspace).expanduser()

    # Get memory config if available
    memory_config = {}
    if hasattr(config, 'memory'):
        memory_config = config.memory if isinstance(config.memory, dict) else {}

    pipeline = ConsolidationPipeline(workspace, memory_config)

    # Parse session IDs if provided
    session_id_list = None
    if session_ids:
        session_id_list = [s.strip() for s in session_ids.split(",")]

    console.print("\n[bold cyan]Running Consolidation Pipeline[/bold cyan]")
    console.print("[dim]This may take a moment...[/dim]\n")

    try:
        stats = pipeline.run_full_pipeline(
            session_ids=session_id_list,
            archive_sessions=archive
        )

        console.print("[bold green]✓ Consolidation Complete[/bold green]\n")
        console.print(f"Sessions Processed: {stats['sessions_processed']}")
        console.print(f"Patterns Extracted: {stats['patterns_extracted']}")
        console.print(f"Relationships Extracted: {stats['relationships_extracted']}")
        console.print(f"Fractal Nodes Created: {stats['fractal_nodes_created']}")
        console.print(f"Long-term Entries: {stats['long_term_entries']}")
        console.print()

    except Exception as e:
        console.print(f"[red]Error during consolidation: {e}[/red]")


if __name__ == "__main__":
    app()
