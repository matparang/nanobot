"""Persistent state management for CLI commands."""

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console

from nanobot.cli.audit import AuditAction, audit_log

console = Console()


@dataclass
class PersistentFile:
    name: str
    path: Path
    file_type: Literal["json", "md", "yaml", "directory"]
    description: str


def get_workspace_path() -> Path:
    from nanobot.utils.helpers import get_workspace_path as _get_workspace_path

    return _get_workspace_path()


def get_data_dir() -> Path:
    from nanobot.config.loader import get_data_dir as _get_data_dir

    return _get_data_dir()


def get_persistent_files() -> list[PersistentFile]:
    workspace = get_workspace_path()
    data_dir = get_data_dir()
    return [
        PersistentFile("ALS", workspace / "memory" / "ALS.json", "json", "Active Learning State"),
        PersistentFile(
            "MEMORY.md",
            workspace / "memory" / "MEMORY.md",
            "md",
            "Long-term memory (Markdown)",
        ),
        PersistentFile(
            "MEMORY.yaml",
            workspace / "memory" / "MEMORY.yaml",
            "yaml",
            "Long-term memory (YAML)",
        ),
        PersistentFile("HISTORY.md", workspace / "memory" / "HISTORY.md", "md", "History log"),
        PersistentFile(
            "fractal_index",
            workspace / "memory" / "fractal_index.json",
            "json",
            "Fractal memory index",
        ),
        PersistentFile(
            "archives", workspace / "memory" / "archives", "directory", "Fractal memory archives"
        ),
        PersistentFile(
            "sessions",
            workspace / "sessions" if (workspace / "sessions").exists() else data_dir / "sessions",
            "directory",
            "Session history",
        ),
        PersistentFile("cron_jobs", data_dir / "cron" / "jobs.json", "json", "Scheduled cron jobs"),
        PersistentFile("HEARTBEAT", workspace / "HEARTBEAT.md", "md", "Heartbeat task list"),
        PersistentFile(
            "triune_checksums",
            workspace / ".triune" / "checksums.json",
            "json",
            "Triune sync checksums",
        ),
        PersistentFile("soul_config", workspace / "SOUL.md", "md", "Soul configuration"),
    ]


_CATEGORY_MAPPING = {
    "als": ["ALS"],
    "memory": ["MEMORY.md", "MEMORY.yaml", "HISTORY.md", "fractal_index", "archives"],
    "sessions": ["sessions"],
    "cron": ["cron_jobs"],
    "heartbeat": ["HEARTBEAT"],
    "triune": ["triune_checksums"],
    "soul": ["soul_config"],
    "all": None,
}


def check_persistent_state(category: str | None = None) -> dict[str, dict[str, Any]]:
    files = get_persistent_files()
    if category and category != "all":
        files = [f for f in files if f.name in _CATEGORY_MAPPING.get(category, [])]

    results = {}
    for pf in files:
        if pf.file_type == "directory":
            exists = pf.path.exists() and pf.path.is_dir()
            size = sum(f.stat().st_size for f in pf.path.rglob("*") if f.is_file()) if exists else 0
            count = len(list(pf.path.rglob("*"))) if exists else 0
        else:
            exists = pf.path.exists() and pf.path.is_file()
            size = pf.path.stat().st_size if exists else 0
            count = 1 if exists else 0
        results[pf.name] = {
            "path": str(pf.path),
            "exists": exists,
            "size_bytes": size,
            "item_count": count,
            "description": pf.description,
            "modified": datetime.fromtimestamp(pf.path.stat().st_mtime).isoformat() if exists else None,
        }
    return results


def clear_persistent_state(
    category: str,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, bool]:
    workspace = get_workspace_path()
    data_dir = get_data_dir()
    results = {}

    if category == "als":
        results["ALS.json"] = _clear_file(workspace / "memory" / "ALS.json", dry_run, force)
    elif category == "memory":
        memory_dir = workspace / "memory"
        for file_path in (
            memory_dir / "MEMORY.md",
            memory_dir / "MEMORY.yaml",
            memory_dir / "HISTORY.md",
            memory_dir / "fractal_index.json",
        ):
            results[file_path.name] = _clear_file(file_path, dry_run, force)
        results["archives/"] = _clear_directory(memory_dir / "archives", dry_run, force)
    elif category == "sessions":
        sessions_dir = workspace / "sessions"
        if not sessions_dir.exists():
            sessions_dir = data_dir / "sessions"
        results["sessions/"] = _clear_directory(sessions_dir, dry_run, force)
    elif category == "cron":
        results["jobs.json"] = _clear_file(data_dir / "cron" / "jobs.json", dry_run, force)
    elif category == "heartbeat":
        results["HEARTBEAT.md"] = _reset_heartbeat(workspace / "HEARTBEAT.md", dry_run, force)
    elif category == "triune":
        results["checksums.json"] = _clear_file(
            workspace / ".triune" / "checksums.json", dry_run, force
        )
    elif category == "soul":
        results["SOUL.md"] = _clear_file(workspace / "SOUL.md", dry_run, force)
    audit_log(
        AuditAction.STATE_CLEAR,
        {"category": category, "dry_run": dry_run, "results": results},
        source="state_manager",
    )
    return results


def _clear_file(path: Path, dry_run: bool, force: bool) -> bool:
    """Clear a file; returns True when cleared or already absent."""
    if not path.exists():
        return True
    if dry_run:
        return True
    if not force and not console.input(f"Delete {path.name}? [y/N]: ").lower().startswith("y"):
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _clear_directory(path: Path, dry_run: bool, force: bool) -> bool:
    """Clear directory contents; returns True when cleared or already absent/empty."""
    if not path.exists():
        return True
    items = list(path.iterdir())
    if not items or dry_run:
        return True
    if not force:
        confirm = console.input(f"Clear {len(items)} items from {path.name}/? [y/N]: ")
        if not confirm.lower().startswith("y"):
            return False
    try:
        for item in items:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        return True
    except Exception:
        return False


def _reset_heartbeat(path: Path, dry_run: bool, force: bool) -> bool:
    template = """# Heartbeat Tasks

This file is checked every 30 minutes by your nanobot agent.
Add tasks below that you want the agent to work on periodically.

If this file has no tasks (only headers and comments), the agent will skip the heartbeat.

## Active Tasks

<!-- Add your periodic tasks below this line -->


## Completed

<!-- Move completed tasks here or delete them -->
"""
    if dry_run:
        return True
    requires_confirmation = not force and path.exists()
    if requires_confirmation:
        confirm = console.input("Reset HEARTBEAT.md to template? [y/N]: ")
        if not confirm.lower().startswith("y"):
            return False
    try:
        path.write_text(template, encoding="utf-8")
        return True
    except Exception:
        return False


def reload_persistent_state(category: str) -> bool:
    """Recreate default persistent state for supported categories (`als`, `memory`)."""
    workspace = get_workspace_path()

    if category == "als":
        from nanobot.agent.memory_types import ActiveLearningState

        als_file = workspace / "memory" / "ALS.json"
        als_file.parent.mkdir(parents=True, exist_ok=True)
        if not als_file.exists():
            als_file.write_text(ActiveLearningState().model_dump_json(indent=2), encoding="utf-8")
        audit_log(AuditAction.STATE_RELOAD, {"category": category}, source="state_manager")
        return True

    if category == "memory":
        memory_dir = workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "archives").mkdir(exist_ok=True)
        index_file = memory_dir / "fractal_index.json"
        if not index_file.exists():
            index_file.write_text("[]", encoding="utf-8")
        memory_file = memory_dir / "MEMORY.md"
        if not memory_file.exists():
            memory_file.write_text("# Long-term Memory\n\n", encoding="utf-8")
        audit_log(AuditAction.STATE_RELOAD, {"category": category}, source="state_manager")
        return True

    if category == "triune":
        checksums_file = workspace / ".triune" / "checksums.json"
        checksums_file.parent.mkdir(parents=True, exist_ok=True)
        if not checksums_file.exists():
            checksums_file.write_text("{}", encoding="utf-8")
        audit_log(AuditAction.STATE_RELOAD, {"category": category}, source="state_manager")
        return True

    return False


def clear_all_baseline(
    dry_run: bool = False, force: bool = False
) -> dict[str, dict[str, bool]]:
    results = {}
    for category in ("als", "memory", "sessions", "cron", "heartbeat", "triune"):
        results[category] = clear_persistent_state(category, dry_run=dry_run, force=force)
    audit_log(
        AuditAction.STATE_CLEAR,
        {"categories": list(results.keys()), "dry_run": dry_run},
        source="baseline",
    )
    return results
