"""Centralized audit logging for toggle and state operations."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AuditAction(str, Enum):
    TOGGLE_CHANGE = "toggle_change"
    STATE_INSPECT = "state_inspect"
    STATE_CLEAR = "state_clear"
    STATE_RELOAD = "state_reload"
    BASELINE_ENTER = "baseline_enter"
    BASELINE_EXIT = "baseline_exit"
    SERVICE_SUSPEND = "service_suspend"
    SERVICE_RESUME = "service_resume"
    REASONING_COMPLETED = "reasoning_completed"
    REASONING_ESCALATED = "reasoning_escalated"
    CHI_BUDGET_UPDATE = "chi_budget_update"


def get_audit_log_path() -> Path:
    log_dir = Path.home() / ".nanobot" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "audit.log"


def audit_log(action: AuditAction, details: dict[str, Any], source: str = "cli") -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": str(action.value),
        "source": source,
        "details": details,
    }
    with get_audit_log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_audit_log(
    since: datetime | None = None,
    action_filter: AuditAction | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    path = get_audit_log_path()
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if since:
            timestamp = entry.get("timestamp")
            if not timestamp:
                continue
            try:
                ts = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            if ts < since:
                continue

        if action_filter and entry.get("action") != action_filter.value:
            continue

        entries.append(entry)

    if limit <= 0:
        return []
    return entries[-limit:]
