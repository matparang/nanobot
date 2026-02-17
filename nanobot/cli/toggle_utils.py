import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import click

from nanobot.cli.audit import AuditAction, audit_log


def get_toggle_log_path() -> Path:
    log_dir = Path.home() / ".nanobot" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "toggle.log"


def log_toggle_change(feature: str, old_value: bool, new_value: bool, source: str = "cli") -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "feature": feature,
        "old_value": old_value,
        "new_value": new_value,
        "source": source,
    }
    with get_toggle_log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    audit_log(
        AuditAction.TOGGLE_CHANGE,
        {"feature": feature, "old_value": old_value, "new_value": new_value},
        source=source,
    )


def toggle_feature(
    feature_name: str,
    state_obj: object,
    attr_name: str,
    action: Literal["on", "off", "interactive"] = "interactive",
):
    if action not in {"on", "off", "interactive"}:
        click.echo(f"Invalid action: {action}")
        return False

    current = getattr(state_obj, attr_name)

    if action == "interactive":
        click.echo(f"\n{feature_name.upper()} is currently: {'ON' if current else 'OFF'}")
        click.echo("1. ON")
        click.echo("2. OFF")

        choice = click.prompt("Select option", type=int, default=1 if not current else 2)
        if choice == 1:
            new_value = True
        elif choice == 2:
            new_value = False
        else:
            click.echo("Invalid choice")
            return False
    elif action == "on":
        new_value = True
    else:  # action == "off"
        new_value = False

    setattr(state_obj, attr_name, new_value)
    log_toggle_change(feature_name, current, new_value)
    click.echo(f"{feature_name.upper()} {'ENABLED' if new_value else 'DISABLED'}")
    return True


def batch_toggle(
    state_obj: object, toggles: dict[str, bool], log_source: str = "batch"
) -> dict[str, tuple[bool, bool]]:
    """Apply multiple toggle updates and return old/new values for each changed toggle."""
    results = {}
    attr_mapping = {
        "latent": "latent_reasoning_enabled",
        "mem0": "mem0_enabled",
        "fractal": "fractal_memory_enabled",
        "entangled": "entangled_memory_enabled",
        "triune": "triune_memory_enabled",
        "heartbeat": "heartbeat_enabled",
        "light_reasoner": "light_reasoner_enabled",
        "dual_layer": "dual_layer_enabled",
        "chi_tracking": "chi_tracking_enabled",
        "reasoning_audit": "reasoning_audit_enabled",
    }
    for name, new_value in toggles.items():
        attr_name = attr_mapping.get(name, name)
        if hasattr(state_obj, attr_name):
            old_value = getattr(state_obj, attr_name)
            setattr(state_obj, attr_name, bool(new_value))
            log_toggle_change(name, old_value, bool(new_value), log_source)
            results[name] = (old_value, bool(new_value))
    return results


def baseline_toggle_all(
    state_obj: object,
    enable: bool,
    log_source: str = "baseline",
) -> dict[str, tuple[bool, bool]]:
    all_features = {
        "latent": enable,
        "mem0": enable,
        "fractal": enable,
        "entangled": enable,
        "triune": enable,
        "heartbeat": enable,
        "light_reasoner": enable,
        "dual_layer": enable,
        "chi_tracking": enable,
        "reasoning_audit": enable,
    }
    results = batch_toggle(state_obj, all_features, log_source)
    audit_log(
        AuditAction.BASELINE_EXIT if enable else AuditAction.BASELINE_ENTER,
        {"toggles": {k: v[1] for k, v in results.items()}},
        source=log_source,
    )
    return results


def get_toggle_status_table(state_obj: object) -> dict[str, bool]:
    return state_obj.get_all_toggles()
