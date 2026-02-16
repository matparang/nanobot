"""Gateway helpers and optional health route."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from nanobot.config.loader import load_config
from nanobot.config import get_extension_loader

try:
    from fastapi import APIRouter
except Exception:  # pragma: no cover - FastAPI is optional at runtime
    APIRouter = None  # type: ignore[assignment]


router = APIRouter() if APIRouter else None


def _version() -> str:
    try:
        return f"v{version('nanobot-ai')}"
    except PackageNotFoundError:
        return "v0.1.4"


def build_health_payload() -> dict[str, Any]:
    """Build health payload from current settings."""
    config = load_config()
    loader = get_extension_loader()
    
    # Get memory config from extensions
    memory_config = loader.get_memory_config()
    memory_enabled = memory_config.get("enabled", True) if memory_config else True
    entanglement_weight = memory_config.get("entanglement_weight", 0.3) if memory_config else 0.3
    latent_max_depth = memory_config.get("latent_max_depth", 1) if memory_config else 1
    
    # Get custom config for quantum latent
    custom_config = loader.get_custom_config()
    enable_quantum_latent = custom_config.get("enable_quantum_latent", True) if custom_config else True
    
    payload = {
        "status": "ok",
        "version": _version(),
        "memory": {
            "enabled": memory_enabled,
            "entanglement_weight": entanglement_weight,
        },
        "latent_reasoning": {
            "enabled": enable_quantum_latent,
            "depth": latent_max_depth,
        },
    }

    # Add Triune status if available
    try:
        triune_status = _get_triune_status(config.workspace_path)
        if triune_status:
            payload["triune"] = triune_status
    except Exception:
        pass  # Silently skip if Triune status unavailable

    return payload


def _get_triune_status(workspace: Path) -> dict[str, Any] | None:
    """Get Triune Memory System status."""
    try:
        from nanobot.triune.verifier import TriuneVerifier

        checksums_file = workspace.parent / ".triune" / "checksums.json"
        verifier = TriuneVerifier(workspace.parent, checksums_file)
        result = verifier.verify_all(fix=False)

        # Check loader status
        loader_status = {}
        try:
            from nanobot.soul.loader import SoulLoader
            SoulLoader.get_instance(workspace).load()
            loader_status["soul"] = "loaded"
        except Exception:
            loader_status["soul"] = "error"

        try:
            from nanobot.governance.loader import GovernanceLoader
            gov_path = workspace.parent / "nanobot" / "governance" / "governance.yaml"
            if gov_path.exists():
                GovernanceLoader.get_instance(gov_path).load()
                loader_status["governance"] = "loaded"
            else:
                loader_status["governance"] = "not_configured"
        except Exception:
            loader_status["governance"] = "error"

        try:
            from nanobot.memory.loader import MemoryLoader
            mem_path = workspace / "memory" / "memory.yaml"
            if mem_path.exists():
                MemoryLoader.get_instance(mem_path).load()
                loader_status["memory"] = "loaded"
            else:
                loader_status["memory"] = "not_configured"
        except Exception:
            loader_status["memory"] = "error"

        try:
            from nanobot.latent.loader import LatentLoader
            lat_path = workspace.parent / "nanobot" / "latent" / "latent.yaml"
            if lat_path.exists():
                LatentLoader.get_instance(lat_path).load()
                loader_status["latent"] = "loaded"
            else:
                loader_status["latent"] = "not_configured"
        except Exception:
            loader_status["latent"] = "error"

        try:
            from nanobot.game.loader import GameLoader
            game_path = workspace.parent / "nanobot" / "game" / "game.yaml"
            if game_path.exists():
                GameLoader.get_instance(game_path).load()
                loader_status["game"] = "loaded"
            else:
                loader_status["game"] = "not_configured"
        except Exception:
            loader_status["game"] = "error"

        yaml_validity = "valid"
        if result.invalid_yaml:
            yaml_validity = "invalid"
        elif result.drifted_files:
            yaml_validity = "drifted"

        return {
            "sync_status": result.sync_status,
            "yaml_validity": yaml_validity,
            "loader_status": loader_status,
            "stats": {
                "total_md_files": result.total_md_files,
                "synced_yaml_files": result.synced_yaml_files,
                "drifted_files": len(result.drifted_files),
                "missing_yaml": len(result.missing_yaml),
                "orphaned_yaml": len(result.orphaned_yaml),
            },
            "last_verification": result.last_verification,
        }
    except Exception:
        return None


if router:
    @router.get("/health")
    async def health_check() -> dict[str, Any]:
        """Basic health endpoint for observability."""
        return build_health_payload()
