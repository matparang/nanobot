"""Configuration migration utilities for deprecated fields."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fields that have been deprecated and should be moved to extensions.json
DEPRECATED_FIELDS = {
    "memory",
    "translator", 
    "telemetry",
    "enable_quantum_latent",
    "rate_limit_enabled",
    "rate_limit_max_calls",
    "rate_limit_window_seconds",
    "use_keyring",
}


class ConfigMigrator:
    """Handles migration of deprecated config fields to extensions.json."""
    
    def __init__(self, config_path: Path, extensions_path: Path | None = None):
        """
        Initialize the migrator.
        
        Args:
            config_path: Path to the main config.json file
            extensions_path: Path to extensions.json (defaults to same dir as config)
        """
        self.config_path = config_path
        self.extensions_path = extensions_path or config_path.parent / "extensions.json"
    
    def migrate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Migrate deprecated fields from config to extensions.
        
        Returns:
            Tuple of (core_config, extensions_config)
        """
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}, {}
        
        # Load the config file
        with open(self.config_path) as f:
            full_config = json.load(f)
        
        # Separate core config from deprecated fields
        core_config, deprecated = self._separate_fields(full_config)
        
        # Load existing extensions if present
        extensions_config = self._load_extensions()
        
        # Merge deprecated fields into extensions
        if deprecated:
            extensions_config = self._merge_deprecated_fields(extensions_config, deprecated)
            logger.info(f"Migrated {len(deprecated)} deprecated field(s) to extensions: {list(deprecated.keys())}")
        
        return core_config, extensions_config
    
    def _separate_fields(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Separate core config from deprecated fields.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Tuple of (core_config, deprecated_fields)
        """
        core_config = {}
        deprecated = {}
        
        for key, value in config.items():
            if key in DEPRECATED_FIELDS:
                deprecated[key] = value
            else:
                core_config[key] = value
        
        return core_config, deprecated
    
    def _load_extensions(self) -> dict[str, Any]:
        """Load existing extensions config or return default structure."""
        if self.extensions_path.exists():
            try:
                with open(self.extensions_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load extensions config: {e}, using empty config")
        
        return {"extensions": {}}
    
    def _merge_deprecated_fields(self, extensions: dict[str, Any], deprecated: dict[str, Any]) -> dict[str, Any]:
        """
        Merge deprecated fields into extensions config structure.
        
        Args:
            extensions: Existing extensions configuration
            deprecated: Deprecated fields to merge
            
        Returns:
            Updated extensions configuration
        """
        if "extensions" not in extensions:
            extensions["extensions"] = {}
        
        ext = extensions["extensions"]
        
        # Map deprecated fields to extension structure
        if "memory" in deprecated:
            ext["memory"] = deprecated["memory"]
        
        if "translator" in deprecated:
            ext["translator"] = deprecated["translator"]
        
        if "telemetry" in deprecated:
            ext["telemetry"] = deprecated["telemetry"]
        
        # Map rate limit fields to a single extension
        rate_limit_fields = {}
        if "rate_limit_enabled" in deprecated:
            rate_limit_fields["enabled"] = deprecated["rate_limit_enabled"]
        if "rate_limit_max_calls" in deprecated:
            rate_limit_fields["max_calls"] = deprecated["rate_limit_max_calls"]
        if "rate_limit_window_seconds" in deprecated:
            rate_limit_fields["window_seconds"] = deprecated["rate_limit_window_seconds"]
        
        if rate_limit_fields:
            ext["rate_limit"] = rate_limit_fields
        
        # Map custom/misc fields
        custom_fields = {}
        if "enable_quantum_latent" in deprecated:
            custom_fields["enable_quantum_latent"] = deprecated["enable_quantum_latent"]
        if "use_keyring" in deprecated:
            custom_fields["use_keyring"] = deprecated["use_keyring"]
        
        if custom_fields:
            if "custom" not in ext:
                ext["custom"] = {}
            ext["custom"].update(custom_fields)
        
        return extensions
    
    def save_extensions(self, extensions_config: dict[str, Any]) -> None:
        """
        Save extensions configuration to file.
        
        Args:
            extensions_config: Extensions configuration to save
        """
        self.extensions_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.extensions_path, "w") as f:
            json.dump(extensions_config, f, indent=2)
        
        logger.info(f"Saved extensions config to {self.extensions_path}")
    
    def save_core_config(self, core_config: dict[str, Any]) -> None:
        """
        Save core configuration to file.
        
        Args:
            core_config: Core configuration to save
        """
        # Create backup of original config
        backup_path = self.config_path.with_suffix(".json.bak")
        if self.config_path.exists() and not backup_path.exists():
            import shutil
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        with open(self.config_path, "w") as f:
            json.dump(core_config, f, indent=2)
        
        logger.info(f"Saved core config to {self.config_path}")


def migrate_config(config_path: Path | None = None, extensions_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Migrate configuration, separating deprecated fields into extensions.
    
    Args:
        config_path: Path to config.json (uses default if None)
        extensions_path: Path to extensions.json (uses default if None)
    
    Returns:
        Tuple of (core_config, extensions_config)
    """
    from nanobot.config.loader import get_config_path
    
    if config_path is None:
        config_path = get_config_path()
    
    migrator = ConfigMigrator(config_path, extensions_path)
    return migrator.migrate()
