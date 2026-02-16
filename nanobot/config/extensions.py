"""Extension configuration loader for nanobot."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExtensionConfigLoader:
    """Loads and manages extension configurations from extensions.json."""
    
    def __init__(self, extensions_path: Path | None = None):
        """
        Initialize the extension config loader.
        
        Args:
            extensions_path: Path to extensions.json file
        """
        if extensions_path is None:
            from nanobot.config.loader import get_config_path
            config_path = get_config_path()
            extensions_path = config_path.parent / "extensions.json"
        
        self.extensions_path = extensions_path
        self._config: dict[str, Any] = {}
        self._loaded = False
    
    def load(self) -> dict[str, Any]:
        """
        Load extensions configuration from file.
        
        Returns:
            Extensions configuration dictionary
        """
        if self._loaded:
            return self._config
        
        if not self.extensions_path.exists():
            logger.debug(f"Extensions file not found: {self.extensions_path}")
            self._config = {"extensions": {}}
            self._loaded = True
            return self._config
        
        try:
            with open(self.extensions_path) as f:
                self._config = json.load(f)
            
            if "extensions" not in self._config:
                self._config = {"extensions": self._config}
            
            logger.info(f"Loaded extensions config from {self.extensions_path}")
            self._loaded = True
            return self._config
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load extensions config: {e}")
            self._config = {"extensions": {}}
            self._loaded = True
            return self._config
    
    def get_extension(self, name: str) -> dict[str, Any] | None:
        """
        Get configuration for a specific extension.
        
        Args:
            name: Extension name (e.g., 'memory', 'translator', 'telemetry')
        
        Returns:
            Extension configuration or None if not found
        """
        if not self._loaded:
            self.load()
        
        extensions = self._config.get("extensions", {})
        return extensions.get(name)
    
    def get_memory_config(self) -> dict[str, Any] | None:
        """Get memory extension configuration."""
        return self.get_extension("memory")
    
    def get_translator_config(self) -> dict[str, Any] | None:
        """Get translator extension configuration."""
        return self.get_extension("translator")
    
    def get_telemetry_config(self) -> dict[str, Any] | None:
        """Get telemetry extension configuration."""
        return self.get_extension("telemetry")
    
    def get_rate_limit_config(self) -> dict[str, Any] | None:
        """Get rate limit extension configuration."""
        return self.get_extension("rate_limit")
    
    def get_custom_config(self) -> dict[str, Any] | None:
        """Get custom extension configuration."""
        return self.get_extension("custom")
    
    def is_extension_enabled(self, name: str) -> bool:
        """
        Check if an extension is enabled.
        
        Args:
            name: Extension name
        
        Returns:
            True if extension is enabled, False otherwise
        """
        ext_config = self.get_extension(name)
        if ext_config is None:
            return False
        
        # Check for 'enabled' field
        if isinstance(ext_config, dict):
            return ext_config.get("enabled", False)
        
        return False
    
    def get_all_extensions(self) -> dict[str, Any]:
        """
        Get all extension configurations.
        
        Returns:
            Dictionary of all extension configurations
        """
        if not self._loaded:
            self.load()
        
        return self._config.get("extensions", {})


# Global extension config loader instance
_extension_loader: ExtensionConfigLoader | None = None


def get_extension_loader(extensions_path: Path | None = None) -> ExtensionConfigLoader:
    """
    Get the global extension config loader instance.
    
    Args:
        extensions_path: Optional path to extensions.json
    
    Returns:
        ExtensionConfigLoader instance
    """
    global _extension_loader
    
    if _extension_loader is None or extensions_path is not None:
        _extension_loader = ExtensionConfigLoader(extensions_path)
    
    return _extension_loader
