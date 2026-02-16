"""Configuration module for nanobot."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nanobot.config.loader import get_config_path, load_config
from nanobot.config.schema import Config, MemoryConfig
from nanobot.config.extensions import ExtensionConfigLoader, get_extension_loader
from nanobot.config.migration import ConfigMigrator, migrate_config


class AgentConfig(BaseSettings):
    """Runtime configuration for latent reasoning."""

    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    enable_quantum_latent: bool = True

    @property
    def clarify_entropy_threshold(self) -> float:
        return self.memory.clarify_entropy_threshold

    @property
    def latent_timeout_seconds(self) -> int:
        return self.memory.latent_timeout_seconds

    @property
    def max_context_nodes(self) -> int:
        return self.memory.max_context_nodes

    model_config = SettingsConfigDict(env_prefix="NANOBOT_")


settings = AgentConfig()

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "AgentConfig",
    "settings",
    "ExtensionConfigLoader",
    "get_extension_loader",
    "ConfigMigrator",
    "migrate_config",
]
