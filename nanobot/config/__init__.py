"""Configuration module for nanobot."""

from pydantic_settings import BaseSettings, SettingsConfigDict

from nanobot.config.loader import get_config_path, load_config
from nanobot.config.schema import Config


class AgentConfig(BaseSettings):
    """Runtime configuration for latent reasoning."""

    clarify_entropy_threshold: float = 0.8
    latent_timeout_seconds: int = 10
    max_context_nodes: int = 5
    latent_max_depth: int = 1
    latent_entropy_threshold: float = 0.8
    monte_carlo_samples: int = 1
    monte_carlo_top_k: int = 3

    model_config = SettingsConfigDict(env_prefix="NANOBOT_")


settings = AgentConfig()

__all__ = ["Config", "load_config", "get_config_path", "AgentConfig", "settings"]
