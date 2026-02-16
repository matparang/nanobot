"""Tests for configuration migration functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.config.migration import ConfigMigrator, migrate_config, DEPRECATED_FIELDS
from nanobot.config.extensions import ExtensionConfigLoader, get_extension_loader
from nanobot.config.loader import load_config


class TestConfigMigration:
    """Test configuration migration from deprecated fields to extensions."""

    def test_deprecated_fields_constant(self):
        """Test that DEPRECATED_FIELDS contains expected fields."""
        expected = {
            "memory",
            "translator",
            "telemetry",
            "enable_quantum_latent",
            "rate_limit_enabled",
            "rate_limit_max_calls",
            "rate_limit_window_seconds",
            "use_keyring",
        }
        assert DEPRECATED_FIELDS == expected

    def test_migrator_separates_core_from_deprecated(self):
        """Test that migrator correctly separates core config from deprecated fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Create config with both core and deprecated fields
            config_data = {
                "agents": {
                    "defaults": {
                        "model": "gpt-4",
                        "maxTokens": 8192
                    }
                },
                "memory": {
                    "enabled": True,
                    "provider": "local",
                    "topK": 5
                },
                "telemetry": {
                    "enabled": True,
                    "port": 9090
                },
                "enable_quantum_latent": True,
                "use_keyring": True,
                "rate_limit_enabled": True,
                "rate_limit_max_calls": 10,
                "rate_limit_window_seconds": 60
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            migrator = ConfigMigrator(config_path)
            core_config, extensions_config = migrator.migrate()
            
            # Check core config only has non-deprecated fields
            assert "agents" in core_config
            assert "memory" not in core_config
            assert "telemetry" not in core_config
            assert "enable_quantum_latent" not in core_config
            assert "use_keyring" not in core_config
            assert "rate_limit_enabled" not in core_config
            
            # Check extensions config has deprecated fields
            ext = extensions_config["extensions"]
            assert "memory" in ext
            assert "telemetry" in ext
            assert "rate_limit" in ext
            assert "custom" in ext

    def test_migrator_creates_rate_limit_section(self):
        """Test that rate limit fields are merged into rate_limit extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            config_data = {
                "rate_limit_enabled": True,
                "rate_limit_max_calls": 20,
                "rate_limit_window_seconds": 120
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            migrator = ConfigMigrator(config_path)
            _, extensions_config = migrator.migrate()
            
            rate_limit = extensions_config["extensions"]["rate_limit"]
            assert rate_limit["enabled"] is True
            assert rate_limit["max_calls"] == 20
            assert rate_limit["window_seconds"] == 120

    def test_migrator_creates_custom_section(self):
        """Test that custom fields are placed in custom extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            config_data = {
                "enable_quantum_latent": False,
                "use_keyring": False
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            migrator = ConfigMigrator(config_path)
            _, extensions_config = migrator.migrate()
            
            custom = extensions_config["extensions"]["custom"]
            assert custom["enable_quantum_latent"] is False
            assert custom["use_keyring"] is False

    def test_migrator_preserves_memory_structure(self):
        """Test that memory config structure is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            memory_config = {
                "enabled": True,
                "provider": "mem0",
                "topK": 10,
                "archiveDir": "my_archives",
                "alsEnabled": False
            }
            
            config_data = {"memory": memory_config}
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            migrator = ConfigMigrator(config_path)
            _, extensions_config = migrator.migrate()
            
            assert extensions_config["extensions"]["memory"] == memory_config

    def test_migrator_saves_extensions(self):
        """Test that migrator can save extensions to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            extensions_path = Path(tmpdir) / "extensions.json"
            
            config_data = {
                "memory": {"enabled": True},
                "telemetry": {"enabled": True, "port": 9090}
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            migrator = ConfigMigrator(config_path, extensions_path)
            _, extensions_config = migrator.migrate()
            migrator.save_extensions(extensions_config)
            
            # Verify file was created
            assert extensions_path.exists()
            
            # Verify content
            with open(extensions_path) as f:
                saved_data = json.load(f)
            
            assert "extensions" in saved_data
            assert "memory" in saved_data["extensions"]
            assert "telemetry" in saved_data["extensions"]

    def test_migrator_creates_backup(self):
        """Test that migrator creates backup of original config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            config_data = {"memory": {"enabled": True}}
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            migrator = ConfigMigrator(config_path)
            core_config, _ = migrator.migrate()
            migrator.save_core_config(core_config)
            
            backup_path = config_path.with_suffix(".json.bak")
            assert backup_path.exists()
            
            # Verify backup contains original data
            with open(backup_path) as f:
                backup_data = json.load(f)
            assert "memory" in backup_data

    def test_migrate_config_function(self):
        """Test the migrate_config convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            config_data = {
                "agents": {"defaults": {"model": "gpt-4"}},
                "memory": {"enabled": True},
                "enable_quantum_latent": True
            }
            
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            core_config, extensions_config = migrate_config(config_path)
            
            assert "agents" in core_config
            assert "memory" not in core_config
            assert "memory" in extensions_config["extensions"]


class TestExtensionConfigLoader:
    """Test extension configuration loader."""

    def test_loader_loads_extensions(self):
        """Test that loader can load extensions from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions_path = Path(tmpdir) / "extensions.json"
            
            extensions_data = {
                "extensions": {
                    "memory": {"enabled": True, "provider": "local"},
                    "telemetry": {"enabled": True, "port": 9090}
                }
            }
            
            with open(extensions_path, "w") as f:
                json.dump(extensions_data, f)
            
            loader = ExtensionConfigLoader(extensions_path)
            config = loader.load()
            
            assert "extensions" in config
            assert "memory" in config["extensions"]
            assert "telemetry" in config["extensions"]

    def test_loader_get_extension(self):
        """Test getting specific extension config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions_path = Path(tmpdir) / "extensions.json"
            
            extensions_data = {
                "extensions": {
                    "memory": {"enabled": True, "topK": 5}
                }
            }
            
            with open(extensions_path, "w") as f:
                json.dump(extensions_data, f)
            
            loader = ExtensionConfigLoader(extensions_path)
            memory_config = loader.get_memory_config()
            
            assert memory_config is not None
            assert memory_config["enabled"] is True
            assert memory_config["topK"] == 5

    def test_loader_is_extension_enabled(self):
        """Test checking if extension is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions_path = Path(tmpdir) / "extensions.json"
            
            extensions_data = {
                "extensions": {
                    "memory": {"enabled": True},
                    "telemetry": {"enabled": False}
                }
            }
            
            with open(extensions_path, "w") as f:
                json.dump(extensions_data, f)
            
            loader = ExtensionConfigLoader(extensions_path)
            
            assert loader.is_extension_enabled("memory") is True
            assert loader.is_extension_enabled("telemetry") is False
            assert loader.is_extension_enabled("nonexistent") is False

    def test_loader_handles_missing_file(self):
        """Test that loader handles missing extensions file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions_path = Path(tmpdir) / "nonexistent.json"
            
            loader = ExtensionConfigLoader(extensions_path)
            config = loader.load()
            
            # Should return empty extensions structure
            assert config == {"extensions": {}}

    def test_loader_handles_invalid_json(self):
        """Test that loader handles invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extensions_path = Path(tmpdir) / "extensions.json"
            
            with open(extensions_path, "w") as f:
                f.write("{ invalid json")
            
            loader = ExtensionConfigLoader(extensions_path)
            config = loader.load()
            
            # Should return empty extensions structure
            assert config == {"extensions": {}}

    def test_get_extension_loader_singleton(self):
        """Test that get_extension_loader returns singleton instance."""
        loader1 = get_extension_loader()
        loader2 = get_extension_loader()
        
        assert loader1 is loader2


class TestConfigLoaderIntegration:
    """Test integration of migration with config loader."""

    def test_load_config_auto_migrates(self):
        """Test that load_config automatically migrates deprecated fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            extensions_path = Path(tmpdir) / "extensions.json"
            
            # Create old-style config with deprecated fields
            old_config = {
                "agents": {
                    "defaults": {
                        "workspace": "~/.nanobot/workspace",
                        "model": "gpt-4",
                        "maxTokens": 8192
                    }
                },
                "providers": {
                    "openai": {
                        "apiKey": "test-key"
                    }
                },
                "memory": {
                    "enabled": True,
                    "provider": "local",
                    "topK": 5
                },
                "telemetry": {
                    "enabled": True,
                    "port": 9090
                },
                "enable_quantum_latent": True,
                "use_keyring": False
            }
            
            with open(config_path, "w") as f:
                json.dump(old_config, f)
            
            # Load config - should trigger migration
            config = load_config(config_path, auto_migrate=True)
            
            # Verify core config is valid
            assert config.agents.defaults.model == "gpt-4"
            assert config.providers.openai.api_key == "test-key"
            
            # Verify extensions file was created
            assert extensions_path.exists()
            
            # Verify extensions content
            with open(extensions_path) as f:
                ext_data = json.load(f)
            
            assert "memory" in ext_data["extensions"]
            assert "telemetry" in ext_data["extensions"]
            assert "custom" in ext_data["extensions"]

    def test_load_config_without_auto_migrate(self):
        """Test that load_config can skip auto migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Create config without deprecated fields
            clean_config = {
                "agents": {
                    "defaults": {
                        "model": "gpt-4"
                    }
                }
            }
            
            with open(config_path, "w") as f:
                json.dump(clean_config, f)
            
            # Load config without migration
            config = load_config(config_path, auto_migrate=False)
            
            assert config.agents.defaults.model == "gpt-4"


class TestEndToEndMigration:
    """End-to-end migration tests."""

    def test_full_migration_workflow(self):
        """Test complete migration workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            extensions_path = Path(tmpdir) / "extensions.json"
            
            # Start with old config
            old_config = {
                "agents": {"defaults": {"model": "gpt-4"}},
                "providers": {"openai": {"apiKey": "sk-test"}},
                "gateway": {"host": "0.0.0.0", "port": 18790},
                "memory": {
                    "enabled": True,
                    "provider": "local",
                    "topK": 5
                },
                "translator": {
                    "enabled": True,
                    "autoSyncOnStartup": True
                },
                "telemetry": {
                    "enabled": True,
                    "port": 9090
                },
                "enable_quantum_latent": True,
                "use_keyring": True,
                "rate_limit_enabled": True,
                "rate_limit_max_calls": 10,
                "rate_limit_window_seconds": 60
            }
            
            with open(config_path, "w") as f:
                json.dump(old_config, f)
            
            # Run migration
            migrator = ConfigMigrator(config_path, extensions_path)
            core_config, extensions_config = migrator.migrate()
            migrator.save_extensions(extensions_config)
            migrator.save_core_config(core_config)
            
            # Verify both files exist
            assert config_path.exists()
            assert extensions_path.exists()
            assert (config_path.with_suffix(".json.bak")).exists()
            
            # Verify core config
            with open(config_path) as f:
                saved_core = json.load(f)
            
            assert "agents" in saved_core
            assert "providers" in saved_core
            assert "gateway" in saved_core
            assert "memory" not in saved_core
            assert "telemetry" not in saved_core
            
            # Verify extensions config
            with open(extensions_path) as f:
                saved_ext = json.load(f)
            
            assert "memory" in saved_ext["extensions"]
            assert "translator" in saved_ext["extensions"]
            assert "telemetry" in saved_ext["extensions"]
            assert "rate_limit" in saved_ext["extensions"]
            assert "custom" in saved_ext["extensions"]
            
            # Load with ExtensionConfigLoader
            loader = ExtensionConfigLoader(extensions_path)
            
            assert loader.get_memory_config()["enabled"] is True
            assert loader.get_telemetry_config()["port"] == 9090
            assert loader.is_extension_enabled("memory") is True
