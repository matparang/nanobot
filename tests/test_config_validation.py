"""
Test configuration validation for the memory field issue.

This test demonstrates the problem described in the issue:
- Old configs with extra fields in 'memory' section cause Pydantic validation errors
- New example configs avoid these errors by only using valid schema fields
"""

import json
from pathlib import Path

from nanobot.config.loader import convert_keys
from nanobot.config.schema import Config, MemoryConfig


def test_memory_config_ignores_extra_fields():
    """Test that MemoryConfig ignores extra/unknown fields (Pydantic V2 default behavior)."""
    # By default, Pydantic V2 ignores extra fields rather than raising errors
    memory_data_with_extra = {
        "enabled": True,
        "provider": "local",
        "memory_type": "fractal",  # Extra field - will be ignored
        "vector_store": "chromadb",  # Extra field - will be ignored
    }

    # This should succeed - extra fields are simply ignored
    memory_config = MemoryConfig(**memory_data_with_extra)
    assert memory_config.enabled is True
    assert memory_config.provider == "local"

    # Extra fields should not be in the model
    assert not hasattr(memory_config, 'memory_type')
    assert not hasattr(memory_config, 'vector_store')


def test_memory_config_accepts_valid_fields():
    """Test that MemoryConfig accepts all valid fields from the schema."""
    valid_memory_data = {
        "enabled": True,
        "provider": "local",
        "top_k": 5,
        "archive_dir": "archives",
        "als_enabled": True,
        "mem0_api_key": "",
        "mem0_user_id": "test_user",
        "mem0_org_id": "",
        "mem0_project_id": "",
        "mem0_version": "v1.1",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "use_hybrid_search": True,
        "latent_retry_attempts": 3,
        "latent_retry_min_wait": 1.0,
        "latent_retry_max_wait": 5.0,
        "latent_retry_multiplier": 1.0,
    }

    # This should succeed
    memory_config = MemoryConfig(**valid_memory_data)
    assert memory_config.enabled is True
    assert memory_config.provider == "local"
    assert memory_config.top_k == 5
    assert memory_config.latent_retry_attempts == 3


def test_example_config_validates():
    """Test that config.example.json validates successfully."""
    example_path = Path(__file__).parent.parent / "config.example.json"

    # Load and validate
    with open(example_path) as f:
        data = json.load(f)

    # Convert camelCase to snake_case and validate
    config = Config.model_validate(convert_keys(data))

    # Verify core config is configured as expected (no deprecated fields)
    # Note: memory, translator, telemetry have been moved to extensions.json
    
    # Verify providers are set up
    assert config.providers.openai.api_key != ""
    assert config.providers.gemini.api_key != ""
    assert config.agents.defaults.enable_latent_reasoning is True


def test_minimal_config_validates():
    """Test that config.minimal.json validates successfully."""
    minimal_path = Path(__file__).parent.parent / "config.minimal.json"

    # Load and validate
    with open(minimal_path) as f:
        data = json.load(f)

    # Convert camelCase to snake_case and validate
    config = Config.model_validate(convert_keys(data))

    # Verify defaults are applied (memory has been moved to extensions)
    assert config.agents.defaults.workspace == "~/.nanobot/workspace"  # Default value
    assert config.agents.defaults.enable_latent_reasoning is True


def test_config_accepts_enable_latent_reasoning_override():
    """Test that enable_latent_reasoning can be disabled via config."""
    config = Config.model_validate(
        convert_keys({"agents": {"defaults": {"enableLatentReasoning": False}}})
    )
    assert config.agents.defaults.enable_latent_reasoning is False


def test_old_config_with_deprecated_fields_migrates():
    """Test that a config with deprecated fields gets migrated automatically."""
    old_config_data = {
        "agents": {
            "defaults": {
                "model": "gpt-4"
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
            "memory_type": "fractal",  # Extra field - will be ignored by migration
            "vector_store": "chromadb",  # Extra field - will be ignored by migration
            "als_enabled": True
        }
    }

    # Create a temp file to test migration
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(old_config_data, f)
        
        # Load config with auto-migration enabled
        from nanobot.config.loader import load_config
        config = load_config(config_path, auto_migrate=True)
        
        # Verify core config is valid (memory field has been migrated)
        assert config.agents.defaults.model == "gpt-4"
        assert config.providers.openai.api_key == "test-key"
        
        # Verify extensions file was created
        ext_path = config_path.parent / "extensions.json"
        assert ext_path.exists()


def test_extensions_example_validates():
    """Test that extensions.example.json has valid structure."""
    example_path = Path(__file__).parent.parent / "extensions.example.json"

    # Load and validate structure
    with open(example_path) as f:
        data = json.load(f)

    # Verify it has the expected structure
    assert "extensions" in data
    ext = data["extensions"]
    
    # Verify all expected extension sections exist
    assert "memory" in ext
    assert "translator" in ext
    assert "telemetry" in ext
    assert "rate_limit" in ext
    assert "custom" in ext
    
    # Verify memory extension has expected fields
    assert ext["memory"]["enabled"] is True
    assert ext["memory"]["provider"] == "local"
    assert ext["memory"]["topK"] == 5
    
    # Verify telemetry extension
    assert ext["telemetry"]["enabled"] is True
    assert ext["telemetry"]["port"] == 9090
    
    # Verify rate_limit extension
    assert ext["rate_limit"]["enabled"] is True
    assert ext["rate_limit"]["max_calls"] == 10
    assert ext["rate_limit"]["window_seconds"] == 60
    
    # Verify custom extension
    assert ext["custom"]["enable_quantum_latent"] is True
    assert ext["custom"]["use_keyring"] is True


def test_multi_provider_config():
    """Test that configs can have multiple providers configured."""
    multi_provider_config = {
        "agents": {
            "defaults": {
                "model": "gemini/gemini-2.0-flash-exp"
            }
        },
        "providers": {
            "openai": {
                "apiKey": "sk-openai-key"
            },
            "gemini": {
                "apiKey": "gemini-key"
            },
            "anthropic": {
                "apiKey": "sk-ant-key"
            },
            "deepseek": {
                "apiKey": "deepseek-key"
            }
        }
    }

    config = Config.model_validate(convert_keys(multi_provider_config))

    # Verify all providers are configured
    assert config.providers.openai.api_key == "sk-openai-key"
    assert config.providers.gemini.api_key == "gemini-key"
    assert config.providers.anthropic.api_key == "sk-ant-key"
    assert config.providers.deepseek.api_key == "deepseek-key"

    # Verify model selection
    assert config.agents.defaults.model == "gemini/gemini-2.0-flash-exp"
