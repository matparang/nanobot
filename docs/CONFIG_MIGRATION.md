# Configuration Migration Guide

## Overview

Starting from version 0.2.0, nanobot has introduced a new configuration architecture that separates core configuration from extension configurations. This guide explains the deprecated fields and how to migrate your existing configuration.

## Deprecated Fields

The following fields have been moved from `config.json` to `extensions.json`:

### Extension-Specific Fields

1. **`memory`** - Memory/fractal configuration
   - **New location**: `extensions.memory`
   - **Contains**: Memory provider settings, vector embeddings, ALS configuration

2. **`translator`** - Triune Memory Translator configuration
   - **New location**: `extensions.translator`
   - **Contains**: Translation settings, auto-sync options

3. **`telemetry`** - Telemetry exporter configuration
   - **New location**: `extensions.telemetry`
   - **Contains**: Telemetry settings, port configuration

### Rate Limiting Fields

The following rate limiting fields have been consolidated:

- `rate_limit_enabled` → `extensions.rate_limit.enabled`
- `rate_limit_max_calls` → `extensions.rate_limit.max_calls`
- `rate_limit_window_seconds` → `extensions.rate_limit.window_seconds`

### Custom/Experimental Fields

- `enable_quantum_latent` → `extensions.custom.enable_quantum_latent`
- `use_keyring` → `extensions.custom.use_keyring`

## Migration Process

### Automatic Migration

Nanobot automatically migrates deprecated fields when loading your configuration. When you start nanobot with an old config file:

1. Deprecated fields are detected
2. An `extensions.json` file is created in the same directory as `config.json`
3. Deprecated fields are moved to `extensions.json`
4. Your `config.json` is updated to remove the deprecated fields
5. A backup `config.json.bak` is created (only on first migration)

**Example:**

```bash
# Just start nanobot normally - migration happens automatically
nanobot start
```

### Manual Migration

You can also manually migrate your configuration using the migration script:

```bash
# Migrate with default paths (~/.nanobot/config.json)
python scripts/migrate_config.py

# Specify custom paths
python scripts/migrate_config.py --config /path/to/config.json --extensions /path/to/extensions.json

# Dry run - see what would be migrated without making changes
python scripts/migrate_config.py --dry-run

# Skip backup creation
python scripts/migrate_config.py --no-backup

# Verbose output
python scripts/migrate_config.py -v
```

## Extension Config Structure

After migration, your `extensions.json` will have the following structure:

```json
{
  "extensions": {
    "memory": {
      "enabled": true,
      "provider": "local",
      "topK": 5,
      "archiveDir": "archives",
      "alsEnabled": true,
      "mem0ApiKey": "",
      "mem0UserId": "nanobot_user",
      "mem0OrgId": "",
      "mem0ProjectId": "",
      "mem0Version": "v1.1",
      "embeddingModel": "text-embedding-3-small",
      "embeddingDim": 1536,
      "useHybridSearch": true
    },
    "translator": {
      "enabled": true,
      "autoSyncOnStartup": true,
      "watchForChanges": false,
      "syncDirection": "md_to_yaml",
      "excludedFiles": ["README.md", "CHANGELOG.md"]
    },
    "telemetry": {
      "enabled": true,
      "port": 9090
    },
    "rate_limit": {
      "enabled": true,
      "max_calls": 10,
      "window_seconds": 60
    },
    "custom": {
      "enable_quantum_latent": true,
      "use_keyring": true
    }
  }
}
```

## Using Extension Configs in Code

### Loading Extension Configurations

```python
from nanobot.config import get_extension_loader

# Get the extension loader
loader = get_extension_loader()

# Load all extensions
loader.load()

# Get specific extension config
memory_config = loader.get_memory_config()
telemetry_config = loader.get_telemetry_config()
translator_config = loader.get_translator_config()
rate_limit_config = loader.get_rate_limit_config()
custom_config = loader.get_custom_config()

# Check if extension is enabled
if loader.is_extension_enabled("memory"):
    print("Memory extension is enabled")

# Get all extensions
all_extensions = loader.get_all_extensions()
```

### Example: Memory Extension

```python
from nanobot.config import get_extension_loader

loader = get_extension_loader()
memory_config = loader.get_memory_config()

if memory_config and memory_config.get("enabled"):
    provider = memory_config.get("provider", "local")
    top_k = memory_config.get("topK", 5)
    print(f"Memory provider: {provider}, topK: {top_k}")
```

## Backward Compatibility

- Old `config.json` files will continue to work
- Migration happens automatically on first load
- Original config is backed up as `config.json.bak`
- No manual intervention required for existing deployments

## Core Config Structure

After migration, your `config.json` will only contain core fields:

```json
{
  "agents": { ... },
  "channels": { ... },
  "providers": { ... },
  "gateway": { ... },
  "tools": { ... }
}
```

## Troubleshooting

### Migration Not Happening

**Problem**: Deprecated fields still in config.json, no extensions.json created

**Solution**:
1. Check file permissions - ensure nanobot can write to the config directory
2. Run manual migration: `python scripts/migrate_config.py -v`
3. Check logs for error messages

### Validation Errors After Migration

**Problem**: Getting Pydantic validation errors after migration

**Solution**:
1. Check that `extensions.json` was created successfully
2. Verify the JSON syntax is valid: `python -m json.tool extensions.json`
3. Ensure all required fields are present in the core config

### Extension Not Loading

**Problem**: Extension configuration not being used by the application

**Solution**:
1. Verify `extensions.json` exists in the same directory as `config.json`
2. Check the extension is enabled: `"enabled": true`
3. Ensure the application code is using `get_extension_loader()` to access configs

### Lost Configuration After Migration

**Problem**: Configuration values missing after migration

**Solution**:
1. Check the backup file: `config.json.bak`
2. Verify both `config.json` and `extensions.json` have the expected values
3. Re-run migration with `--dry-run` to see what would be migrated

## Migration Checklist

Before migrating:
- [ ] Backup your current `config.json` manually (extra safety)
- [ ] Note which deprecated fields you're using
- [ ] Review this migration guide

After migration:
- [ ] Verify `extensions.json` was created
- [ ] Check that deprecated fields are in `extensions.json`
- [ ] Confirm core `config.json` only has supported fields
- [ ] Test that your application starts successfully
- [ ] Verify extensions are working as expected

## Examples

### Example 1: Old Config (Before Migration)

```json
{
  "agents": {
    "defaults": {
      "model": "gpt-4"
    }
  },
  "memory": {
    "enabled": true,
    "provider": "local",
    "topK": 5
  },
  "telemetry": {
    "enabled": true,
    "port": 9090
  },
  "enable_quantum_latent": true,
  "rate_limit_enabled": true,
  "rate_limit_max_calls": 10,
  "rate_limit_window_seconds": 60,
  "use_keyring": true
}
```

### Example 2: After Migration

**config.json:**
```json
{
  "agents": {
    "defaults": {
      "model": "gpt-4"
    }
  }
}
```

**extensions.json:**
```json
{
  "extensions": {
    "memory": {
      "enabled": true,
      "provider": "local",
      "topK": 5
    },
    "telemetry": {
      "enabled": true,
      "port": 9090
    },
    "rate_limit": {
      "enabled": true,
      "max_calls": 10,
      "window_seconds": 60
    },
    "custom": {
      "enable_quantum_latent": true,
      "use_keyring": true
    }
  }
}
```

## Support

If you encounter issues with migration:

1. Check the [GitHub Issues](https://github.com/AINAF-OpenFramework/nanobot-modd/issues)
2. Run migration with verbose logging: `python scripts/migrate_config.py -v`
3. Create a new issue with migration logs if the problem persists

## Version History

- **v0.2.0** - Initial introduction of extension configuration system
- **v0.1.x** - Legacy configuration with all fields in config.json
