#!/usr/bin/env python3
"""Standalone configuration migration script.

This script migrates deprecated configuration fields from config.json to extensions.json.
It creates a backup of the original config file before making changes.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate deprecated nanobot config fields to extensions.json"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json (default: ~/.nanobot/config.json)",
    )
    parser.add_argument(
        "--extensions",
        type=Path,
        help="Path to extensions.json (default: same directory as config.json)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a backup of config.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = Path.home() / ".nanobot" / "config.json"
    
    # Check if config exists
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Determine extensions path
    if args.extensions:
        extensions_path = args.extensions
    else:
        extensions_path = config_path.parent / "extensions.json"
    
    logger.info(f"Config path: {config_path}")
    logger.info(f"Extensions path: {extensions_path}")
    
    try:
        # Import here to avoid dependency issues if run standalone
        from nanobot.config.migration import ConfigMigrator
        
        # Create migrator
        migrator = ConfigMigrator(config_path, extensions_path)
        
        # Perform migration
        core_config, extensions_config = migrator.migrate()
        
        if not extensions_config.get("extensions"):
            logger.info("No deprecated fields found - nothing to migrate")
            return
        
        # Show what will be migrated
        logger.info("Deprecated fields found:")
        for key in extensions_config.get("extensions", {}):
            logger.info(f"  - {key}")
        
        if args.dry_run:
            logger.info("Dry run - no changes made")
            logger.info("Extensions that would be created:")
            print(json.dumps(extensions_config, indent=2))
            return
        
        # Create backup unless disabled
        if not args.no_backup:
            backup_path = config_path.with_suffix(".json.bak")
            if config_path.exists() and not backup_path.exists():
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
        
        # Save extensions
        migrator.save_extensions(extensions_config)
        
        # Save updated core config
        migrator.save_core_config(core_config)
        
        logger.info("Migration completed successfully!")
        logger.info(f"Core config saved to: {config_path}")
        logger.info(f"Extensions saved to: {extensions_path}")
        
        # Show summary
        print("\n" + "="*60)
        print("Migration Summary")
        print("="*60)
        print(f"✓ Migrated {len(extensions_config.get('extensions', {}))} extension(s)")
        print(f"✓ Core config updated: {config_path}")
        print(f"✓ Extensions config: {extensions_path}")
        if not args.no_backup:
            print(f"✓ Backup created: {config_path.with_suffix('.json.bak')}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
