"""Relational reasoning cache for pattern storage and cycle detection.

This module provides a pattern cache that stores relational reasoning patterns
discovered from session consolidation. It tracks entity relationships, supports
cycle detection, and maintains tallest/shortest statistics for graph analysis.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class RelationalCache:
    """Manages relational reasoning patterns and graph analysis.

    The cache stores patterns extracted from sessions, tracks entity relationships,
    detects cycles in relationship graphs, and maintains statistics like tallest/shortest
    paths for reasoning optimization.

    Attributes:
        workspace: Path to the workspace directory
        cache_file: Path to workspace/memory/relational_cache.yaml
    """

    SCHEMA_VERSION = 1

    def __init__(self, workspace: Path):
        """Initialize relational cache.

        Args:
            workspace: Path to workspace directory
        """
        self.workspace = Path(workspace)
        memory_dir = self.workspace / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = memory_dir / "relational_cache.yaml"

        # Initialize cache file if it doesn't exist
        if not self.cache_file.exists():
            self._write_cache({
                "schema_version": self.SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "patterns": [],
                "entities": {},
                "relationships": [],
                "statistics": {
                    "tallest": None,
                    "shortest": None,
                    "total_patterns": 0,
                    "total_relationships": 0
                }
            })

    def add_pattern(
        self,
        pattern_type: str,
        data: dict[str, Any],
        entities: list[str] | None = None
    ) -> None:
        """Add a reasoning pattern to the cache.

        Args:
            pattern_type: Type of pattern (e.g., 'inference', 'comparison', 'causation')
            data: Pattern data dictionary
            entities: Optional list of entity names involved in this pattern
        """
        cache = self._read_cache()

        pattern = {
            "id": len(cache["patterns"]) + 1,
            "type": pattern_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
            "entities": entities or []
        }

        cache["patterns"].append(pattern)
        cache["statistics"]["total_patterns"] = len(cache["patterns"])

        # Update entity tracking
        if entities:
            for entity in entities:
                if entity not in cache["entities"]:
                    cache["entities"][entity] = {
                        "first_seen": pattern["timestamp"],
                        "pattern_count": 0,
                        "attributes": {}
                    }
                cache["entities"][entity]["pattern_count"] += 1

        self._write_cache(cache)

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: dict[str, Any] | None = None
    ) -> None:
        """Add a relationship between entities.

        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Type of relationship (e.g., 'parent_of', 'taller_than')
            properties: Optional properties dictionary (e.g., {'height': 180})
        """
        cache = self._read_cache()

        relationship = {
            "source": source,
            "target": target,
            "type": relation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "properties": properties or {}
        }

        cache["relationships"].append(relationship)
        cache["statistics"]["total_relationships"] = len(cache["relationships"])

        # Ensure entities exist
        for entity in [source, target]:
            if entity not in cache["entities"]:
                cache["entities"][entity] = {
                    "first_seen": relationship["timestamp"],
                    "pattern_count": 0,
                    "attributes": {}
                }

        self._write_cache(cache)

    def update_entity_attribute(
        self,
        entity: str,
        attribute: str,
        value: Any
    ) -> None:
        """Update an attribute for an entity.

        Args:
            entity: Entity name
            attribute: Attribute name (e.g., 'height', 'age')
            value: Attribute value
        """
        cache = self._read_cache()

        if entity not in cache["entities"]:
            cache["entities"][entity] = {
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "pattern_count": 0,
                "attributes": {}
            }

        cache["entities"][entity]["attributes"][attribute] = value

        # Update tallest/shortest statistics if attribute is height
        if attribute == "height":
            self._update_height_statistics(cache, entity, value)

        self._write_cache(cache)

    def detect_cycles(self) -> list[list[str]]:
        """Detect cycles in the relationship graph.

        Returns:
            List of cycles, where each cycle is a list of entity names
        """
        cache = self._read_cache()

        # Build adjacency list
        graph = {}
        for rel in cache["relationships"]:
            source = rel["source"]
            target = rel["target"]
            if source not in graph:
                graph[source] = []
            graph[source].append(target)

        # DFS-based cycle detection
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

    def get_entity_relationships(
        self,
        entity: str,
        relation_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Get all relationships for an entity.

        Args:
            entity: Entity name
            relation_type: Optional filter by relationship type

        Returns:
            List of relationship dictionaries
        """
        cache = self._read_cache()

        relationships = [
            rel for rel in cache["relationships"]
            if rel["source"] == entity or rel["target"] == entity
        ]

        if relation_type:
            relationships = [
                rel for rel in relationships
                if rel["type"] == relation_type
            ]

        return relationships

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics including tallest/shortest entities.

        Returns:
            Statistics dictionary
        """
        cache = self._read_cache()
        return cache.get("statistics", {})

    def get_patterns(
        self,
        pattern_type: str | None = None,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get patterns from cache with optional filtering.

        Args:
            pattern_type: Optional filter by pattern type
            limit: Optional limit on number of patterns (most recent first)

        Returns:
            List of pattern dictionaries
        """
        cache = self._read_cache()
        patterns = cache.get("patterns", [])

        if pattern_type:
            patterns = [p for p in patterns if p["type"] == pattern_type]

        # Most recent first
        patterns = sorted(patterns, key=lambda x: x["timestamp"], reverse=True)

        if limit:
            patterns = patterns[:limit]

        return patterns

    def get_entities(self) -> dict[str, Any]:
        """Get all entities with their attributes.

        Returns:
            Dictionary mapping entity names to entity data
        """
        cache = self._read_cache()
        return cache.get("entities", {})

    def clear(self) -> None:
        """Clear all cache data (use with caution)."""
        self._write_cache({
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "patterns": [],
            "entities": {},
            "relationships": [],
            "statistics": {
                "tallest": None,
                "shortest": None,
                "total_patterns": 0,
                "total_relationships": 0
            }
        })

    def _update_height_statistics(
        self,
        cache: dict[str, Any],
        entity: str,
        height: float
    ) -> None:
        """Update tallest/shortest statistics based on height values."""
        stats = cache["statistics"]

        # Update tallest
        if stats["tallest"] is None or height > stats["tallest"].get("height", float('-inf')):
            stats["tallest"] = {"entity": entity, "height": height}

        # Update shortest
        if stats["shortest"] is None or height < stats["shortest"].get("height", float('inf')):
            stats["shortest"] = {"entity": entity, "height": height}

    def _read_cache(self) -> dict[str, Any]:
        """Read cache file safely."""
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _write_cache(self, data: dict[str, Any]) -> None:
        """Write cache file safely."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
