"""YAML-based persistence for RelationalCacheV2.

Saves and loads the knowledge graph to/from a YAML file. Zero new dependencies
(uses stdlib yaml via PyYAML which is already a transitive dependency).
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from nanobot.memory.config_v2 import LOGGER_NAME
from nanobot.memory.types_v2 import RelationType

if TYPE_CHECKING:
    from nanobot.memory.relational_cache_v2 import RelationalCacheV2

logger = logging.getLogger(LOGGER_NAME)

_GRAPH_FILE = "knowledge_graph.yaml"


class GraphPersistence:
    """YAML-based save/load for RelationalCacheV2.

    Attributes:
        path: Path to the YAML file
    """

    def __init__(self, workspace: Path):
        """Initialize persistence.

        Creates workspace/memory/ directory if needed.

        Args:
            workspace: Root workspace directory
        """
        mem_dir = workspace / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        self.path = mem_dir / _GRAPH_FILE
        logger.debug(f"GraphPersistence initialized at {self.path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, cache: "RelationalCacheV2") -> None:
        """Serialize *cache* to YAML.

        Args:
            cache: RelationalCacheV2 instance
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            logger.warning("PyYAML not available; skipping graph save")
            return

        data: dict = {
            "entities": sorted(cache.entities),
            "relations": [],
        }
        for (entity, rel_type), targets in cache.relations.items():
            for target in sorted(targets):
                source = cache.sources.get((entity, rel_type, target), "")
                data["relations"].append(
                    {
                        "a": entity,
                        "relation": rel_type.value,
                        "b": target,
                        "source": source,
                    }
                )

        with self.path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        logger.info(
            f"Graph saved to {self.path}: "
            f"{len(data['entities'])} entities, "
            f"{len(data['relations'])} relation edges"
        )

    def load(self, cache: "Optional[RelationalCacheV2]" = None) -> "RelationalCacheV2":
        """Deserialize from YAML into *cache*.

        Creates a new RelationalCacheV2 if *cache* is None.

        Args:
            cache: Optional RelationalCacheV2 to populate (creates new if None)

        Returns:
            Populated RelationalCacheV2 instance
        """
        from nanobot.memory.relational_cache_v2 import RelationalCacheV2

        if cache is None:
            cache = RelationalCacheV2()

        if not self.path.exists():
            logger.debug(f"No graph file at {self.path}; returning empty cache")
            return cache

        try:
            import yaml  # type: ignore[import]
        except ImportError:
            logger.warning("PyYAML not available; returning empty cache")
            return cache

        with self.path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        for row in data.get("relations", []):
            try:
                rel_type = RelationType(row["relation"])
                cache.add_relation(
                    row["a"],
                    rel_type,
                    row["b"],
                    source=row.get("source") or None,
                )
            except (KeyError, ValueError) as exc:
                logger.warning(f"Skipping malformed relation row {row}: {exc}")

        logger.info(
            f"Graph loaded from {self.path}: "
            f"{len(cache.entities)} entities"
        )
        return cache
