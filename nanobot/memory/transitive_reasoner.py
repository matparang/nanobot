"""Transitive inference engine for deterministic relational reasoning.

Owns all pairwise inference logic: direct lookups, multi-hop BFS, opposite-direction
proofs, and cycle-safety guards. Zero LLM usage.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from nanobot.memory.config_v2 import (
    LOGGER_NAME,
    UNKNOWN_ENTITY_MESSAGE,
    UNKNOWN_PAIRWISE_MESSAGE,
    UNKNOWN_SELF_COMPARISON_MESSAGE,
)
from nanobot.memory.types_v2 import RelationType, TruthValue

if TYPE_CHECKING:
    from nanobot.memory.relational_cache_v2 import RelationalCacheV2

logger = logging.getLogger(LOGGER_NAME)


@dataclass
class TransitiveResult:
    """Result of a transitive pairwise query.

    Attributes:
        value: Truth value (TRUE/FALSE/UNKNOWN)
        message: Human-readable explanation
        source: Source of the result (e.g., "direct", "transitive", "opposite_provable")
        path: Proof path as list of entity names
        depth: Number of hops (0 for UNKNOWN, 1 for direct, n for n-hop)
        cycle_detected: Whether a cycle was found in the graph
    """
    value: TruthValue
    message: str
    source: str
    path: list[str] = field(default_factory=list)
    depth: int = 0
    cycle_detected: bool = False


class TransitiveReasoner:
    """Dedicated transitive inference engine.

    Delegates all graph storage to RelationalCacheV2; owns BFS, cycle detection,
    and all proof-path generation.

    Attributes:
        cache: RelationalCacheV2 instance used as the underlying graph store
        max_depth: Maximum BFS hops before giving up
    """

    def __init__(self, cache: "RelationalCacheV2", max_depth: int = 100):
        """Initialize the transitive reasoner.

        Args:
            cache: RelationalCacheV2 instance
            max_depth: Maximum BFS depth (default: 100)
        """
        self.cache = cache
        self.max_depth = max_depth
        logger.debug(f"TransitiveReasoner initialized with max_depth={max_depth}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        a: str,
        relation_type: RelationType,
        b: str,
    ) -> TransitiveResult:
        """Query whether 'a relation_type b' holds.

        Decision tree:
        1. Self-comparison → UNKNOWN
        2. Unknown entity → UNKNOWN
        3. Cycle detected → flag but still attempt answer
        4. Direct edge a→b → TRUE (depth=1, path=[a,b])
        5. BFS path a→…→b → TRUE (depth=len-1, full path)
        6. BFS path b→…→a → FALSE (opposite provable)
        7. Otherwise → UNKNOWN

        Args:
            a: Source entity
            relation_type: Relation type
            b: Target entity

        Returns:
            TransitiveResult with truth value and proof
        """
        # 1. Self-comparison
        if a == b:
            return TransitiveResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SELF_COMPARISON_MESSAGE,
                source="self_check",
            )

        # 2. Unknown entity
        if a not in self.cache.entities or b not in self.cache.entities:
            return TransitiveResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_ENTITY_MESSAGE,
                source="entity_check",
            )

        # 3. Cycle detection (flag only; still attempt)
        cycle_detected = self.has_cycle(relation_type)
        if cycle_detected:
            logger.warning(
                f"Cycle detected in {relation_type.value} graph while querying "
                f"{a} → {b}"
            )

        # 4. Direct edge
        if b in self.cache.relations.get((a, relation_type), set()):
            return TransitiveResult(
                value=TruthValue.TRUE,
                message=f"{a} {relation_type.value} {b}",
                source="direct",
                path=[a, b],
                depth=1,
                cycle_detected=cycle_detected,
            )

        # 5. BFS forward
        path = self._bfs(a, b, relation_type)
        if path:
            return TransitiveResult(
                value=TruthValue.TRUE,
                message=f"{a} {relation_type.value} {b} (transitive)",
                source="transitive",
                path=path,
                depth=len(path) - 1,
                cycle_detected=cycle_detected,
            )

        # 6. BFS opposite → FALSE
        opposite_path = self._bfs(b, a, relation_type)
        if opposite_path:
            return TransitiveResult(
                value=TruthValue.FALSE,
                message=f"NOT {a} {relation_type.value} {b} (opposite provable)",
                source="opposite_provable",
                path=opposite_path,
                depth=len(opposite_path) - 1,
                cycle_detected=cycle_detected,
            )

        # 7. No information
        return TransitiveResult(
            value=TruthValue.UNKNOWN,
            message=UNKNOWN_PAIRWISE_MESSAGE,
            source="no_information",
            cycle_detected=cycle_detected,
        )

    def all_reachable(
        self,
        entity: str,
        relation_type: RelationType,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Return all entities reachable from *entity* via *relation_type*.

        Args:
            entity: Starting entity
            relation_type: Relation type to traverse

        Returns:
            Tuple of (sorted list of reachable entities, {entity: proof_path})
        """
        if entity not in self.cache.entities:
            return [], {}

        visited: dict[str, list[str]] = {}  # entity → path from start
        queue: deque[tuple[str, list[str]]] = deque([(entity, [entity])])
        seen = {entity}

        while queue:
            current, path = queue.popleft()
            for neighbor in self.cache.relations.get((current, relation_type), set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    new_path = path + [neighbor]
                    visited[neighbor] = new_path
                    queue.append((neighbor, new_path))

        return sorted(visited.keys()), visited

    def has_cycle(self, relation_type: RelationType) -> bool:
        """Return True if the graph for *relation_type* contains a cycle.

        Args:
            relation_type: Relation type to check

        Returns:
            True if a cycle exists
        """
        return self.cache._has_cycle(relation_type)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bfs(
        self,
        start: str,
        target: str,
        relation_type: RelationType,
    ) -> Optional[list[str]]:
        """Shortest-path BFS from *start* to *target*.

        Args:
            start: Starting entity
            target: Target entity
            relation_type: Relation type to traverse

        Returns:
            Path as list of entities if found, None otherwise
        """
        if start == target:
            return [start]

        visited = {start}
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= self.max_depth:
                continue

            for neighbor in self.cache.relations.get((current, relation_type), set()):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None
