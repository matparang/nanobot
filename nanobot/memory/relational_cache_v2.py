"""Deterministic relational cache with explicit inverse storage and transitive reasoning.

This module provides RelationalCacheV2 that maintains a graph of entity relationships
with bidirectional storage, transitive reasoning via graph reachability, and tiered ranking.
"""

import logging
from collections import defaultdict, deque
from typing import Optional, Set

from nanobot.memory.config_v2 import (
    LOGGER_NAME,
    UNKNOWN_ENTITY_MESSAGE,
    UNKNOWN_INCONSISTENT_MESSAGE,
    UNKNOWN_PAIRWISE_MESSAGE,
    UNKNOWN_SELF_COMPARISON_MESSAGE,
)
from nanobot.memory.types_v2 import QueryResult, RelationType, TruthValue

logger = logging.getLogger(LOGGER_NAME)


class RelationalCacheV2:
    """Deterministic relational cache with explicit inverse storage.
    
    This cache stores entity relationships as a directed graph and supports:
    - Explicit storage of both direct and inverse relations
    - Deterministic transitive reasoning via graph reachability
    - Tiered ranking (partial order levels)
    - Cycle detection for inconsistency checking
    - Strict UNKNOWN handling for self-comparisons and unknown entities
    
    Attributes:
        entities: Set of known entities
        relations: Adjacency list mapping (entity, relation_type) -> set of targets
        sources: Metadata about relation sources (for debugging)
    """
    
    def __init__(self):
        """Initialize an empty relational cache."""
        self.entities: Set[str] = set()
        # relations[(entity, relation_type)] = set of targets
        self.relations: dict[tuple[str, RelationType], Set[str]] = defaultdict(set)
        # Track sources for debugging
        self.sources: dict[tuple[str, RelationType, str], str] = {}
        logger.debug("RelationalCacheV2 initialized")
    
    def add_relation(
        self,
        a: str,
        relation_type: RelationType,
        b: str,
        confidence: Optional[float] = None,
        source: Optional[str] = None
    ) -> None:
        """Add a relation and its inverse to the cache.
        
        Args:
            a: Source entity
            relation_type: Type of relation
            b: Target entity
            confidence: Optional confidence score (not used for storage, informational only)
            source: Optional source identifier for debugging
        """
        # Add entities to known set
        self.entities.add(a)
        self.entities.add(b)
        
        # Add direct relation
        self.relations[(a, relation_type)].add(b)
        if source:
            self.sources[(a, relation_type, b)] = source
        
        # Add inverse relation
        inverse_type = self._get_inverse(relation_type)
        self.relations[(b, inverse_type)].add(a)
        if source:
            self.sources[(b, inverse_type, a)] = f"{source} (inverse)"
        
        logger.debug(
            f"Added relation: {a} {relation_type.value} {b} "
            f"(and inverse {b} {inverse_type.value} {a})"
        )
    
    def query_relation(
        self,
        a: str,
        relation_type: RelationType,
        b: str
    ) -> QueryResult:
        """Query whether a relation holds between two entities.
        
        Uses deterministic transitive reasoning:
        - Direct relation -> TRUE
        - Provable via transitive closure -> TRUE
        - Opposite relation provable -> FALSE
        - Self-comparison -> UNKNOWN
        - Unknown entities -> UNKNOWN
        - Otherwise -> UNKNOWN
        
        Args:
            a: Source entity
            relation_type: Type of relation to query
            b: Target entity
            
        Returns:
            QueryResult with truth value, message, and source
        """
        # Self-comparison always UNKNOWN
        if a == b:
            logger.debug(f"Query {a} {relation_type.value} {b}: self-comparison -> UNKNOWN")
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SELF_COMPARISON_MESSAGE,
                source="self_check"
            )
        
        # Unknown entities -> UNKNOWN
        if a not in self.entities or b not in self.entities:
            logger.debug(
                f"Query {a} {relation_type.value} {b}: unknown entity -> UNKNOWN"
            )
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_ENTITY_MESSAGE,
                source="entity_check",
                details={"known_entities": list(self.entities)}
            )
        
        # Check direct relation
        if b in self.relations.get((a, relation_type), set()):
            logger.debug(f"Query {a} {relation_type.value} {b}: direct -> TRUE")
            return QueryResult(
                value=TruthValue.TRUE,
                message=f"{a} {relation_type.value} {b}",
                source="direct"
            )
        
        # Check transitive relation via BFS
        path = self._find_path_bfs(a, b, relation_type)
        if path:
            logger.debug(
                f"Query {a} {relation_type.value} {b}: transitive via {path} -> TRUE"
            )
            return QueryResult(
                value=TruthValue.TRUE,
                message=f"{a} {relation_type.value} {b} (transitive)",
                source="transitive",
                details={"path": path}
            )
        
        # Check if opposite relation is provable (makes this FALSE)
        inverse_type = self._get_inverse(relation_type)
        inverse_path = self._find_path_bfs(b, a, inverse_type)
        if inverse_path:
            logger.debug(
                f"Query {a} {relation_type.value} {b}: opposite provable -> FALSE"
            )
            return QueryResult(
                value=TruthValue.FALSE,
                message=f"NOT {a} {relation_type.value} {b} (opposite provable)",
                source="opposite_provable",
                details={"opposite_path": inverse_path}
            )
        
        # No information -> UNKNOWN
        logger.debug(f"Query {a} {relation_type.value} {b}: no info -> UNKNOWN")
        return QueryResult(
            value=TruthValue.UNKNOWN,
            message=UNKNOWN_PAIRWISE_MESSAGE,
            source="no_information"
        )
    
    def rank_entities(
        self,
        relation_type: RelationType
    ) -> QueryResult:
        """Rank entities by relation type using tiered ranking.
        
        Returns partial order levels (tiers) where entities in the same tier
        are incomparable. Detects cycles and returns UNKNOWN if inconsistent.
        
        Args:
            relation_type: Relation type to use for ranking
            
        Returns:
            QueryResult with tiered ranking or UNKNOWN if inconsistent
        """
        # Check for cycles first
        if self._has_cycle(relation_type):
            logger.warning(f"Ranking for {relation_type.value}: cycles detected -> UNKNOWN")
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_INCONSISTENT_MESSAGE,
                source="cycle_detection"
            )
        
        # Perform topological sort with levels (Kahn's algorithm variant)
        tiers = self._compute_tiers(relation_type)
        
        if not tiers:
            # No entities with this relation type
            logger.debug(f"Ranking for {relation_type.value}: no entities")
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message="UNKNOWN: No entities to rank",
                source="no_entities"
            )
        
        # Stable sort within each tier
        stable_tiers = [sorted(tier) for tier in tiers]
        
        logger.debug(
            f"Ranking for {relation_type.value}: {len(stable_tiers)} tiers"
        )
        return QueryResult(
            value=TruthValue.TRUE,
            message=f"Tiered ranking by {relation_type.value}",
            source="tiered_ranking",
            details={"tiers": stable_tiers}
        )
    
    def _get_inverse(self, relation_type: RelationType) -> RelationType:
        """Get the inverse of a relation type."""
        inverses = {
            RelationType.TALLER_THAN: RelationType.SHORTER_THAN,
            RelationType.SHORTER_THAN: RelationType.TALLER_THAN,
        }
        return inverses[relation_type]
    
    def _find_path_bfs(
        self,
        start: str,
        target: str,
        relation_type: RelationType
    ) -> Optional[list[str]]:
        """Find path from start to target using BFS.
        
        Returns:
            Path as list of entities if found, None otherwise
        """
        if start == target:
            return [start]
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            # Get neighbors via this relation type
            neighbors = self.relations.get((current, relation_type), set())
            
            for neighbor in neighbors:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _has_cycle(self, relation_type: RelationType) -> bool:
        """Check if graph has cycles for given relation type.
        
        Uses DFS-based cycle detection.
        """
        # Build adjacency list for this relation type
        graph = defaultdict(set)
        all_nodes = set()
        
        for (entity, rel_type), targets in self.relations.items():
            if rel_type == relation_type:
                graph[entity].update(targets)
                all_nodes.add(entity)
                all_nodes.update(targets)
        
        visited = set()
        rec_stack = set()
        
        def has_cycle_dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle_dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in all_nodes:
            if node not in visited:
                if has_cycle_dfs(node):
                    return True
        
        return False
    
    def _compute_tiers(self, relation_type: RelationType) -> list[list[str]]:
        """Compute tiered ranking using topological sort levels.
        
        Returns list of tiers, where tier 0 has no predecessors,
        tier 1 has predecessors only from tier 0, etc.
        """
        # Build graph and compute in-degrees
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for (entity, rel_type), targets in self.relations.items():
            if rel_type == relation_type:
                graph[entity].update(targets)
                all_nodes.add(entity)
                for target in targets:
                    in_degree[target] += 1
                    all_nodes.add(target)
        
        # Ensure all nodes are in in_degree
        for node in all_nodes:
            if node not in in_degree:
                in_degree[node] = 0
        
        # Kahn's algorithm with levels
        tiers = []
        current_tier = [node for node in all_nodes if in_degree[node] == 0]
        
        while current_tier:
            tiers.append(current_tier[:])
            next_tier = []
            
            for node in current_tier:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_tier.append(neighbor)
            
            current_tier = next_tier
        
        return tiers
