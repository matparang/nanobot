"""Aggregate reasoning engine for count, reachable, and satisfaction queries.

Delegates all graph traversal to TransitiveReasoner so BFS logic lives in
exactly one place. Zero LLM usage.
"""

import logging
from typing import Optional, TYPE_CHECKING

from nanobot.memory.config_v2 import LOGGER_NAME, UNKNOWN_ENTITY_MESSAGE
from nanobot.memory.types_v2 import AggregateResult, RelationType, TruthValue

if TYPE_CHECKING:
    from nanobot.memory.relational_cache_v2 import RelationalCacheV2
    from nanobot.memory.transitive_reasoner import TransitiveReasoner

logger = logging.getLogger(LOGGER_NAME)


class AggregateEngine:
    """Aggregate reasoning over the relational graph.

    Provides count, reachable-set, satisfaction, and enumeration queries by
    delegating BFS to TransitiveReasoner.

    Attributes:
        cache: RelationalCacheV2 instance
        tr: TransitiveReasoner instance
    """

    def __init__(self, cache: "RelationalCacheV2", transitive_reasoner: "Optional[TransitiveReasoner]" = None):
        """Initialize the aggregate engine.

        Args:
            cache: RelationalCacheV2 instance
            transitive_reasoner: Optional TransitiveReasoner instance.
                If None, a new one is created from *cache*.
        """
        self.cache = cache
        if transitive_reasoner is None:
            from nanobot.memory.transitive_reasoner import TransitiveReasoner
            self.tr = TransitiveReasoner(cache)
        else:
            self.tr = transitive_reasoner
        logger.debug("AggregateEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reachable_from(
        self,
        entity: str,
        relation_type: RelationType,
    ) -> AggregateResult:
        """Return all entities reachable from *entity* via *relation_type*.

        Args:
            entity: Starting entity
            relation_type: Relation type to traverse

        Returns:
            AggregateResult with the reachable set
        """
        if entity not in self.cache.entities:
            return AggregateResult(
                value=TruthValue.UNKNOWN,
                count=0,
                entities=[],
                message=UNKNOWN_ENTITY_MESSAGE,
                source="entity_check",
            )

        reachable, paths = self.tr.all_reachable(entity, relation_type)
        return AggregateResult(
            value=TruthValue.TRUE if reachable else TruthValue.UNKNOWN,
            count=len(reachable),
            entities=reachable,
            message=(
                f"{entity} reaches {len(reachable)} entities via {relation_type.value}"
                if reachable
                else f"No entities reachable from {entity} via {relation_type.value}"
            ),
            source="reachable_from",
            details={"paths": {e: p for e, p in paths.items()}},
        )

    def count_reachable(
        self,
        entity: str,
        relation_type: RelationType,
    ) -> AggregateResult:
        """Return the count of entities reachable from *entity*.

        Args:
            entity: Starting entity
            relation_type: Relation type to traverse

        Returns:
            AggregateResult with count
        """
        result = self.reachable_from(entity, relation_type)
        return AggregateResult(
            value=result.value,
            count=result.count,
            entities=result.entities,
            message=f"Count: {result.count} entities reachable from {entity} "
                    f"via {relation_type.value}",
            source="count_reachable",
            details=result.details,
        )

    def who_satisfies(
        self,
        relation_type: RelationType,
        target: str,
    ) -> AggregateResult:
        """Find all X such that 'X relation_type target' holds.

        Uses the inverse relation to find all sources pointing to *target*.

        Args:
            relation_type: Relation type
            target: Target entity

        Returns:
            AggregateResult with all satisfying entities
        """
        if target not in self.cache.entities:
            return AggregateResult(
                value=TruthValue.UNKNOWN,
                count=0,
                entities=[],
                message=UNKNOWN_ENTITY_MESSAGE,
                source="entity_check",
            )

        # Use inverse: all X such that X→target means target is in
        # relations[(X, relation_type)]. We scan directly.
        satisfying: list[str] = []
        for entity in self.cache.entities:
            if entity == target:
                continue
            if target in self.cache.relations.get((entity, relation_type), set()):
                satisfying.append(entity)
            else:
                # Check transitively
                path = self.tr._bfs(entity, target, relation_type)
                if path:
                    satisfying.append(entity)

        satisfying = sorted(satisfying)
        return AggregateResult(
            value=TruthValue.TRUE if satisfying else TruthValue.UNKNOWN,
            count=len(satisfying),
            entities=satisfying,
            message=(
                f"{len(satisfying)} entities satisfy "
                f"X {relation_type.value} {target}: {satisfying}"
                if satisfying
                else f"No entities satisfy X {relation_type.value} {target}"
            ),
            source="who_satisfies",
        )

    def all_relations_of_type(
        self,
        relation_type: RelationType,
    ) -> AggregateResult:
        """Enumerate all direct (a, b) pairs for *relation_type*.

        Args:
            relation_type: Relation type to enumerate

        Returns:
            AggregateResult with pairs in details
        """
        pairs: list[tuple[str, str]] = []
        for (entity, rel), targets in self.cache.relations.items():
            if rel == relation_type:
                for target in sorted(targets):
                    pairs.append((entity, target))

        pairs.sort()
        entities = sorted({e for pair in pairs for e in pair})
        return AggregateResult(
            value=TruthValue.TRUE if pairs else TruthValue.UNKNOWN,
            count=len(pairs),
            entities=entities,
            message=f"{len(pairs)} direct {relation_type.value} pairs",
            source="all_relations",
            details={"pairs": pairs},
        )
