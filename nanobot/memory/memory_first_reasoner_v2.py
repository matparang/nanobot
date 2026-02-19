"""Memory-first reasoner with strict no-LLM fallback policy.

This module provides a query interface for relational reasoning that ONLY uses
the relational cache, with no LLM fallback for unknown cases.
"""

import logging
from typing import Optional

from nanobot.memory.config_v2 import (
    LOGGER_NAME,
    SUPERLATIVE_POLICY,
    UNKNOWN_SUPERLATIVE_MESSAGE,
)
from nanobot.memory.relational_cache_v2 import RelationalCacheV2
from nanobot.memory.types_v2 import QueryResult, RelationType, TruthValue

logger = logging.getLogger(LOGGER_NAME)


class MemoryFirstReasonerV2:
    """Memory-first reasoner with no LLM fallback.
    
    This reasoner provides a query interface for relational reasoning that
    strictly enforces memory-first policy: all answers come from the cache,
    and UNKNOWN is returned when information is insufficient.
    
    Attributes:
        cache: RelationalCacheV2 instance
        superlative_policy: Policy for superlative queries
    """
    
    def __init__(
        self,
        cache: Optional[RelationalCacheV2] = None,
        superlative_policy: str = SUPERLATIVE_POLICY
    ):
        """Initialize memory-first reasoner.
        
        Args:
            cache: Optional RelationalCacheV2 instance (creates new one if None)
            superlative_policy: Policy for superlative queries (default: "unknown_if_not_unique")
        """
        self.cache = cache if cache is not None else RelationalCacheV2()
        self.superlative_policy = superlative_policy
        logger.info(
            f"MemoryFirstReasonerV2 initialized with superlative_policy={superlative_policy}"
        )
    
    def query_pairwise(
        self,
        a: str,
        b: str,
        relation_type: RelationType = RelationType.TALLER_THAN
    ) -> QueryResult:
        """Query pairwise relation between two entities.
        
        Args:
            a: First entity
            b: Second entity
            relation_type: Type of relation to query (default: TALLER_THAN)
            
        Returns:
            QueryResult from cache
        """
        logger.debug(f"Pairwise query: {a} {relation_type.value} {b}")
        result = self.cache.query_relation(a, relation_type, b)
        logger.info(
            f"Pairwise result: {a} {relation_type.value} {b} -> {result.value.value} "
            f"(source={result.source})"
        )
        return result
    
    def query_superlative(self, kind: str) -> QueryResult:
        """Query for superlative entity (tallest or shortest).
        
        Returns UNKNOWN if not uniquely determined (multiple candidates at same level).
        
        Args:
            kind: "tallest" or "shortest"
            
        Returns:
            QueryResult with superlative entity or UNKNOWN
        """
        logger.debug(f"Superlative query: {kind}")
        
        if kind not in ["tallest", "shortest"]:
            logger.warning(f"Invalid superlative kind: {kind}")
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message=f"UNKNOWN: Invalid kind '{kind}'",
                source="invalid_kind"
            )
        
        # Determine which relation type to use for ranking
        if kind == "tallest":
            relation_type = RelationType.TALLER_THAN
            # Tallest is in the first tier (no one taller)
            tier_index = 0
        else:  # shortest
            relation_type = RelationType.SHORTER_THAN
            # Shortest is in the first tier (no one shorter)
            tier_index = 0
        
        # Get tiered ranking
        ranking_result = self.cache.rank_entities(relation_type)
        
        if ranking_result.value != TruthValue.TRUE:
            # Ranking failed (cycles or no entities)
            logger.info(f"Superlative {kind}: ranking failed -> UNKNOWN")
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="ranking_failed"
            )
        
        tiers = ranking_result.details.get("tiers", [])
        
        if not tiers or tier_index >= len(tiers):
            logger.info(f"Superlative {kind}: no entities in tier {tier_index} -> UNKNOWN")
            return QueryResult(
                value=TruthValue.UNKNOWN,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="no_tier"
            )
        
        candidates = tiers[tier_index]
        
        # Check uniqueness based on policy
        if self.superlative_policy == "unknown_if_not_unique":
            if len(candidates) > 1:
                logger.info(
                    f"Superlative {kind}: multiple candidates {candidates} -> UNKNOWN"
                )
                return QueryResult(
                    value=TruthValue.UNKNOWN,
                    message=UNKNOWN_SUPERLATIVE_MESSAGE,
                    source="ambiguous",
                    details={"candidates": candidates}
                )
        
        # Unique candidate (or policy allows non-unique)
        superlative_entity = candidates[0] if len(candidates) == 1 else candidates
        logger.info(f"Superlative {kind}: {superlative_entity}")
        
        if len(candidates) == 1:
            return QueryResult(
                value=TruthValue.TRUE,
                message=f"{superlative_entity} is {kind}",
                source="superlative",
                details={"entity": superlative_entity, "kind": kind}
            )
        else:
            # Multiple candidates but policy allows it
            return QueryResult(
                value=TruthValue.TRUE,
                message=f"Multiple {kind}: {candidates}",
                source="superlative_multiple",
                details={"entities": candidates, "kind": kind}
            )
    
    def rank_all_entities(
        self,
        relation_type: RelationType = RelationType.TALLER_THAN
    ) -> QueryResult:
        """Rank all entities using tiered ranking.
        
        Args:
            relation_type: Relation type to use for ranking (default: TALLER_THAN)
            
        Returns:
            QueryResult with tiered ranking
        """
        logger.debug(f"Ranking query: {relation_type.value}")
        result = self.cache.rank_entities(relation_type)
        logger.info(
            f"Ranking result: {relation_type.value} -> {result.value.value} "
            f"(source={result.source})"
        )
        return result
