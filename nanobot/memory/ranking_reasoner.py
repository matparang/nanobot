"""Ranking and superlative reasoning engine for deterministic relational reasoning.

Owns all tiered partial-order ranking and superlative (tallest/shortest/fastest/…)
logic. Zero LLM usage.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from nanobot.memory.config_v2 import (
    LOGGER_NAME,
    SUPERLATIVE_POLICY,
    UNKNOWN_INCONSISTENT_MESSAGE,
    UNKNOWN_SUPERLATIVE_MESSAGE,
)
from nanobot.memory.types_v2 import RelationType, TruthValue

if TYPE_CHECKING:
    from nanobot.memory.relational_cache_v2 import RelationalCacheV2

logger = logging.getLogger(LOGGER_NAME)

# Maps superlative keywords to (relation_type, tier_index).
# tier_index=0 means "first in the ranking" (no predecessor at that tier).
SUPERLATIVE_MAP: dict[str, tuple[RelationType, int]] = {
    "tallest": (RelationType.TALLER_THAN, 0),
    "shortest": (RelationType.SHORTER_THAN, 0),
    "fastest": (RelationType.FASTER_THAN, 0),
    "slowest": (RelationType.SLOWER_THAN, 0),
    "greatest": (RelationType.GREATER_THAN, 0),
    "least": (RelationType.LESS_THAN, 0),
}


@dataclass
class RankingResult:
    """Result of a ranking or superlative query.

    Attributes:
        value: Truth value (TRUE/FALSE/UNKNOWN)
        tiers: List of tiers (each tier is a sorted list of entity names)
        superlative_entity: Unique superlative entity name (if applicable)
        candidates: All candidate entities at the superlative tier
        total_entities: Total entities in the ranking
        message: Human-readable explanation
        source: Source of the result
        is_consistent: Whether the graph is cycle-free
    """
    value: TruthValue
    tiers: list[list[str]]
    superlative_entity: Optional[str]
    candidates: list[str]
    total_entities: int
    message: str
    source: str
    is_consistent: bool = True


class RankingReasoner:
    """Dedicated ranking and superlative reasoning engine.

    Uses Kahn's algorithm to produce tiered partial orders, and enforces the
    configured superlative uniqueness policy.

    Attributes:
        cache: RelationalCacheV2 instance
        superlative_policy: Policy for superlative queries
    """

    def __init__(self, cache: "RelationalCacheV2", superlative_policy: str = SUPERLATIVE_POLICY):
        """Initialize the ranking reasoner.

        Args:
            cache: RelationalCacheV2 instance
            superlative_policy: "unknown_if_not_unique" (default) or "first"
        """
        self.cache = cache
        self.superlative_policy = superlative_policy
        logger.debug(
            f"RankingReasoner initialized with superlative_policy={superlative_policy}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, relation_type: RelationType) -> RankingResult:
        """Produce a full tiered ranking for *relation_type*.

        Returns UNKNOWN with is_consistent=False if a cycle is detected.

        Args:
            relation_type: Relation type to rank by

        Returns:
            RankingResult with tiers or UNKNOWN
        """
        if not self.is_consistent(relation_type):
            return RankingResult(
                value=TruthValue.UNKNOWN,
                tiers=[],
                superlative_entity=None,
                candidates=[],
                total_entities=0,
                message=UNKNOWN_INCONSISTENT_MESSAGE,
                source="cycle_detection",
                is_consistent=False,
            )

        tiers = self.cache._compute_tiers(relation_type)
        if not tiers:
            return RankingResult(
                value=TruthValue.UNKNOWN,
                tiers=[],
                superlative_entity=None,
                candidates=[],
                total_entities=0,
                message="UNKNOWN: No entities to rank",
                source="no_entities",
            )

        stable_tiers = [sorted(tier) for tier in tiers]
        total = sum(len(t) for t in stable_tiers)
        logger.debug(
            f"Ranking {relation_type.value}: {len(stable_tiers)} tiers, "
            f"{total} entities"
        )
        return RankingResult(
            value=TruthValue.TRUE,
            tiers=stable_tiers,
            superlative_entity=None,
            candidates=[],
            total_entities=total,
            message=f"Tiered ranking by {relation_type.value}",
            source="tiered_ranking",
        )

    def superlative(self, kind: str) -> RankingResult:
        """Answer a superlative query ("tallest", "shortest", etc.).

        Enforces uniqueness policy: if *superlative_policy* is
        "unknown_if_not_unique" and multiple entities tie for the top tier,
        returns UNKNOWN.

        Args:
            kind: Superlative keyword (e.g., "tallest", "shortest")

        Returns:
            RankingResult with superlative entity or UNKNOWN
        """
        kind_lower = kind.lower()
        if kind_lower not in SUPERLATIVE_MAP:
            logger.warning(f"Unsupported superlative kind: '{kind}'")
            return RankingResult(
                value=TruthValue.UNKNOWN,
                tiers=[],
                superlative_entity=None,
                candidates=[],
                total_entities=0,
                message=f"UNKNOWN: Invalid superlative kind '{kind}'",
                source="invalid_kind",
            )

        relation_type, tier_index = SUPERLATIVE_MAP[kind_lower]
        ranking = self.rank(relation_type)

        if ranking.value != TruthValue.TRUE:
            return RankingResult(
                value=TruthValue.UNKNOWN,
                tiers=ranking.tiers,
                superlative_entity=None,
                candidates=[],
                total_entities=0,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="ranking_failed",
                is_consistent=ranking.is_consistent,
            )

        if tier_index >= len(ranking.tiers):
            return RankingResult(
                value=TruthValue.UNKNOWN,
                tiers=ranking.tiers,
                superlative_entity=None,
                candidates=[],
                total_entities=ranking.total_entities,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="no_tier",
            )

        candidates = ranking.tiers[tier_index]

        if self.superlative_policy == "unknown_if_not_unique" and len(candidates) > 1:
            logger.info(
                f"Superlative {kind}: multiple candidates {candidates} → UNKNOWN"
            )
            return RankingResult(
                value=TruthValue.UNKNOWN,
                tiers=ranking.tiers,
                superlative_entity=None,
                candidates=candidates,
                total_entities=ranking.total_entities,
                message=UNKNOWN_SUPERLATIVE_MESSAGE,
                source="ambiguous",
            )

        entity = candidates[0]
        logger.info(f"Superlative {kind}: {entity}")
        return RankingResult(
            value=TruthValue.TRUE,
            tiers=ranking.tiers,
            superlative_entity=entity,
            candidates=candidates,
            total_entities=ranking.total_entities,
            message=f"{entity} is {kind}",
            source="superlative",
        )

    def is_consistent(self, relation_type: RelationType) -> bool:
        """Return True if the graph for *relation_type* is cycle-free.

        Args:
            relation_type: Relation type to check

        Returns:
            True if consistent (no cycles)
        """
        return not self.cache._has_cycle(relation_type)
