"""Unified Deterministic Reasoning Agent.

Single entry point that composes all reasoning modules. Zero LLM usage.
All answers are deterministic, traceable, and structured.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from nanobot.memory.config_v2 import LOGGER_NAME, SUPERLATIVE_POLICY
from nanobot.memory.types_v2 import RelationType, TruthValue

logger = logging.getLogger(LOGGER_NAME)


@dataclass
class AgentResponse:
    """Structured response from the DeterministicReasoningAgent.

    Attributes:
        query: Original query string
        query_type: Detected query type string (e.g., "PAIRWISE")
        answer: Raw answer object from the reasoning engine
        truth_value: String representation of TruthValue
        count: Entity count for aggregate queries
        entities: Entities involved in the answer
        reasoning_path: Proof path for transitive answers
        depth: Hop depth for transitive answers
        tiers: Tiered ranking result
        source: Reasoning source identifier
        message: Human-readable explanation
        is_consistent: Whether the knowledge graph is cycle-free
    """
    query: str
    query_type: str
    answer: Any
    truth_value: str
    count: Optional[int] = None
    entities: list[str] = field(default_factory=list)
    reasoning_path: Optional[list[str]] = None
    depth: int = 0
    tiers: Optional[list[list[str]]] = None
    source: str = ""
    message: str = ""
    is_consistent: bool = True


class DeterministicReasoningAgent:
    """Single entry-point for deterministic graph-based reasoning.

    Composes:
    - TransitiveReasoner — pairwise / transitive inference
    - RankingReasoner — superlative / tiered ranking
    - AggregateEngine — count / reachable / who-satisfies
    - DeterministicQueryParser — NL → structured query
    - RelationExtractionEngineV2 — NL fact ingestion
    - GraphPersistence (optional) — YAML save/load

    Attributes:
        cache: RelationalCacheV2 knowledge graph
        tr: TransitiveReasoner
        rr: RankingReasoner
        ae: AggregateEngine
        parser: DeterministicQueryParser
        extractor: RelationExtractionEngineV2
        persistence: Optional GraphPersistence
    """

    def __init__(
        self,
        cache=None,
        workspace: Optional[Path] = None,
        superlative_policy: str = SUPERLATIVE_POLICY,
    ):
        """Initialize the agent.

        Args:
            cache: Optional RelationalCacheV2 instance (creates new one if None)
            workspace: Optional workspace path for YAML persistence
            superlative_policy: Superlative uniqueness policy
        """
        from nanobot.memory.relational_cache_v2 import RelationalCacheV2
        from nanobot.memory.transitive_reasoner import TransitiveReasoner
        from nanobot.memory.ranking_reasoner import RankingReasoner
        from nanobot.memory.aggregate_engine import AggregateEngine
        from nanobot.memory.query_parser_v2 import DeterministicQueryParser
        from nanobot.memory.relation_extractor_v2 import RelationExtractionEngineV2

        self.cache = cache if cache is not None else RelationalCacheV2()
        self.tr = TransitiveReasoner(self.cache)
        self.rr = RankingReasoner(self.cache, superlative_policy=superlative_policy)
        self.ae = AggregateEngine(self.cache, transitive_reasoner=self.tr)
        self.parser = DeterministicQueryParser()
        self.extractor = RelationExtractionEngineV2()

        self.persistence: Optional[object] = None
        if workspace is not None:
            try:
                from nanobot.memory.graph_persistence import GraphPersistence
                self.persistence = GraphPersistence(workspace)
                logger.info(f"GraphPersistence enabled at {workspace}")
            except Exception as exc:
                logger.warning(f"Could not initialize GraphPersistence: {exc}")

        logger.info("DeterministicReasoningAgent initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, query: str) -> AgentResponse:
        """Parse and answer *query* deterministically.

        Dispatch table:
        - PAIRWISE → TransitiveReasoner.query()
        - SUPERLATIVE → RankingReasoner.superlative()
        - RANK → RankingReasoner.rank()
        - COUNT → AggregateEngine.count_reachable()
        - REACHABLE → AggregateEngine.reachable_from()
        - ALL_SATISFYING → AggregateEngine.who_satisfies()
        - ADD_FACT → RelationExtractionEngineV2 → cache

        Args:
            query: Natural language query string

        Returns:
            AgentResponse with structured answer
        """
        from nanobot.memory.query_parser_v2 import QueryType

        parsed = self.parser.parse(query)
        logger.debug(
            f"Parsed query '{query}': type={parsed.query_type.value}, "
            f"entities={parsed.entities}, relation={parsed.relation_type}"
        )

        qt = parsed.query_type

        if qt == QueryType.PAIRWISE:
            return self._handle_pairwise(query, parsed)

        if qt == QueryType.SUPERLATIVE:
            return self._handle_superlative(query, parsed)

        if qt == QueryType.RANK:
            return self._handle_rank(query, parsed)

        if qt == QueryType.COUNT:
            return self._handle_count(query, parsed)

        if qt == QueryType.REACHABLE:
            return self._handle_reachable(query, parsed)

        if qt == QueryType.ALL_SATISFYING:
            return self._handle_all_satisfying(query, parsed)

        if qt == QueryType.ADD_FACT:
            return self.add_fact(query)

        # UNKNOWN / fallback
        return AgentResponse(
            query=query,
            query_type=qt.value,
            answer=None,
            truth_value=TruthValue.UNKNOWN.value,
            message="Could not parse query",
            source="parser",
        )

    def add_fact(self, text: str) -> AgentResponse:
        """Ingest a natural language fact via RelationExtractionEngineV2.

        Args:
            text: Natural language fact (e.g., "Alice is taller than Bob")

        Returns:
            AgentResponse describing ingestion status
        """
        from nanobot.memory.types_v2 import IngestionStatus

        result = self.extractor.extract(text)
        if result.status == IngestionStatus.ACCEPTED and result.relation:
            a, rel_type, b = result.relation
            self.cache.add_relation(a, rel_type, b, source="nl_ingestion")
            self._auto_persist()
            return AgentResponse(
                query=text,
                query_type="ADD_FACT",
                answer=result,
                truth_value=TruthValue.TRUE.value,
                entities=[a, b],
                message=f"Accepted: {a} {rel_type.value} {b}",
                source="nl_ingestion",
            )

        return AgentResponse(
            query=text,
            query_type="ADD_FACT",
            answer=result,
            truth_value=TruthValue.UNKNOWN.value,
            message=f"Rejected: {result.reason}",
            source="nl_ingestion",
        )

    def add_relation(
        self,
        a: str,
        relation_type: RelationType,
        b: str,
        source: str = "api",
    ) -> None:
        """Add a relation directly via the programmatic API.

        Args:
            a: Source entity
            relation_type: Relation type
            b: Target entity
            source: Source identifier for provenance tracking
        """
        self.cache.add_relation(a, relation_type, b, source=source)
        self._auto_persist()

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def _handle_pairwise(self, query: str, parsed) -> AgentResponse:
        if len(parsed.entities) < 2 or parsed.relation_type is None:
            return self._unknown(query, "PAIRWISE", "Insufficient entities/relation")

        a, b = parsed.entities[0], parsed.entities[1]
        result = self.tr.query(a, parsed.relation_type, b)
        return AgentResponse(
            query=query,
            query_type="PAIRWISE",
            answer=result,
            truth_value=result.value.value,
            entities=result.path or [a, b],
            reasoning_path=result.path if result.path else None,
            depth=result.depth,
            source=result.source,
            message=result.message,
            is_consistent=not result.cycle_detected,
        )

    def _handle_superlative(self, query: str, parsed) -> AgentResponse:
        kind = parsed.superlative_kind or ""
        if not kind:
            return self._unknown(query, "SUPERLATIVE", "No superlative kind detected")

        result = self.rr.superlative(kind)
        return AgentResponse(
            query=query,
            query_type="SUPERLATIVE",
            answer=result,
            truth_value=result.value.value,
            entities=[result.superlative_entity] if result.superlative_entity else result.candidates,
            tiers=result.tiers if result.tiers else None,
            source=result.source,
            message=result.message,
            is_consistent=result.is_consistent,
        )

    def _handle_rank(self, query: str, parsed) -> AgentResponse:
        if parsed.relation_type is None:
            # Default to TALLER_THAN if no relation specified
            relation_type = RelationType.TALLER_THAN
        else:
            relation_type = parsed.relation_type

        result = self.rr.rank(relation_type)
        all_entities: list[str] = [e for tier in result.tiers for e in tier]
        return AgentResponse(
            query=query,
            query_type="RANK",
            answer=result,
            truth_value=result.value.value,
            entities=all_entities,
            tiers=result.tiers if result.tiers else None,
            count=result.total_entities,
            source=result.source,
            message=result.message,
            is_consistent=result.is_consistent,
        )

    def _handle_count(self, query: str, parsed) -> AgentResponse:
        if not parsed.entities or parsed.relation_type is None:
            return self._unknown(query, "COUNT", "Insufficient entities/relation")

        entity = parsed.entities[0]
        result = self.ae.count_reachable(entity, parsed.relation_type)
        return AgentResponse(
            query=query,
            query_type="COUNT",
            answer=result,
            truth_value=result.value.value,
            count=result.count,
            entities=result.entities,
            source=result.source,
            message=result.message,
        )

    def _handle_reachable(self, query: str, parsed) -> AgentResponse:
        if not parsed.entities or parsed.relation_type is None:
            return self._unknown(query, "REACHABLE", "Insufficient entities/relation")

        entity = parsed.entities[0]
        result = self.ae.reachable_from(entity, parsed.relation_type)
        return AgentResponse(
            query=query,
            query_type="REACHABLE",
            answer=result,
            truth_value=result.value.value,
            count=result.count,
            entities=result.entities,
            source=result.source,
            message=result.message,
        )

    def _handle_all_satisfying(self, query: str, parsed) -> AgentResponse:
        if not parsed.entities or parsed.relation_type is None:
            return self._unknown(query, "ALL_SATISFYING", "Insufficient entities/relation")

        target = parsed.entities[0]
        result = self.ae.who_satisfies(parsed.relation_type, target)
        return AgentResponse(
            query=query,
            query_type="ALL_SATISFYING",
            answer=result,
            truth_value=result.value.value,
            count=result.count,
            entities=result.entities,
            source=result.source,
            message=result.message,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unknown(self, query: str, query_type: str, reason: str) -> AgentResponse:
        return AgentResponse(
            query=query,
            query_type=query_type,
            answer=None,
            truth_value=TruthValue.UNKNOWN.value,
            message=f"UNKNOWN: {reason}",
            source="agent",
        )

    def _auto_persist(self) -> None:
        """Persist graph if persistence is configured."""
        if self.persistence is not None:
            try:
                self.persistence.save(self.cache)
            except Exception as exc:
                logger.warning(f"Auto-persist failed: {exc}")
