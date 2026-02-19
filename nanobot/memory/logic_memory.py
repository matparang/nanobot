"""LogicMemory: toggleable deterministic graph-based reasoning integration.

Wraps DeterministicReasoningAgent to provide:
- Cycle/contradiction-checked fact ingestion (store_fact / add_relation)
- Full deterministic query routing: pairwise, superlative, range/all-satisfying
- SuperpositionalState output compatible with MemoryAwareReasoner

Enable via config: memory.deterministic_logic = true
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nanobot.memory.config_v2 import LOGGER_NAME
from nanobot.memory.types_v2 import IngestionResult, IngestionStatus, RelationType

if TYPE_CHECKING:
    from nanobot.agent.memory_types import SuperpositionalState
    from nanobot.memory.deterministic_agent import DeterministicReasoningAgent

logger = logging.getLogger(LOGGER_NAME)

# Only enforce cycle/contradiction checks for strict ordering relations.
# Structural relations (DEPENDS_ON, IMPACTS, CONTAINS, …) may intentionally form cycles.
_ORDERING_RELATIONS = frozenset(
    {
        RelationType.TALLER_THAN,
        RelationType.SHORTER_THAN,
        RelationType.FASTER_THAN,
        RelationType.SLOWER_THAN,
        RelationType.GREATER_THAN,
        RelationType.LESS_THAN,
    }
)


class LogicMemory:
    """Deterministic logic memory with cycle/contradiction-checked ingestion.

    Wraps DeterministicReasoningAgent and adds:
    - Pre-ingestion cycle detection for ordering relations so impossible
      statements (e.g. "A > B > A") are rejected before they corrupt the graph.
    - Deterministic query routing for pairwise, superlative, and
      range/all-satisfying queries, returning a SuperpositionalState that is
      compatible with MemoryAwareReasoner (LLM only needed for formatting).

    Attributes:
        agent: DeterministicReasoningAgent instance
    """

    def __init__(
        self,
        agent: "Optional[DeterministicReasoningAgent]" = None,
        workspace: Optional[Path] = None,
    ):
        """Initialize LogicMemory.

        Args:
            agent: Optional DeterministicReasoningAgent (creates new one if None)
            workspace: Optional workspace path for YAML graph persistence
        """
        if agent is not None:
            self.agent = agent
        else:
            from nanobot.memory.deterministic_agent import DeterministicReasoningAgent

            self.agent = DeterministicReasoningAgent(workspace=workspace)
        logger.info("LogicMemory initialized")

    # ------------------------------------------------------------------
    # Fact ingestion
    # ------------------------------------------------------------------

    def store_fact(self, text: str) -> IngestionResult:
        """Ingest a natural language fact with cycle/contradiction validation.

        Extracts a relation from *text* via RelationExtractionEngineV2, then
        checks whether adding it would create a cycle (contradiction) in the
        graph before committing it to the cache.

        Args:
            text: Natural language fact, e.g. "Alice is taller than Bob"

        Returns:
            IngestionResult with ACCEPTED or REJECTED status
        """
        result = self.agent.extractor.extract(text)
        if result.status != IngestionStatus.ACCEPTED or not result.relation:
            return result

        a, rel_type, b = result.relation

        if self._would_create_cycle(a, rel_type, b):
            reason = (
                f"Contradiction: adding '{a} {rel_type.value} {b}' would create a "
                f"cycle; a conflicting ordering already exists in the graph"
            )
            logger.warning(f"store_fact rejected: {reason}")
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                confidence=result.confidence,
                relation=result.relation,
                reason=reason,
                template_id=result.template_id,
            )

        self.agent.cache.add_relation(a, rel_type, b, source="logic_memory")
        self.agent._auto_persist()
        logger.debug(f"store_fact accepted: {a} {rel_type.value} {b}")
        return result

    def add_relation(
        self,
        a: str,
        relation_type: RelationType,
        b: str,
        source: str = "api",
    ) -> bool:
        """Add a relation directly via the programmatic API with cycle checking.

        Args:
            a: Source entity
            relation_type: Relation type
            b: Target entity
            source: Source identifier for provenance tracking

        Returns:
            True if the relation was added, False if rejected (would cause a cycle)
        """
        if self._would_create_cycle(a, relation_type, b):
            logger.warning(
                f"add_relation rejected: {a} {relation_type.value} {b} would create a cycle"
            )
            return False

        self.agent.cache.add_relation(a, relation_type, b, source=source)
        self.agent._auto_persist()
        return True

    # ------------------------------------------------------------------
    # Deterministic query routing
    # ------------------------------------------------------------------

    def query(self, text: str) -> "Optional[SuperpositionalState]":
        """Process a natural language query and return a deterministic answer.

        Routes the query through DeterministicReasoningAgent.process() which
        covers pairwise, superlative, range/all-satisfying, count, and reachable
        query types. Returns a SuperpositionalState only when the logic module
        can produce a definitive TRUE or FALSE answer so the LLM is only
        invoked for formatting, not reasoning.

        Args:
            text: Natural language query

        Returns:
            SuperpositionalState if deterministically answerable, None otherwise
        """
        from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
        from nanobot.memory.query_parser_v2 import QueryType
        from nanobot.memory.types_v2 import TruthValue

        try:
            response = self.agent.process(text)

            # ADD_FACT queries are ingestion, not questions — do not route
            if response.query_type == QueryType.ADD_FACT.value:
                return None

            # Only return a definitive answer for TRUE or FALSE results
            if response.truth_value not in (TruthValue.TRUE.value, TruthValue.FALSE.value):
                logger.debug(
                    f"LogicMemory: UNKNOWN for '{text[:60]}' "
                    f"(type={response.query_type})"
                )
                return None

            hypothesis = Hypothesis(
                intent=f"{response.query_type.lower()}_{response.truth_value.lower()}",
                confidence=0.95,
                reasoning=response.message,
            )
            state = SuperpositionalState(
                hypotheses=[hypothesis],
                entropy=0.05,
                strategic_direction=(
                    f"Deterministic logic answer: {response.message}"
                ),
            )
            logger.info(
                f"LogicMemory: answered '{text[:60]}' -> {response.truth_value} "
                f"(type={response.query_type})"
            )
            return state

        except Exception as exc:
            logger.warning(f"LogicMemory: error processing '{text[:60]}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _would_create_cycle(
        self, a: str, rel_type: RelationType, b: str
    ) -> bool:
        """Return True if adding (a, rel_type, b) would introduce a cycle.

        Only checks strict ordering relations; structural relations are allowed
        to cycle. New entities (not yet in the cache) cannot create a cycle.

        Args:
            a: Source entity
            rel_type: Relation type
            b: Target entity

        Returns:
            True if the addition would create a cycle in the graph
        """
        if rel_type not in _ORDERING_RELATIONS:
            return False
        if (
            a not in self.agent.cache.entities
            or b not in self.agent.cache.entities
        ):
            # One or both entities are new; no existing path can close a cycle
            return False
        # A cycle would form if there is already a path b → … → a via rel_type
        path = self.agent.tr._bfs(b, a, rel_type)
        return path is not None
