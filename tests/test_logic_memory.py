"""Tests for LogicMemory: deterministic graph-based reasoning integration.

Covers:
- Routing (pairwise, superlative, range/all-satisfying) via LogicMemory.query()
- Cycle/contradiction rejection in store_fact() and add_relation()
- Integration with MemoryAwareReasoner via deterministic_logic config flag
"""

import tempfile
from pathlib import Path

import pytest

from nanobot.memory.logic_memory import LogicMemory
from nanobot.memory.relational_cache_v2 import RelationalCacheV2
from nanobot.memory.types_v2 import IngestionStatus, RelationType, TruthValue
from nanobot.memory.deterministic_agent import DeterministicReasoningAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_logic_memory():
    """LogicMemory backed by an empty cache."""
    return LogicMemory()


@pytest.fixture
def height_cache():
    """Alice > Bob > Carol > Dave via TALLER_THAN."""
    cache = RelationalCacheV2()
    cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
    cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
    cache.add_relation("Carol", RelationType.TALLER_THAN, "Dave")
    return cache


@pytest.fixture
def height_logic(height_cache):
    """LogicMemory backed by height_cache."""
    agent = DeterministicReasoningAgent(cache=height_cache)
    return LogicMemory(agent=agent)


@pytest.fixture
def supply_cache():
    """Supply chain: A→B, B→C, B→D via IMPACTS."""
    cache = RelationalCacheV2()
    cache.add_relation("ServiceA", RelationType.IMPACTS, "ServiceB")
    cache.add_relation("ServiceB", RelationType.IMPACTS, "ServiceC")
    cache.add_relation("ServiceB", RelationType.IMPACTS, "ServiceD")
    return cache


@pytest.fixture
def supply_logic(supply_cache):
    """LogicMemory backed by supply_cache."""
    agent = DeterministicReasoningAgent(cache=supply_cache)
    return LogicMemory(agent=agent)


@pytest.fixture
def temp_workspace():
    """Temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# TestStoreFact — cycle/contradiction checks
# ---------------------------------------------------------------------------


class TestStoreFact:
    """Tests for store_fact() with cycle/contradiction validation."""

    def test_store_fact_accepted(self, empty_logic_memory):
        """Valid fact is accepted and added to the cache."""
        result = empty_logic_memory.store_fact("Alice is taller than Bob")

        assert result.status == IngestionStatus.ACCEPTED
        assert result.relation == ("Alice", RelationType.TALLER_THAN, "Bob")
        # Verify via public query API
        state = empty_logic_memory.query("Is Alice taller than Bob?")
        assert state is not None

    def test_store_fact_cycle_rejected(self, height_logic):
        """Fact that would create a cycle is rejected."""
        # height_cache: Alice > Bob > Carol > Dave
        # Adding Dave > Alice would create a cycle
        result = height_logic.store_fact("Dave is taller than Alice")

        assert result.status == IngestionStatus.REJECTED
        assert "cycle" in result.reason.lower() or "contradiction" in result.reason.lower()
        # Verify via public query: Dave should NOT be taller than Alice
        state = height_logic.query("Is Dave taller than Alice?")
        assert state is None or "FALSE" in state.hypotheses[0].intent.upper()

    def test_store_fact_direct_contradiction_rejected(self, height_logic):
        """Direct contradiction (b > a when a > b already known) is rejected."""
        # Alice is already taller than Bob; Bob taller than Alice is a contradiction
        result = height_logic.store_fact("Bob is taller than Alice")

        assert result.status == IngestionStatus.REJECTED
        assert result.reason  # Non-empty reason

    def test_store_fact_transitive_cycle_rejected(self, height_logic):
        """Transitive cycle (Carol > Alice, when Alice > Bob > Carol) is rejected."""
        result = height_logic.store_fact("Carol is taller than Alice")

        assert result.status == IngestionStatus.REJECTED

    def test_store_fact_new_entity_accepted(self, height_logic):
        """A new entity can be added without cycle concerns."""
        result = height_logic.store_fact("Eve is taller than Alice")

        # Eve is a new entity, no cycle possible
        assert result.status == IngestionStatus.ACCEPTED

    def test_store_fact_self_relation_rejected(self, empty_logic_memory):
        """Self-relation is rejected by extraction layer."""
        result = empty_logic_memory.store_fact("Alice is taller than Alice")

        assert result.status == IngestionStatus.REJECTED
        assert "self" in result.reason.lower()

    def test_store_fact_non_ordering_relation_allows_cycle(self):
        """Structural relations (IMPACTS) are not cycle-checked."""
        cache = RelationalCacheV2()
        cache.add_relation("A", RelationType.IMPACTS, "B")
        cache.add_relation("B", RelationType.IMPACTS, "C")
        agent = DeterministicReasoningAgent(cache=cache)
        lm = LogicMemory(agent=agent)

        # C impacts A would form a cycle in IMPACTS graph, but IMPACTS is structural
        # — store_fact uses extraction which only handles known NL patterns
        # We directly test add_relation for structural non-check
        result = lm.add_relation("C", RelationType.IMPACTS, "A", source="test")
        assert result is True  # No cycle rejection for IMPACTS


# ---------------------------------------------------------------------------
# TestAddRelation — programmatic API with cycle checking
# ---------------------------------------------------------------------------


class TestAddRelation:
    """Tests for add_relation() with cycle checking."""

    def test_add_relation_accepted(self, empty_logic_memory):
        """Valid relation is added and returns True."""
        ok = empty_logic_memory.add_relation("X", RelationType.TALLER_THAN, "Y")
        assert ok is True
        # Verify via public query API
        state = empty_logic_memory.query("Is X taller than Y?")
        assert state is not None

    def test_add_relation_cycle_rejected(self, height_logic):
        """Relation that would create a cycle returns False."""
        ok = height_logic.add_relation("Dave", RelationType.TALLER_THAN, "Alice")
        assert ok is False

    def test_add_relation_direct_contradiction_rejected(self, height_logic):
        """Inverse relation (Bob > Alice when Alice > Bob exists) returns False."""
        ok = height_logic.add_relation("Bob", RelationType.TALLER_THAN, "Alice")
        assert ok is False

    def test_add_relation_new_entity_accepted(self, height_logic):
        """New entity can always be added without cycle rejection."""
        ok = height_logic.add_relation("Zara", RelationType.TALLER_THAN, "Alice")
        assert ok is True


# ---------------------------------------------------------------------------
# TestLogicMemoryQuery — pairwise routing
# ---------------------------------------------------------------------------


class TestLogicMemoryPairwise:
    """Tests for pairwise query routing via LogicMemory.query()."""

    def test_pairwise_true(self, height_logic):
        """Direct pairwise query returns TRUE state."""
        state = height_logic.query("Is Alice taller than Bob?")

        assert state is not None
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].confidence == 0.95
        assert state.entropy < 0.1

    def test_pairwise_transitive_true(self, height_logic):
        """Transitive pairwise query returns TRUE state."""
        state = height_logic.query("Is Alice taller than Dave?")

        assert state is not None
        assert "TRUE" in state.hypotheses[0].intent.upper()

    def test_pairwise_false(self, height_logic):
        """Inverted pairwise query returns FALSE state."""
        state = height_logic.query("Is Dave taller than Alice?")

        assert state is not None
        assert "FALSE" in state.hypotheses[0].intent.upper()

    def test_pairwise_unknown_entity_returns_none(self, height_logic):
        """Query with unknown entity returns None (UNKNOWN)."""
        state = height_logic.query("Is Zara taller than Alice?")

        assert state is None

    def test_pairwise_self_comparison_returns_none(self, height_logic):
        """Self-comparison returns None (UNKNOWN)."""
        state = height_logic.query("Is Alice taller than Alice?")

        assert state is None


# ---------------------------------------------------------------------------
# TestLogicMemorySuperlative — superlative routing
# ---------------------------------------------------------------------------


class TestLogicMemorySuperlative:
    """Tests for superlative query routing."""

    def test_tallest(self, height_logic):
        """'Who is the tallest?' returns Alice."""
        state = height_logic.query("Who is the tallest?")

        assert state is not None
        assert "Alice" in state.hypotheses[0].reasoning

    def test_shortest(self, height_logic):
        """'Who is the shortest?' returns Dave."""
        state = height_logic.query("Who is the shortest?")

        assert state is not None
        assert "Dave" in state.hypotheses[0].reasoning

    def test_ambiguous_superlative_returns_none(self):
        """Ambiguous superlative (tie at top) returns None."""
        cache = RelationalCacheV2()
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Carol")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        agent = DeterministicReasoningAgent(cache=cache)
        lm = LogicMemory(agent=agent)

        state = lm.query("Who is the tallest?")

        # Both Alice and Bob tie at top → UNKNOWN → None
        assert state is None

    def test_fastest(self, empty_logic_memory):
        """Superlative for FASTER_THAN works."""
        empty_logic_memory.add_relation("Car", RelationType.FASTER_THAN, "Bike")
        empty_logic_memory.add_relation("Bike", RelationType.FASTER_THAN, "Walk")

        state = empty_logic_memory.query("Who is the fastest?")
        assert state is not None
        assert "Car" in state.hypotheses[0].reasoning


# ---------------------------------------------------------------------------
# TestLogicMemoryAllSatisfying — range/all-satisfying routing
# ---------------------------------------------------------------------------


class TestLogicMemoryAllSatisfying:
    """Tests for range/all-satisfying query routing."""

    def test_who_is_taller_than_carol(self, height_logic):
        """'Who is taller than Carol?' returns Alice and Bob (via transitivity)."""
        state = height_logic.query("Who is taller than Carol?")

        assert state is not None
        reasoning = state.hypotheses[0].reasoning
        assert "Alice" in reasoning or "Bob" in reasoning

    def test_all_satisfying_empty_returns_none(self, height_logic):
        """Query with no satisfying entities returns None."""
        state = height_logic.query("Who is taller than Alice?")

        # Nobody is taller than Alice in the test graph
        assert state is None

    def test_reachable_via_impacts(self, supply_logic):
        """Pairwise IMPACTS query returns TRUE for reachable service."""
        state = supply_logic.query("Is ServiceC impacted by ServiceA?")

        assert state is not None
        reasoning = state.hypotheses[0].reasoning
        assert "TRUE" in state.hypotheses[0].intent.upper() or "TRUE" in reasoning.upper()


# ---------------------------------------------------------------------------
# TestLogicMemoryAddFact — add_fact queries not routed
# ---------------------------------------------------------------------------


class TestLogicMemoryAddFact:
    """Tests that ADD_FACT queries are not routed as questions."""

    def test_declarative_statement_not_routed(self, empty_logic_memory):
        """Declarative 'Alice is taller than Bob' is not routed as a question."""
        # store_fact() handles ingestion; query() should return None for ADD_FACT
        state = empty_logic_memory.query("Alice is taller than Bob")

        # ADD_FACT queries should return None from query()
        assert state is None


# ---------------------------------------------------------------------------
# TestMemoryAwareReasonerDeterministicLogic — integration with MemoryAwareReasoner
# ---------------------------------------------------------------------------


class TestMemoryAwareReasonerDeterministicLogic:
    """Integration tests for deterministic_logic flag in MemoryAwareReasoner."""

    def test_initialization_with_deterministic_logic(self, temp_workspace):
        """LogicMemory is initialized when deterministic_logic=True."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True},
        )

        assert reasoner.use_deterministic_logic is True
        assert reasoner.logic_memory is not None
        assert reasoner.hypothesis_engine is None
        assert reasoner.reasoner_v2 is None

    def test_initialization_without_flag_uses_v1(self, temp_workspace):
        """v1 HypothesisEngine is used when deterministic_logic=False (default)."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": False},
        )

        assert reasoner.use_deterministic_logic is False
        assert reasoner.logic_memory is None
        assert reasoner.hypothesis_engine is not None

    def test_deterministic_logic_takes_priority_over_v2(self, temp_workspace):
        """deterministic_logic=True takes priority over use_memory_v2=True."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True, "use_memory_v2": True},
        )

        assert reasoner.use_deterministic_logic is True
        assert reasoner.logic_memory is not None
        assert reasoner.reasoner_v2 is None

    def test_pairwise_query_answered_by_logic_memory(self, temp_workspace):
        """Pairwise query is answered deterministically via LogicMemory."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True},
        )
        # Populate the cache directly
        reasoner.logic_memory.add_relation("Alice", RelationType.TALLER_THAN, "Bob")

        can_answer, state = reasoner.check_memory_first("Is Alice taller than Bob?")

        assert can_answer is True
        assert state is not None
        assert state.entropy < 0.1
        assert state.hypotheses[0].confidence == 0.95

    def test_superlative_query_answered_by_logic_memory(self, temp_workspace):
        """Superlative query is answered deterministically via LogicMemory."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True},
        )
        reasoner.logic_memory.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        reasoner.logic_memory.add_relation("Bob", RelationType.TALLER_THAN, "Carol")

        can_answer, state = reasoner.check_memory_first("Who is the tallest?")

        assert can_answer is True
        assert state is not None
        assert "Alice" in state.hypotheses[0].reasoning

    def test_all_satisfying_answered_by_logic_memory(self, temp_workspace):
        """All-satisfying query is answered via LogicMemory."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True},
        )
        reasoner.logic_memory.add_relation("Alice", RelationType.TALLER_THAN, "Carol")
        reasoner.logic_memory.add_relation("Bob", RelationType.TALLER_THAN, "Carol")

        can_answer, state = reasoner.check_memory_first("Who is taller than Carol?")

        assert can_answer is True
        assert state is not None

    def test_unknown_query_returns_false(self, temp_workspace):
        """Query with no matching data returns (False, None)."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True},
        )

        can_answer, state = reasoner.check_memory_first("Is Alice taller than Bob?")

        assert can_answer is False
        assert state is None

    def test_cycle_rejected_fact_not_stored(self, temp_workspace):
        """Cycle-creating fact is rejected and does not enter the cache."""
        from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner

        reasoner = MemoryAwareReasoner(
            workspace=temp_workspace,
            memory_config={"deterministic_logic": True},
        )
        lm = reasoner.logic_memory
        lm.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        lm.add_relation("Bob", RelationType.TALLER_THAN, "Carol")

        # This would create a cycle
        result = lm.store_fact("Carol is taller than Alice")

        assert result.status == IngestionStatus.REJECTED
        # Verify via public query: Carol should NOT be taller than Alice
        state = lm.query("Is Carol taller than Alice?")
        assert state is None or "FALSE" in state.hypotheses[0].intent.upper()


# ---------------------------------------------------------------------------
# TestLogicMemoryPersistence — workspace-based YAML persistence
# ---------------------------------------------------------------------------


class TestLogicMemoryPersistence:
    """Tests for graph persistence through LogicMemory."""

    def test_facts_persisted_on_store(self, temp_workspace):
        """store_fact() auto-persists to YAML."""
        lm = LogicMemory(workspace=temp_workspace)
        lm.store_fact("Alice is taller than Bob")

        graph_file = temp_workspace / "memory" / "knowledge_graph.yaml"
        assert graph_file.exists()

    def test_persist_and_reload(self, temp_workspace):
        """Persisted graph can be reloaded into a new LogicMemory instance."""
        # Write facts
        lm1 = LogicMemory(workspace=temp_workspace)
        lm1.store_fact("Alice is taller than Bob")

        # Reload from YAML
        from nanobot.memory.graph_persistence import GraphPersistence
        from nanobot.memory.relational_cache_v2 import RelationalCacheV2
        from nanobot.memory.deterministic_agent import DeterministicReasoningAgent

        cache2 = GraphPersistence(temp_workspace).load()
        agent2 = DeterministicReasoningAgent(cache=cache2)
        lm2 = LogicMemory(agent=agent2)

        state = lm2.query("Is Alice taller than Bob?")
        assert state is not None
        assert "TRUE" in state.hypotheses[0].intent.upper()
