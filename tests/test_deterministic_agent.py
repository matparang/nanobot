"""Comprehensive tests for the Deterministic Reasoning Agent.

Tests all engines via the unified agent:
- TestTransitiveReasoner
- TestRankingReasoner
- TestAggregateEngine
- TestAgentIntegration
"""

import pytest

from nanobot.memory.relational_cache_v2 import RelationalCacheV2
from nanobot.memory.types_v2 import RelationType, TruthValue
from nanobot.memory.transitive_reasoner import TransitiveReasoner, TransitiveResult
from nanobot.memory.ranking_reasoner import RankingReasoner, RankingResult
from nanobot.memory.aggregate_engine import AggregateEngine
from nanobot.memory.deterministic_agent import DeterministicReasoningAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def height_cache():
    """Alice > Bob > Carol > Dave via TALLER_THAN."""
    cache = RelationalCacheV2()
    cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
    cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
    cache.add_relation("Carol", RelationType.TALLER_THAN, "Dave")
    return cache


@pytest.fixture
def supply_chain_cache():
    """Supply chain: A→B, B→C, B→D, E→B via IMPACTS."""
    cache = RelationalCacheV2()
    cache.add_relation("ServiceA", RelationType.IMPACTS, "ServiceB")
    cache.add_relation("ServiceB", RelationType.IMPACTS, "ServiceC")
    cache.add_relation("ServiceB", RelationType.IMPACTS, "ServiceD")
    cache.add_relation("ServiceE", RelationType.IMPACTS, "ServiceB")
    return cache


# ---------------------------------------------------------------------------
# TestTransitiveReasoner
# ---------------------------------------------------------------------------


class TestTransitiveReasoner:
    """Tests for TransitiveReasoner pairwise inference."""

    def test_direct(self, height_cache):
        """Direct edge returns TRUE with path=[A,B], depth=1."""
        tr = TransitiveReasoner(height_cache)
        result = tr.query("Alice", RelationType.TALLER_THAN, "Bob")

        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
        assert result.path == ["Alice", "Bob"]
        assert result.depth == 1

    def test_transitive_2_hop(self, height_cache):
        """A > B > C, query A > C returns TRUE, depth=2, full path."""
        tr = TransitiveReasoner(height_cache)
        result = tr.query("Alice", RelationType.TALLER_THAN, "Carol")

        assert result.value == TruthValue.TRUE
        assert result.depth == 2
        assert result.path == ["Alice", "Bob", "Carol"]

    def test_transitive_3_hop(self, height_cache):
        """A > B > C > D, query A > D returns TRUE, depth=3."""
        tr = TransitiveReasoner(height_cache)
        result = tr.query("Alice", RelationType.TALLER_THAN, "Dave")

        assert result.value == TruthValue.TRUE
        assert result.depth == 3
        assert result.path == ["Alice", "Bob", "Carol", "Dave"]

    def test_opposite(self, height_cache):
        """Query D > A returns FALSE, source='opposite_provable'."""
        tr = TransitiveReasoner(height_cache)
        result = tr.query("Dave", RelationType.TALLER_THAN, "Alice")

        assert result.value == TruthValue.FALSE
        assert result.source == "opposite_provable"

    def test_self_comparison(self, height_cache):
        """Query A > A returns UNKNOWN."""
        tr = TransitiveReasoner(height_cache)
        result = tr.query("Alice", RelationType.TALLER_THAN, "Alice")

        assert result.value == TruthValue.UNKNOWN
        assert result.source == "self_check"

    def test_unknown_entity(self, height_cache):
        """Query with unknown entity returns UNKNOWN."""
        tr = TransitiveReasoner(height_cache)
        result = tr.query("Zara", RelationType.TALLER_THAN, "Alice")

        assert result.value == TruthValue.UNKNOWN
        assert result.source == "entity_check"

    def test_all_reachable(self, height_cache):
        """From Alice via TALLER_THAN returns {Bob, Carol, Dave} with paths."""
        tr = TransitiveReasoner(height_cache)
        reachable, paths = tr.all_reachable("Alice", RelationType.TALLER_THAN)

        assert set(reachable) == {"Bob", "Carol", "Dave"}
        assert reachable == sorted(reachable)
        # Each path must start at Alice
        for entity, path in paths.items():
            assert path[0] == "Alice"
            assert path[-1] == entity


# ---------------------------------------------------------------------------
# TestRankingReasoner
# ---------------------------------------------------------------------------


class TestRankingReasoner:
    """Tests for RankingReasoner tiered ranking and superlatives."""

    def test_full_ranking(self, height_cache):
        """4 entities produce 4 tiers; tier[0]=[Alice], tier[-1]=[Dave]."""
        rr = RankingReasoner(height_cache)
        result = rr.rank(RelationType.TALLER_THAN)

        assert result.value == TruthValue.TRUE
        assert len(result.tiers) == 4
        assert result.tiers[0] == ["Alice"]
        assert result.tiers[-1] == ["Dave"]
        assert result.total_entities == 4

    def test_tallest(self, height_cache):
        """Superlative 'tallest' returns Alice."""
        rr = RankingReasoner(height_cache)
        result = rr.superlative("tallest")

        assert result.value == TruthValue.TRUE
        assert result.superlative_entity == "Alice"

    def test_shortest(self, height_cache):
        """Superlative 'shortest' returns Dave."""
        rr = RankingReasoner(height_cache)
        result = rr.superlative("shortest")

        assert result.value == TruthValue.TRUE
        assert result.superlative_entity == "Dave"

    def test_invalid_superlative(self):
        """Unsupported superlative kind returns UNKNOWN."""
        cache = RelationalCacheV2()
        rr = RankingReasoner(cache)
        result = rr.superlative("wisest")

        assert result.value == TruthValue.UNKNOWN
        assert result.source == "invalid_kind"

    def test_cycle_detection(self):
        """Cyclic graph returns UNKNOWN, is_consistent=False."""
        cache = RelationalCacheV2()
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("C", RelationType.TALLER_THAN, "A")

        rr = RankingReasoner(cache)
        result = rr.rank(RelationType.TALLER_THAN)

        assert result.value == TruthValue.UNKNOWN
        assert result.is_consistent is False


# ---------------------------------------------------------------------------
# TestAggregateEngine
# ---------------------------------------------------------------------------


class TestAggregateEngine:
    """Tests for AggregateEngine count/reachable/who-satisfies."""

    def test_reachable_from(self, supply_chain_cache):
        """ServiceA impacts {B, C, D}, count=3."""
        ae = AggregateEngine(supply_chain_cache)
        result = ae.reachable_from("ServiceA", RelationType.IMPACTS)

        assert result.value == TruthValue.TRUE
        assert result.count == 3
        assert set(result.entities) == {"ServiceB", "ServiceC", "ServiceD"}

    def test_reachable_from_deeper(self, supply_chain_cache):
        """ServiceE transitively impacts {B, C, D}."""
        ae = AggregateEngine(supply_chain_cache)
        result = ae.reachable_from("ServiceE", RelationType.IMPACTS)

        assert result.value == TruthValue.TRUE
        assert set(result.entities) == {"ServiceB", "ServiceC", "ServiceD"}

    def test_who_satisfies(self, supply_chain_cache):
        """Who impacts ServiceC? → {ServiceA, ServiceB, ServiceE}."""
        ae = AggregateEngine(supply_chain_cache)
        result = ae.who_satisfies(RelationType.IMPACTS, "ServiceC")

        assert result.value == TruthValue.TRUE
        assert set(result.entities) == {"ServiceA", "ServiceB", "ServiceE"}

    def test_unknown_entity(self, supply_chain_cache):
        """Query with unknown entity returns UNKNOWN."""
        ae = AggregateEngine(supply_chain_cache)
        result = ae.reachable_from("Ghost", RelationType.IMPACTS)

        assert result.value == TruthValue.UNKNOWN
        assert result.source == "entity_check"


# ---------------------------------------------------------------------------
# TestAgentIntegration
# ---------------------------------------------------------------------------


class TestAgentIntegration:
    """End-to-end tests via DeterministicReasoningAgent NL interface."""

    @pytest.fixture
    def height_agent(self, height_cache):
        """Agent backed by height_cache."""
        return DeterministicReasoningAgent(cache=height_cache)

    def test_pairwise_via_nl(self, height_agent):
        """'Is Alice taller than Dave?' → TRUE, depth=3."""
        response = height_agent.process("Is Alice taller than Dave?")

        assert response.truth_value == TruthValue.TRUE.value
        assert response.depth == 3
        assert response.query_type == "PAIRWISE"

    def test_superlative_via_nl(self, height_agent):
        """'Who is the tallest?' → Alice."""
        response = height_agent.process("Who is the tallest?")

        assert response.truth_value == TruthValue.TRUE.value
        assert "Alice" in response.entities
        assert response.query_type == "SUPERLATIVE"

    def test_rank_via_nl(self, height_agent):
        """'Rank all' → 4 tiers."""
        response = height_agent.process("Rank all")

        assert response.query_type == "RANK"
        assert response.tiers is not None
        assert len(response.tiers) == 4

    def test_add_fact_then_query(self, height_agent):
        """Add 'Eve is taller than Alice', then query Eve > Dave → TRUE, depth=4."""
        add_resp = height_agent.add_fact("Eve is taller than Alice")
        assert add_resp.truth_value == TruthValue.TRUE.value

        response = height_agent.process("Is Eve taller than Dave?")
        assert response.truth_value == TruthValue.TRUE.value
        assert response.depth == 4

    def test_add_relation_api(self, height_agent):
        """Add via API then query superlative → Eve."""
        height_agent.add_relation("Eve", RelationType.TALLER_THAN, "Alice", source="api_test")

        response = height_agent.process("Who is the tallest?")
        assert response.truth_value == TruthValue.TRUE.value
        assert "Eve" in response.entities
