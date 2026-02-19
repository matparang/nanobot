"""Tests for MemoryFirstReasonerV2 with strict no-LLM policy."""

import pytest

from nanobot.memory.memory_first_reasoner_v2 import MemoryFirstReasonerV2
from nanobot.memory.relational_cache_v2 import RelationalCacheV2
from nanobot.memory.types_v2 import RelationType, TruthValue


@pytest.fixture
def cache():
    """Create a fresh RelationalCacheV2."""
    return RelationalCacheV2()


@pytest.fixture
def reasoner(cache):
    """Create a MemoryFirstReasonerV2 with cache."""
    return MemoryFirstReasonerV2(cache=cache)


class TestPairwiseQueries:
    """Test pairwise relation queries."""
    
    def test_query_pairwise_direct_true(self, reasoner, cache):
        """Test pairwise query for direct relation returns TRUE."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        result = reasoner.query_pairwise("Alice", "Bob", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
    
    def test_query_pairwise_transitive_true(self, reasoner, cache):
        """Test pairwise query for transitive relation returns TRUE."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        result = reasoner.query_pairwise("Alice", "Carol", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        assert result.source == "transitive"
    
    def test_query_pairwise_opposite_false(self, reasoner, cache):
        """Test pairwise query for opposite relation returns FALSE."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        result = reasoner.query_pairwise("Bob", "Alice", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.FALSE
        assert result.source == "opposite_provable"
    
    def test_query_pairwise_unknown_entity(self, reasoner):
        """Test pairwise query with unknown entity returns UNKNOWN."""
        result = reasoner.query_pairwise("Alice", "Bob", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "entity_check"
    
    def test_query_pairwise_unknown_relation(self, reasoner, cache):
        """Test pairwise query with known entities but no relation returns UNKNOWN."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Carol", RelationType.TALLER_THAN, "Dave")
        
        result = reasoner.query_pairwise("Alice", "Carol", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "no_information"
    
    def test_query_pairwise_self_comparison(self, reasoner):
        """Test pairwise query for self-comparison returns UNKNOWN."""
        result = reasoner.query_pairwise("Alice", "Alice", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "self_check"


class TestSuperlativeQueries:
    """Test superlative queries (tallest/shortest)."""
    
    def test_query_tallest_unique_chain(self, reasoner, cache):
        """Test tallest query with unique tallest entity."""
        # A > B > C, so A is uniquely tallest
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        result = reasoner.query_superlative("tallest")
        
        assert result.value == TruthValue.TRUE
        assert result.source == "superlative"
        assert result.details["entity"] == "Alice"
        assert result.details["kind"] == "tallest"
    
    def test_query_shortest_unique_chain(self, reasoner, cache):
        """Test shortest query with unique shortest entity."""
        # A > B > C, so C is uniquely shortest
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        result = reasoner.query_superlative("shortest")
        
        assert result.value == TruthValue.TRUE
        assert result.source == "superlative"
        assert result.details["entity"] == "Carol"
        assert result.details["kind"] == "shortest"
    
    def test_query_tallest_ambiguous(self, reasoner, cache):
        """Test tallest query with multiple candidates returns UNKNOWN."""
        # A > C, B > C (A and B are both tallest, incomparable)
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Carol")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        result = reasoner.query_superlative("tallest")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "ambiguous"
        # Should report the candidates
        assert "candidates" in result.details
        assert set(result.details["candidates"]) == {"Alice", "Bob"}
    
    def test_query_shortest_ambiguous(self, reasoner, cache):
        """Test shortest query with multiple candidates returns UNKNOWN."""
        # A > B, A > C (B and C are both shortest, incomparable)
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Carol")
        
        result = reasoner.query_superlative("shortest")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "ambiguous"
        assert set(result.details["candidates"]) == {"Bob", "Carol"}
    
    def test_query_superlative_no_entities(self, reasoner):
        """Test superlative query with no entities returns UNKNOWN."""
        result = reasoner.query_superlative("tallest")
        
        assert result.value == TruthValue.UNKNOWN
        # Should indicate ranking failed or no tier
        assert result.source in ["ranking_failed", "no_tier"]
    
    def test_query_superlative_invalid_kind(self, reasoner):
        """Test superlative query with invalid kind returns UNKNOWN."""
        result = reasoner.query_superlative("fastest")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "invalid_kind"


class TestRankingQueries:
    """Test ranking queries."""
    
    def test_rank_total_order(self, reasoner, cache):
        """Test ranking with total order (linear chain)."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("C", RelationType.TALLER_THAN, "D")
        
        result = reasoner.rank_all_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        assert result.source == "tiered_ranking"
        
        tiers = result.details["tiers"]
        assert len(tiers) == 4
        assert tiers[0] == ["A"]
        assert tiers[1] == ["B"]
        assert tiers[2] == ["C"]
        assert tiers[3] == ["D"]
    
    def test_rank_partial_order(self, reasoner, cache):
        """Test ranking with partial order (tiered)."""
        # A > C, B > C (A and B in tier 0, C in tier 1)
        cache.add_relation("A", RelationType.TALLER_THAN, "C")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        
        result = reasoner.rank_all_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        
        tiers = result.details["tiers"]
        assert len(tiers) == 2
        assert sorted(tiers[0]) == ["A", "B"]
        assert tiers[1] == ["C"]
    
    def test_rank_with_cycle(self, reasoner, cache):
        """Test ranking with cycle returns UNKNOWN."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("C", RelationType.TALLER_THAN, "A")
        
        result = reasoner.rank_all_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "cycle_detection"
    
    def test_rank_empty(self, reasoner):
        """Test ranking with no entities returns UNKNOWN."""
        result = reasoner.rank_all_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "no_entities"
    
    def test_rank_stable_sorting(self, reasoner, cache):
        """Test that ranking within tiers is stable (alphabetical)."""
        cache.add_relation("Z", RelationType.TALLER_THAN, "X")
        cache.add_relation("A", RelationType.TALLER_THAN, "X")
        cache.add_relation("M", RelationType.TALLER_THAN, "X")
        
        result = reasoner.rank_all_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        tiers = result.details["tiers"]
        
        # First tier should be alphabetically sorted
        assert tiers[0] == ["A", "M", "Z"]


class TestMemoryFirstEnforcement:
    """Test that reasoner strictly uses cache (no LLM fallback)."""
    
    def test_no_llm_fallback_for_unknown(self, reasoner):
        """Test that UNKNOWN is returned, not LLM invocation."""
        # Query unknown entities - should return UNKNOWN, not invoke LLM
        result = reasoner.query_pairwise("Alice", "Bob", RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        # Should be from cache check, not LLM
        assert result.source == "entity_check"
    
    def test_no_llm_fallback_for_ambiguous(self, reasoner, cache):
        """Test that ambiguous superlative returns UNKNOWN, not LLM."""
        cache.add_relation("A", RelationType.TALLER_THAN, "C")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        
        result = reasoner.query_superlative("tallest")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "ambiguous"


class TestInitialization:
    """Test reasoner initialization."""
    
    def test_init_with_cache(self):
        """Test initialization with provided cache."""
        cache = RelationalCacheV2()
        reasoner = MemoryFirstReasonerV2(cache=cache)
        
        assert reasoner.cache is cache
    
    def test_init_without_cache(self):
        """Test initialization creates new cache if not provided."""
        reasoner = MemoryFirstReasonerV2()
        
        assert isinstance(reasoner.cache, RelationalCacheV2)
    
    def test_init_with_custom_policy(self):
        """Test initialization with custom superlative policy."""
        reasoner = MemoryFirstReasonerV2(superlative_policy="custom")
        
        assert reasoner.superlative_policy == "custom"


class TestDeterminism:
    """Test that reasoner is deterministic."""
    
    def test_repeated_pairwise_same_result(self, reasoner, cache):
        """Test that repeated pairwise queries return same result."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        results = [
            reasoner.query_pairwise("Alice", "Bob", RelationType.TALLER_THAN)
            for _ in range(10)
        ]
        
        assert all(r.value == TruthValue.TRUE for r in results)
        assert all(r.source == "direct" for r in results)
    
    def test_repeated_superlative_same_result(self, reasoner, cache):
        """Test that repeated superlative queries return same result."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        
        results = [reasoner.query_superlative("tallest") for _ in range(10)]
        
        assert all(r.value == TruthValue.TRUE for r in results)
        assert all(r.details["entity"] == "A" for r in results)
    
    def test_repeated_ranking_same_result(self, reasoner, cache):
        """Test that repeated ranking queries return same result."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        
        results = [
            reasoner.rank_all_entities(RelationType.TALLER_THAN)
            for _ in range(10)
        ]
        
        first_tiers = results[0].details["tiers"]
        assert all(r.details["tiers"] == first_tiers for r in results)


class TestIntegration:
    """Integration tests with extraction and reasoning."""
    
    def test_extraction_to_reasoning_flow(self, reasoner, cache):
        """Test full flow from manual cache loading to reasoning."""
        # Manually add relations (simulating extraction)
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        cache.add_relation("Carol", RelationType.TALLER_THAN, "Dave")
        
        # Test pairwise
        result = reasoner.query_pairwise("Alice", "Dave", RelationType.TALLER_THAN)
        assert result.value == TruthValue.TRUE
        
        # Test superlative
        tallest = reasoner.query_superlative("tallest")
        assert tallest.value == TruthValue.TRUE
        assert tallest.details["entity"] == "Alice"
        
        shortest = reasoner.query_superlative("shortest")
        assert shortest.value == TruthValue.TRUE
        assert shortest.details["entity"] == "Dave"
        
        # Test ranking
        ranking = reasoner.rank_all_entities(RelationType.TALLER_THAN)
        assert ranking.value == TruthValue.TRUE
        assert len(ranking.details["tiers"]) == 4
