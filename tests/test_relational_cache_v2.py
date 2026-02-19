"""Tests for RelationalCacheV2 with deterministic reasoning."""

import pytest

from nanobot.memory.relational_cache_v2 import RelationalCacheV2
from nanobot.memory.types_v2 import RelationType, TruthValue


@pytest.fixture
def cache():
    """Create a fresh RelationalCacheV2 instance."""
    return RelationalCacheV2()


class TestDirectAndInverseStorage:
    """Test that both direct and inverse relations are stored."""
    
    def test_add_taller_than_stores_both_directions(self, cache):
        """Test that adding TALLER_THAN also stores SHORTER_THAN inverse."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        # Direct relation should be TRUE
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Bob")
        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
        
        # Inverse relation should also be TRUE
        result = cache.query_relation("Bob", RelationType.SHORTER_THAN, "Alice")
        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
    
    def test_add_shorter_than_stores_both_directions(self, cache):
        """Test that adding SHORTER_THAN also stores TALLER_THAN inverse."""
        cache.add_relation("Bob", RelationType.SHORTER_THAN, "Carol")
        
        # Direct relation should be TRUE
        result = cache.query_relation("Bob", RelationType.SHORTER_THAN, "Carol")
        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
        
        # Inverse relation should also be TRUE
        result = cache.query_relation("Carol", RelationType.TALLER_THAN, "Bob")
        assert result.value == TruthValue.TRUE
        assert result.source == "direct"
    
    def test_entities_tracked(self, cache):
        """Test that entities are tracked in the known set."""
        assert len(cache.entities) == 0
        
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        assert "Alice" in cache.entities
        assert "Bob" in cache.entities
        assert len(cache.entities) == 2


class TestSelfComparison:
    """Test that self-comparisons always return UNKNOWN."""
    
    def test_self_comparison_returns_unknown(self, cache):
        """Test that querying A rel A returns UNKNOWN."""
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Alice")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "self_check"
        assert "Self-comparison" in result.message
    
    def test_self_comparison_even_if_in_cache(self, cache):
        """Test that self-comparison is UNKNOWN even if entity is known."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Alice")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "self_check"


class TestUnknownEntities:
    """Test that queries with unknown entities return UNKNOWN."""
    
    def test_both_entities_unknown(self, cache):
        """Test that querying unknown entities returns UNKNOWN."""
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "entity_check"
        assert "unknown entity" in result.message.lower() or "not in knowledge base" in result.message.lower()
    
    def test_first_entity_unknown(self, cache):
        """Test UNKNOWN when first entity is unknown."""
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "entity_check"
    
    def test_second_entity_unknown(self, cache):
        """Test UNKNOWN when second entity is unknown."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Carol")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "entity_check"


class TestUnknownRelation:
    """Test that unknown relations between known entities return UNKNOWN."""
    
    def test_no_known_relation(self, cache):
        """Test UNKNOWN when entities are known but no relation exists."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Carol", RelationType.TALLER_THAN, "Dave")
        
        # Alice and Carol are both known, but no relation between them
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Carol")
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "no_information"


class TestTransitiveReasoning:
    """Test deterministic transitive reasoning via graph reachability."""
    
    def test_simple_transitive_chain(self, cache):
        """Test A > B, B > C => A > C."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        # Query A > C should be TRUE via transitivity
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Carol")
        
        assert result.value == TruthValue.TRUE
        assert result.source == "transitive"
        assert "path" in result.details
    
    def test_longer_transitive_chain(self, cache):
        """Test transitivity over longer chain."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("C", RelationType.TALLER_THAN, "D")
        cache.add_relation("D", RelationType.TALLER_THAN, "E")
        
        # Query A > E
        result = cache.query_relation("A", RelationType.TALLER_THAN, "E")
        
        assert result.value == TruthValue.TRUE
        assert result.source == "transitive"
        assert len(result.details["path"]) == 5
    
    def test_transitive_with_branches(self, cache):
        """Test transitivity with branching graph."""
        # A > B, A > C, B > D, C > D
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("A", RelationType.TALLER_THAN, "C")
        cache.add_relation("B", RelationType.TALLER_THAN, "D")
        cache.add_relation("C", RelationType.TALLER_THAN, "D")
        
        # A > D should be provable via multiple paths
        result = cache.query_relation("A", RelationType.TALLER_THAN, "D")
        
        assert result.value == TruthValue.TRUE


class TestFalseViaOpposite:
    """Test that FALSE is returned when opposite relation is provable."""
    
    def test_direct_opposite_is_false(self, cache):
        """Test that A > B => B > A is FALSE."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        # Query Bob > Alice (opposite is stored as inverse)
        result = cache.query_relation("Bob", RelationType.TALLER_THAN, "Alice")
        
        assert result.value == TruthValue.FALSE
        assert result.source == "opposite_provable"
    
    def test_transitive_opposite_is_false(self, cache):
        """Test that opposite via transitivity is FALSE."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        cache.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        # Query Carol > Alice should be FALSE (opposite is provable)
        result = cache.query_relation("Carol", RelationType.TALLER_THAN, "Alice")
        
        assert result.value == TruthValue.FALSE
        assert result.source == "opposite_provable"


class TestCycleDetection:
    """Test cycle detection for inconsistent relations."""
    
    def test_simple_cycle(self, cache):
        """Test detection of simple cycle A > B > C > A."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("C", RelationType.TALLER_THAN, "A")
        
        # Ranking should detect cycle and return UNKNOWN
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "cycle_detection"
    
    def test_self_loop(self, cache):
        """Test detection of self-loop (should not happen with self-check, but test graph)."""
        # Manually add to test cycle detection (bypassing add_relation)
        cache.entities.add("A")
        cache.relations[(("A", RelationType.TALLER_THAN))].add("A")
        
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "cycle_detection"
    
    def test_no_cycle(self, cache):
        """Test that acyclic graph does not trigger cycle detection."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("A", RelationType.TALLER_THAN, "C")
        
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        assert result.source == "tiered_ranking"


class TestTieredRanking:
    """Test tiered ranking for partial orders."""
    
    def test_total_order_single_tier_per_level(self, cache):
        """Test total order produces one entity per tier."""
        cache.add_relation("A", RelationType.TALLER_THAN, "B")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        cache.add_relation("C", RelationType.TALLER_THAN, "D")
        
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        tiers = result.details["tiers"]
        
        # Should have 4 tiers, one entity each
        assert len(tiers) == 4
        assert tiers[0] == ["A"]
        assert tiers[1] == ["B"]
        assert tiers[2] == ["C"]
        assert tiers[3] == ["D"]
    
    def test_partial_order_multiple_per_tier(self, cache):
        """Test partial order with incomparable elements in same tier."""
        # A > C, B > C (A and B are incomparable, both in tier 0)
        cache.add_relation("A", RelationType.TALLER_THAN, "C")
        cache.add_relation("B", RelationType.TALLER_THAN, "C")
        
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        tiers = result.details["tiers"]
        
        # Tier 0 should have both A and B (sorted)
        assert len(tiers) >= 2
        assert sorted(tiers[0]) == ["A", "B"]
        assert tiers[1] == ["C"]
    
    def test_stable_sort_within_tiers(self, cache):
        """Test that entities within a tier are sorted stably (alphabetically)."""
        # Create incomparable entities in same tier
        cache.add_relation("Z", RelationType.TALLER_THAN, "X")
        cache.add_relation("A", RelationType.TALLER_THAN, "X")
        cache.add_relation("M", RelationType.TALLER_THAN, "X")
        
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.TRUE
        tiers = result.details["tiers"]
        
        # First tier should be sorted alphabetically
        assert tiers[0] == ["A", "M", "Z"]
    
    def test_empty_ranking(self, cache):
        """Test ranking when no entities exist."""
        result = cache.rank_entities(RelationType.TALLER_THAN)
        
        assert result.value == TruthValue.UNKNOWN
        assert result.source == "no_entities"


class TestConfidenceAndSource:
    """Test confidence and source tracking."""
    
    def test_add_relation_with_confidence(self, cache):
        """Test that confidence can be passed (informational only)."""
        # Should not raise error
        cache.add_relation(
            "Alice",
            RelationType.TALLER_THAN,
            "Bob",
            confidence=0.95
        )
        
        result = cache.query_relation("Alice", RelationType.TALLER_THAN, "Bob")
        assert result.value == TruthValue.TRUE
    
    def test_add_relation_with_source(self, cache):
        """Test that source tracking works."""
        cache.add_relation(
            "Alice",
            RelationType.TALLER_THAN,
            "Bob",
            source="test_extraction"
        )
        
        # Source should be tracked internally
        assert ("Alice", RelationType.TALLER_THAN, "Bob") in cache.sources
        assert cache.sources[("Alice", RelationType.TALLER_THAN, "Bob")] == "test_extraction"


class TestDeterminism:
    """Test that queries are deterministic."""
    
    def test_repeated_queries_same_result(self, cache):
        """Test that repeated queries return the same result."""
        cache.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        results = [
            cache.query_relation("Alice", RelationType.TALLER_THAN, "Bob")
            for _ in range(10)
        ]
        
        # All results should be identical
        assert all(r.value == TruthValue.TRUE for r in results)
        assert all(r.source == "direct" for r in results)
    
    def test_ranking_deterministic(self, cache):
        """Test that ranking is deterministic."""
        cache.add_relation("C", RelationType.TALLER_THAN, "D")
        cache.add_relation("A", RelationType.TALLER_THAN, "D")
        cache.add_relation("B", RelationType.TALLER_THAN, "D")
        
        results = [
            cache.rank_entities(RelationType.TALLER_THAN)
            for _ in range(10)
        ]
        
        # All tier structures should be identical
        first_tiers = results[0].details["tiers"]
        for result in results[1:]:
            assert result.details["tiers"] == first_tiers
