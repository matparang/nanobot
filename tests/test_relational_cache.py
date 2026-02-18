"""Tests for relational cache functionality."""

import tempfile
from pathlib import Path

import pytest
import yaml

from nanobot.memory.relational_cache import RelationalCache


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_workspace):
    """Create a RelationalCache instance."""
    return RelationalCache(temp_workspace)


def test_cache_initialization(cache, temp_workspace):
    """Test that RelationalCache initializes correctly."""
    assert cache.workspace == temp_workspace
    assert cache.cache_file.exists()

    # Verify initial structure
    data = yaml.safe_load(cache.cache_file.read_text())
    assert data["schema_version"] == 1
    assert data["patterns"] == []
    assert data["entities"] == {}
    assert data["relationships"] == []
    assert "statistics" in data


def test_add_pattern(cache):
    """Test adding a pattern to the cache."""
    cache.add_pattern(
        pattern_type="inference",
        data={"conclusion": "A implies B", "confidence": 0.9},
        entities=["A", "B"]
    )

    patterns = cache.get_patterns()
    assert len(patterns) == 1
    assert patterns[0]["type"] == "inference"
    assert patterns[0]["data"]["conclusion"] == "A implies B"
    assert "A" in patterns[0]["entities"]


def test_add_multiple_patterns(cache):
    """Test adding multiple patterns."""
    for i in range(5):
        cache.add_pattern(
            pattern_type="test",
            data={"index": i}
        )

    patterns = cache.get_patterns()
    assert len(patterns) == 5

    stats = cache.get_statistics()
    assert stats["total_patterns"] == 5


def test_get_patterns_with_type_filter(cache):
    """Test filtering patterns by type."""
    cache.add_pattern("type_a", {"data": "a1"})
    cache.add_pattern("type_b", {"data": "b1"})
    cache.add_pattern("type_a", {"data": "a2"})

    type_a_patterns = cache.get_patterns(pattern_type="type_a")
    assert len(type_a_patterns) == 2
    assert all(p["type"] == "type_a" for p in type_a_patterns)


def test_get_patterns_with_limit(cache):
    """Test limiting the number of patterns returned."""
    for i in range(10):
        cache.add_pattern("test", {"index": i})

    limited = cache.get_patterns(limit=3)
    assert len(limited) == 3


def test_add_relationship(cache):
    """Test adding entity relationships."""
    cache.add_relationship(
        source="Alice",
        target="Bob",
        relation_type="friend_of",
        properties={"since": "2020"}
    )

    # Verify entities were created
    entities = cache.get_entities()
    assert "Alice" in entities
    assert "Bob" in entities

    # Verify relationship
    rels = cache.get_entity_relationships("Alice")
    assert len(rels) == 1
    assert rels[0]["target"] == "Bob"
    assert rels[0]["type"] == "friend_of"


def test_get_entity_relationships_filter_by_type(cache):
    """Test filtering relationships by type."""
    cache.add_relationship("A", "B", "parent_of")
    cache.add_relationship("A", "C", "friend_of")
    cache.add_relationship("A", "D", "parent_of")

    parent_rels = cache.get_entity_relationships("A", relation_type="parent_of")
    assert len(parent_rels) == 2
    assert all(r["type"] == "parent_of" for r in parent_rels)


def test_update_entity_attribute(cache):
    """Test updating entity attributes."""
    cache.update_entity_attribute("Person1", "age", 30)
    cache.update_entity_attribute("Person1", "name", "Alice")

    entities = cache.get_entities()
    assert "Person1" in entities
    assert entities["Person1"]["attributes"]["age"] == 30
    assert entities["Person1"]["attributes"]["name"] == "Alice"


def test_tallest_shortest_tracking(cache):
    """Test that tallest and shortest entities are tracked correctly."""
    # Add entities with heights
    cache.update_entity_attribute("Alice", "height", 165)
    cache.update_entity_attribute("Bob", "height", 180)
    cache.update_entity_attribute("Charlie", "height", 155)
    cache.update_entity_attribute("Diana", "height", 175)

    stats = cache.get_statistics()

    # Verify tallest
    assert stats["tallest"]["entity"] == "Bob"
    assert stats["tallest"]["height"] == 180

    # Verify shortest
    assert stats["shortest"]["entity"] == "Charlie"
    assert stats["shortest"]["height"] == 155


def test_tallest_shortest_updates_dynamically(cache):
    """Test that tallest/shortest update as new entities are added."""
    # Start with one entity
    cache.update_entity_attribute("Alice", "height", 170)

    stats = cache.get_statistics()
    assert stats["tallest"]["entity"] == "Alice"
    assert stats["shortest"]["entity"] == "Alice"

    # Add taller entity
    cache.update_entity_attribute("Bob", "height", 190)

    stats = cache.get_statistics()
    assert stats["tallest"]["entity"] == "Bob"
    assert stats["shortest"]["entity"] == "Alice"

    # Add shorter entity
    cache.update_entity_attribute("Charlie", "height", 160)

    stats = cache.get_statistics()
    assert stats["tallest"]["entity"] == "Bob"
    assert stats["shortest"]["entity"] == "Charlie"


def test_cycle_detection_simple_cycle(cache):
    """Test detecting a simple cycle in relationships."""
    # Create a cycle: A -> B -> C -> A
    cache.add_relationship("A", "B", "points_to")
    cache.add_relationship("B", "C", "points_to")
    cache.add_relationship("C", "A", "points_to")

    cycles = cache.detect_cycles()

    assert len(cycles) > 0
    # Should find the cycle (order may vary)
    cycle_entities = set(cycles[0])
    assert cycle_entities == {"A", "B", "C"}


def test_cycle_detection_no_cycles(cache):
    """Test cycle detection when no cycles exist."""
    # Create a tree structure (no cycles)
    cache.add_relationship("Root", "Child1", "parent_of")
    cache.add_relationship("Root", "Child2", "parent_of")
    cache.add_relationship("Child1", "Grandchild1", "parent_of")

    cycles = cache.detect_cycles()

    assert len(cycles) == 0


def test_cycle_detection_multiple_cycles(cache):
    """Test detecting multiple cycles."""
    # First cycle: A -> B -> A
    cache.add_relationship("A", "B", "points_to")
    cache.add_relationship("B", "A", "points_to")

    # Second cycle: C -> D -> E -> C
    cache.add_relationship("C", "D", "points_to")
    cache.add_relationship("D", "E", "points_to")
    cache.add_relationship("E", "C", "points_to")

    cycles = cache.detect_cycles()

    assert len(cycles) >= 2


def test_cycle_detection_self_loop(cache):
    """Test detecting self-referential relationships."""
    cache.add_relationship("A", "A", "self_reference")

    cycles = cache.detect_cycles()

    assert len(cycles) > 0


def test_entity_pattern_count(cache):
    """Test that entity pattern counts are tracked."""
    cache.add_pattern("test", {"data": "1"}, entities=["EntityA", "EntityB"])
    cache.add_pattern("test", {"data": "2"}, entities=["EntityA"])
    cache.add_pattern("test", {"data": "3"}, entities=["EntityB", "EntityC"])

    entities = cache.get_entities()

    assert entities["EntityA"]["pattern_count"] == 2
    assert entities["EntityB"]["pattern_count"] == 2
    assert entities["EntityC"]["pattern_count"] == 1


def test_clear_cache(cache):
    """Test clearing all cache data."""
    # Add some data
    cache.add_pattern("test", {"data": "test"})
    cache.add_relationship("A", "B", "relation")
    cache.update_entity_attribute("A", "attr", "value")

    # Clear
    cache.clear()

    # Verify everything is empty
    assert len(cache.get_patterns()) == 0
    assert len(cache.get_entities()) == 0
    stats = cache.get_statistics()
    assert stats["total_patterns"] == 0
    assert stats["total_relationships"] == 0


def test_persistence_across_sessions(temp_workspace):
    """Test that cache persists across multiple sessions."""
    # First session
    cache1 = RelationalCache(temp_workspace)
    cache1.update_entity_attribute("Alice", "height", 165)
    cache1.update_entity_attribute("Bob", "height", 180)

    # Second session
    cache2 = RelationalCache(temp_workspace)
    stats = cache2.get_statistics()

    # Verify persistence
    assert stats["tallest"]["entity"] == "Bob"
    assert stats["shortest"]["entity"] == "Alice"


def test_tallest_shortest_persistence_scenario(temp_workspace):
    """Test real-world scenario of tallest/shortest tracking across sessions."""
    # Session 1: Initial data
    cache1 = RelationalCache(temp_workspace)
    cache1.update_entity_attribute("Alice", "height", 165)
    cache1.update_entity_attribute("Bob", "height", 180)
    cache1.update_entity_attribute("Charlie", "height", 155)

    stats1 = cache1.get_statistics()
    assert stats1["tallest"]["entity"] == "Bob"
    assert stats1["shortest"]["entity"] == "Charlie"

    # Session 2: Add new tallest person
    cache2 = RelationalCache(temp_workspace)
    cache2.update_entity_attribute("Diana", "height", 190)

    stats2 = cache2.get_statistics()
    assert stats2["tallest"]["entity"] == "Diana"
    assert stats2["shortest"]["entity"] == "Charlie"

    # Session 3: Add new shortest person
    cache3 = RelationalCache(temp_workspace)
    cache3.update_entity_attribute("Eve", "height", 150)

    stats3 = cache3.get_statistics()
    assert stats3["tallest"]["entity"] == "Diana"
    assert stats3["shortest"]["entity"] == "Eve"


def test_complex_cycle_detection(cache):
    """Test cycle detection in a more complex graph."""
    # Create a complex graph with multiple paths and cycles
    # Path 1: A -> B -> C -> D
    cache.add_relationship("A", "B", "next")
    cache.add_relationship("B", "C", "next")
    cache.add_relationship("C", "D", "next")

    # Create cycle by connecting D back to B
    cache.add_relationship("D", "B", "next")

    # Add another branch: B -> E -> F
    cache.add_relationship("B", "E", "branch")
    cache.add_relationship("E", "F", "branch")

    cycles = cache.detect_cycles()

    # Should detect the cycle B -> C -> D -> B
    assert len(cycles) > 0
    cycle_entities = set(cycles[0])
    assert "B" in cycle_entities
    assert "C" in cycle_entities
    assert "D" in cycle_entities


def test_relationship_statistics(cache):
    """Test that relationship statistics are maintained."""
    cache.add_relationship("A", "B", "type1")
    cache.add_relationship("B", "C", "type2")
    cache.add_relationship("C", "D", "type1")

    stats = cache.get_statistics()
    assert stats["total_relationships"] == 3


def test_get_entity_relationships_bidirectional(cache):
    """Test getting relationships where entity is source or target."""
    cache.add_relationship("Alice", "Bob", "friend_of")
    cache.add_relationship("Charlie", "Alice", "knows")

    # Alice should appear in both relationships
    alice_rels = cache.get_entity_relationships("Alice")
    assert len(alice_rels) == 2

    # Bob should only appear in one
    bob_rels = cache.get_entity_relationships("Bob")
    assert len(bob_rels) == 1
