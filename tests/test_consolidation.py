"""Tests for consolidation pipeline functionality."""

import tempfile
from pathlib import Path

import pytest

from nanobot.memory.consolidation import ConsolidationPipeline
from nanobot.memory.relational_cache import RelationalCache
from nanobot.memory.session_store import SessionStore


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pipeline(temp_workspace):
    """Create a ConsolidationPipeline instance."""
    return ConsolidationPipeline(temp_workspace, {"clarify_entropy_threshold": 0.8})


@pytest.fixture
def session_store(temp_workspace):
    """Create a SessionStore instance."""
    return SessionStore(temp_workspace)


@pytest.fixture
def cache(temp_workspace):
    """Create a RelationalCache instance."""
    return RelationalCache(temp_workspace)


def test_pipeline_initialization(pipeline, temp_workspace):
    """Test that ConsolidationPipeline initializes correctly."""
    assert pipeline.workspace == temp_workspace
    assert isinstance(pipeline.session_store, SessionStore)
    assert isinstance(pipeline.relational_cache, RelationalCache)


def test_consolidate_interaction_events(pipeline, session_store):
    """Test consolidating interaction events from a session."""
    session_id = "test_interaction"
    session_store.start(session_id=session_id)

    # Add interaction events
    session_store.append_event(
        session_id,
        "interaction",
        {
            "user_message": "What is the weather?",
            "agent_response": "I'll check the weather for you."
        }
    )

    # Consolidate
    result = pipeline.consolidate_session(session_id)

    assert result["patterns_added"] == 1
    assert result["relationships_added"] == 0


def test_consolidate_reasoning_events(pipeline, session_store):
    """Test consolidating reasoning events from a session."""
    session_id = "test_reasoning"
    session_store.start(session_id=session_id)

    # Add reasoning events
    session_store.append_event(
        session_id,
        "reasoning",
        {
            "hypotheses": [
                {"intent": "weather_query", "confidence": 0.9}
            ],
            "entropy": 0.3,
            "strategic_direction": "Provide weather information"
        }
    )

    # Consolidate
    result = pipeline.consolidate_session(session_id)

    assert result["patterns_added"] == 1


def test_consolidate_entity_relationships(pipeline, session_store, cache):
    """Test consolidating entity relationship events."""
    session_id = "test_entities"
    session_store.start(session_id=session_id)

    # Add entity relationship events
    session_store.append_event(
        session_id,
        "entity_relation",
        {
            "source": "Alice",
            "target": "Bob",
            "relation_type": "taller_than",
            "properties": {}
        }
    )

    # Consolidate
    result = pipeline.consolidate_session(session_id)

    assert result["relationships_added"] == 1

    # Verify in cache
    rels = cache.get_entity_relationships("Alice")
    assert len(rels) == 1
    assert rels[0]["target"] == "Bob"


def test_consolidate_entity_attributes(pipeline, session_store, cache):
    """Test consolidating entity attribute events."""
    session_id = "test_attributes"
    session_store.start(session_id=session_id)

    # Add entity attribute events
    session_store.append_event(
        session_id,
        "entity_attribute",
        {
            "entity": "Alice",
            "attribute": "height",
            "value": 165
        }
    )

    session_store.append_event(
        session_id,
        "entity_attribute",
        {
            "entity": "Bob",
            "attribute": "height",
            "value": 180
        }
    )

    # Consolidate
    pipeline.consolidate_session(session_id)

    # Verify attributes in cache
    entities = cache.get_entities()
    assert entities["Alice"]["attributes"]["height"] == 165
    assert entities["Bob"]["attributes"]["height"] == 180

    # Verify statistics
    stats = cache.get_statistics()
    assert stats["tallest"]["entity"] == "Bob"
    assert stats["shortest"]["entity"] == "Alice"


def test_consolidate_with_archive(pipeline, session_store):
    """Test that sessions can be archived after consolidation."""
    session_id = "test_archive"
    session_store.start(session_id=session_id)
    session_store.append_event(session_id, "interaction", {"test": "data"})

    # Consolidate with archive
    pipeline.consolidate_session(session_id, archive_after=True)

    # Verify session was archived
    sessions = session_store.list_sessions(include_archived=False)
    assert not any(s["session_id"] == session_id for s in sessions)

    # But should still be in archived list
    archived = session_store.list_sessions(include_archived=True)
    assert any(s["session_id"] == session_id and s["archived"] for s in archived)


def test_consolidate_multiple_events(pipeline, session_store):
    """Test consolidating a session with multiple event types."""
    session_id = "test_multiple"
    session_store.start(session_id=session_id)

    # Add various events
    session_store.append_event(
        session_id,
        "interaction",
        {"user_message": "Hello", "agent_response": "Hi"}
    )

    session_store.append_event(
        session_id,
        "reasoning",
        {"hypotheses": [], "entropy": 0.5, "strategic_direction": "Greet"}
    )

    session_store.append_event(
        session_id,
        "entity_relation",
        {"source": "A", "target": "B", "relation_type": "related"}
    )

    # Consolidate
    result = pipeline.consolidate_session(session_id, extract_entities=True)

    assert result["patterns_added"] == 2  # interaction + reasoning
    assert result["relationships_added"] == 1


def test_consolidate_patterns_to_fractal(pipeline, cache):
    """Test consolidating patterns into fractal nodes."""
    # Add several patterns to cache
    for i in range(5):
        cache.add_pattern(
            "interaction",
            {"message": f"test message {i}"}
        )

    # Consolidate to fractal
    created_nodes = pipeline.consolidate_patterns_to_fractal(min_pattern_count=3)

    # Should create at least one node for interaction patterns
    assert len(created_nodes) > 0


def test_consolidate_patterns_below_threshold(pipeline, cache):
    """Test that patterns below minimum count are not consolidated."""
    # Add only 2 patterns (below min_pattern_count=3)
    cache.add_pattern("test_type", {"data": "1"})
    cache.add_pattern("test_type", {"data": "2"})

    # Try to consolidate
    created_nodes = pipeline.consolidate_patterns_to_fractal(min_pattern_count=3)

    # Should not create any nodes
    assert len(created_nodes) == 0


def test_full_pipeline_single_session(pipeline, session_store, cache):
    """Test running the full consolidation pipeline on a single session."""
    session_id = "full_pipeline_test"
    session_store.start(session_id=session_id)

    # Add various events
    session_store.append_event(
        session_id,
        "interaction",
        {"user_message": "Test", "agent_response": "Response"}
    )

    session_store.append_event(
        session_id,
        "entity_attribute",
        {"entity": "Alice", "attribute": "height", "value": 165}
    )

    session_store.append_event(
        session_id,
        "entity_attribute",
        {"entity": "Bob", "attribute": "height", "value": 180}
    )

    # Run full pipeline
    stats = pipeline.run_full_pipeline(session_ids=[session_id])

    assert stats["sessions_processed"] == 1
    assert stats["patterns_extracted"] >= 1

    # Verify tallest/shortest were tracked
    cache_stats = cache.get_statistics()
    assert cache_stats["tallest"]["entity"] == "Bob"
    assert cache_stats["shortest"]["entity"] == "Alice"


def test_full_pipeline_multiple_sessions(pipeline, session_store):
    """Test running the full pipeline on multiple sessions."""
    # Create multiple sessions
    for i in range(3):
        session_id = f"multi_session_{i}"
        session_store.start(session_id=session_id)
        session_store.append_event(
            session_id,
            "interaction",
            {"user_message": f"Message {i}", "agent_response": "Response"}
        )

    # Run pipeline on all sessions
    stats = pipeline.run_full_pipeline()

    assert stats["sessions_processed"] == 3
    assert stats["patterns_extracted"] == 3


def test_full_pipeline_with_archive(pipeline, session_store):
    """Test full pipeline archives sessions when requested."""
    session_id = "pipeline_archive_test"
    session_store.start(session_id=session_id)
    session_store.append_event(session_id, "interaction", {"test": "data"})

    # Run pipeline with archive
    pipeline.run_full_pipeline(session_ids=[session_id], archive_sessions=True)

    # Verify archived
    sessions = session_store.list_sessions(include_archived=False)
    assert not any(s["session_id"] == session_id for s in sessions)


def test_tallest_shortest_persistence_through_pipeline(temp_workspace):
    """Test that tallest/shortest persist correctly through pipeline stages."""
    # Session 1: Add initial entities
    store1 = SessionStore(temp_workspace)
    session1 = "height_session_1"
    store1.start(session_id=session1)
    store1.append_event(
        session1,
        "entity_attribute",
        {"entity": "Alice", "attribute": "height", "value": 165}
    )
    store1.append_event(
        session1,
        "entity_attribute",
        {"entity": "Bob", "attribute": "height", "value": 180}
    )

    # Consolidate session 1
    pipeline1 = ConsolidationPipeline(temp_workspace)
    pipeline1.consolidate_session(session1)

    # Verify initial tallest/shortest
    cache1 = RelationalCache(temp_workspace)
    stats1 = cache1.get_statistics()
    assert stats1["tallest"]["entity"] == "Bob"
    assert stats1["shortest"]["entity"] == "Alice"

    # Session 2: Add new tallest person
    store2 = SessionStore(temp_workspace)
    session2 = "height_session_2"
    store2.start(session_id=session2)
    store2.append_event(
        session2,
        "entity_attribute",
        {"entity": "Charlie", "attribute": "height", "value": 190}
    )

    # Consolidate session 2
    pipeline2 = ConsolidationPipeline(temp_workspace)
    pipeline2.consolidate_session(session2)

    # Verify updated tallest
    cache2 = RelationalCache(temp_workspace)
    stats2 = cache2.get_statistics()
    assert stats2["tallest"]["entity"] == "Charlie"
    assert stats2["shortest"]["entity"] == "Alice"

    # Session 3: Add new shortest person
    store3 = SessionStore(temp_workspace)
    session3 = "height_session_3"
    store3.start(session_id=session3)
    store3.append_event(
        session3,
        "entity_attribute",
        {"entity": "Diana", "attribute": "height", "value": 155}
    )

    # Consolidate session 3
    pipeline3 = ConsolidationPipeline(temp_workspace)
    pipeline3.consolidate_session(session3)

    # Verify final tallest/shortest
    cache3 = RelationalCache(temp_workspace)
    stats3 = cache3.get_statistics()
    assert stats3["tallest"]["entity"] == "Charlie"
    assert stats3["tallest"]["height"] == 190
    assert stats3["shortest"]["entity"] == "Diana"
    assert stats3["shortest"]["height"] == 155


def test_cycle_detection_through_sessions(temp_workspace):
    """Test that relationship cycles can be detected after session consolidation."""
    store = SessionStore(temp_workspace)
    pipeline = ConsolidationPipeline(temp_workspace)

    # Create session with relationships that form a cycle
    session_id = "cycle_test"
    store.start(session_id=session_id)
    store.append_event(
        session_id,
        "entity_relation",
        {"source": "A", "target": "B", "relation_type": "points_to"}
    )
    store.append_event(
        session_id,
        "entity_relation",
        {"source": "B", "target": "C", "relation_type": "points_to"}
    )
    store.append_event(
        session_id,
        "entity_relation",
        {"source": "C", "target": "A", "relation_type": "points_to"}
    )

    # Consolidate
    pipeline.consolidate_session(session_id)

    # Check for cycles
    cache = RelationalCache(temp_workspace)
    cycles = cache.detect_cycles()

    assert len(cycles) > 0
    cycle_entities = set(cycles[0])
    assert cycle_entities == {"A", "B", "C"}


def test_pipeline_handles_empty_sessions(pipeline, session_store):
    """Test that pipeline gracefully handles sessions with no events."""
    session_id = "empty_session"
    session_store.start(session_id=session_id)

    # Consolidate empty session
    result = pipeline.consolidate_session(session_id)

    assert result["patterns_added"] == 0
    assert result["relationships_added"] == 0


def test_pipeline_handles_unknown_event_types(pipeline, session_store):
    """Test that pipeline handles unknown event types gracefully."""
    session_id = "unknown_events"
    session_store.start(session_id=session_id)

    # Add event with unknown type
    session_store.append_event(
        session_id,
        "unknown_type",
        {"data": "should be ignored"}
    )

    # Should not crash
    result = pipeline.consolidate_session(session_id)

    assert result["patterns_added"] == 0
    assert result["relationships_added"] == 0
