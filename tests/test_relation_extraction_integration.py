"""Integration tests for relation extraction and memory-driven reasoning."""

import tempfile
from pathlib import Path

import pytest

from nanobot.memory.consolidation import ConsolidationPipeline
from nanobot.memory.hypothesis_engine import HypothesisEngine
from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner
from nanobot.memory.relation_extractor import RelationExtractionEngine
from nanobot.memory.relational_cache import RelationalCache
from nanobot.memory.session_store import SessionStore


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def extractor():
    """Create a RelationExtractionEngine instance."""
    return RelationExtractionEngine()


@pytest.fixture
def cache(temp_workspace):
    """Create a RelationalCache instance."""
    return RelationalCache(temp_workspace)


@pytest.fixture
def session_store(temp_workspace):
    """Create a SessionStore instance."""
    return SessionStore(temp_workspace)


@pytest.fixture
def engine(temp_workspace):
    """Create a HypothesisEngine instance."""
    return HypothesisEngine(temp_workspace, entropy_threshold=0.8)


@pytest.fixture
def pipeline(temp_workspace):
    """Create a ConsolidationPipeline instance."""
    return ConsolidationPipeline(temp_workspace)


def test_full_ingestion_to_query_cycle(extractor, cache, session_store, engine, temp_workspace):
    """Test complete flow: ingestion → query → deterministic answer."""
    session_id = "test_ingestion"
    session_store.start(session_id)
    
    # Ingest multiple relations
    count1 = extractor.extract_and_ingest(
        "Alice taller than Bob",
        cache,
        session_store=session_store,
        session_id=session_id
    )
    assert count1 == 2  # Primary + inverse
    
    count2 = extractor.extract_and_ingest(
        "Bob taller than Carol",
        cache,
        session_store=session_store,
        session_id=session_id
    )
    assert count2 == 2
    
    # Verify RelationalCache has relationships
    relationships = cache.get_entity_relationships("Alice")
    assert len(relationships) > 0
    
    # Query for superlative
    result = engine.generate_hypotheses("Who is tallest?")
    
    assert "hypotheses" in result
    assert len(result["hypotheses"]) > 0
    
    # Should have high confidence and low entropy
    top_hypothesis = result["hypotheses"][0]
    assert top_hypothesis["confidence"] > 0.9
    assert result["entropy"] < 0.5
    assert not result["requires_llm"]
    
    # Answer should be Alice
    assert "Alice" in top_hypothesis["result"]


def test_superlative_shortest_query(extractor, cache, session_store, engine):
    """Test superlative query for 'shortest'."""
    session_id = "test_shortest"
    session_store.start(session_id)
    
    # Build chain: Alice > Bob > Carol > Diana
    extractor.extract_and_ingest("Alice is taller than Bob", cache, session_store, session_id)
    extractor.extract_and_ingest("Bob is taller than Carol", cache, session_store, session_id)
    extractor.extract_and_ingest("Carol is taller than Diana", cache, session_store, session_id)
    
    # Query for shortest
    result = engine.generate_hypotheses("Who is shortest?")
    
    assert len(result["hypotheses"]) > 0
    assert result["entropy"] < 0.5
    assert not result["requires_llm"]
    
    # Answer should be Diana
    top_hypothesis = result["hypotheses"][0]
    assert "Diana" in top_hypothesis["result"]
    assert top_hypothesis["confidence"] > 0.9


def test_memory_aware_reasoner_bypass(extractor, cache, temp_workspace):
    """Test that MemoryAwareReasoner bypasses LLM when it can answer from cache."""
    # Populate cache with relations
    extractor.extract_and_ingest("Alice is taller than Bob", cache)
    extractor.extract_and_ingest("Bob is taller than Carol", cache)
    
    # Create MemoryAwareReasoner
    reasoner = MemoryAwareReasoner(
        workspace=temp_workspace,
        memory_config={"clarify_entropy_threshold": 0.8}
    )
    
    # Query "Who is tallest?"
    can_answer, state = reasoner.check_memory_first("Who is tallest?")
    
    # Should be able to answer from cache
    assert can_answer
    assert state is not None
    assert state.entropy < 0.5
    assert len(state.hypotheses) > 0
    
    # Top hypothesis should have high confidence
    assert state.hypotheses[0].confidence > 0.8


def test_consolidation_re_extraction(pipeline, session_store):
    """Test that consolidation re-extracts relations from interaction events."""
    session_id = "test_reextract"
    session_store.start(session_id)
    
    # Add interaction events with relational facts (NO entity_relation events)
    session_store.append_event(
        session_id,
        "interaction",
        {
            "user_message": "Alice is taller than Bob",
            "agent_response": "Noted."
        }
    )
    
    session_store.append_event(
        session_id,
        "interaction",
        {
            "user_message": "Bob is taller than Carol",
            "agent_response": "Understood."
        }
    )
    
    # Run consolidation with extract_entities=True
    result = pipeline.consolidate_session(session_id, extract_entities=True)
    
    # Should have re-extracted relationships
    assert result["relationships_reextracted"] > 0
    
    # Verify RelationalCache now contains extracted relationships
    alice_rels = pipeline.relational_cache.get_entity_relationships("Alice")
    bob_rels = pipeline.relational_cache.get_entity_relationships("Bob")
    
    assert len(alice_rels) > 0
    assert len(bob_rels) > 0


def test_transitive_inference_chain(extractor, cache, engine):
    """Test transitive inference over a chain of relationships."""
    # Build chain: A > B > C > D
    extractor.extract_and_ingest("Alice is taller than Bob", cache)
    extractor.extract_and_ingest("Bob is taller than Carol", cache)
    extractor.extract_and_ingest("Carol is taller than Diana", cache)
    
    # Query for tallest → should be Alice
    result_tallest = engine.generate_hypotheses("Who is tallest?")
    assert len(result_tallest["hypotheses"]) > 0
    assert "Alice" in result_tallest["hypotheses"][0]["result"]
    
    # Query for shortest → should be Diana
    result_shortest = engine.generate_hypotheses("Who is shortest?")
    assert len(result_shortest["hypotheses"]) > 0
    assert "Diana" in result_shortest["hypotheses"][0]["result"]


def test_cycle_detection_high_entropy(extractor, cache, engine):
    """Test that cycles in relationships result in high entropy (requires LLM)."""
    # Create a cycle: A > B > C > A
    extractor.extract_and_ingest("Alice is taller than Bob", cache)
    extractor.extract_and_ingest("Bob is taller than Carol", cache)
    extractor.extract_and_ingest("Carol is taller than Alice", cache)  # Creates cycle
    
    # Query for tallest
    result = engine.generate_hypotheses("Who is tallest?")
    
    # Should have low confidence due to inconsistent data
    assert result["requires_llm"]  # High entropy, need LLM


def test_different_relation_types(extractor, cache, engine):
    """Test extraction and querying of different relation types."""
    # Test age relations
    extractor.extract_and_ingest("Alice is older than Bob", cache)
    extractor.extract_and_ingest("Bob is older than Carol", cache)
    
    result_oldest = engine.generate_hypotheses("Who is oldest?")
    assert len(result_oldest["hypotheses"]) > 0
    assert "Alice" in result_oldest["hypotheses"][0]["result"]
    
    # Test speed relations
    extractor.extract_and_ingest("Cheetah is faster than Lion", cache)
    extractor.extract_and_ingest("Lion is faster than Elephant", cache)
    
    result_fastest = engine.generate_hypotheses("Who is fastest?")
    assert len(result_fastest["hypotheses"]) > 0
    assert "Cheetah" in result_fastest["hypotheses"][0]["result"]


def test_comparison_query_with_cached_relations(extractor, cache, engine):
    """Test comparison queries between specific entities."""
    extractor.extract_and_ingest("Alice is taller than Bob", cache)
    
    # Direct comparison query
    result = engine.generate_hypotheses("Who is taller, Alice or Bob?")
    
    assert len(result["hypotheses"]) > 0
    assert result["entropy"] < 0.5
    assert "Alice" in result["hypotheses"][0]["result"] or "taller" in result["hypotheses"][0]["result"]


def test_insufficient_data_high_entropy(engine):
    """Test that queries with no data return high entropy."""
    result = engine.generate_hypotheses("Who is tallest?")
    
    # No data available, should require LLM
    assert result["requires_llm"]


def test_session_events_include_entity_relations(extractor, cache, session_store):
    """Test that entity_relation events are written to session store."""
    session_id = "test_events"
    session_store.start(session_id)
    
    extractor.extract_and_ingest(
        "Alice is taller than Bob",
        cache,
        session_store=session_store,
        session_id=session_id
    )
    
    # Load session and check for entity_relation events
    session_data = session_store.load(session_id)
    events = session_data.get("events", [])
    
    relation_events = [e for e in events if e.get("type") == "entity_relation"]
    assert len(relation_events) == 2  # Primary + inverse
    
    # Verify event structure
    for event in relation_events:
        payload = event.get("payload", {})
        assert "source" in payload
        assert "target" in payload
        assert "relation_type" in payload


def test_multiple_entities_superlative(extractor, cache, engine):
    """Test superlative with multiple entities in the chain."""
    # Create a larger chain
    extractor.extract_and_ingest("Alice is taller than Bob", cache)
    extractor.extract_and_ingest("Bob is taller than Carol", cache)
    extractor.extract_and_ingest("Carol is taller than Diana", cache)
    extractor.extract_and_ingest("Diana is taller than Eve", cache)
    extractor.extract_and_ingest("Eve is taller than Frank", cache)
    
    result_tallest = engine.generate_hypotheses("Who is tallest?")
    assert "Alice" in result_tallest["hypotheses"][0]["result"]
    
    result_shortest = engine.generate_hypotheses("Who is shortest?")
    assert "Frank" in result_shortest["hypotheses"][0]["result"]


def test_inverse_relations_stored(extractor, cache):
    """Test that inverse relations are properly stored in cache."""
    extractor.extract_and_ingest("Alice is taller than Bob", cache)
    
    # Check Alice's relationships
    alice_rels = cache.get_entity_relationships("Alice")
    taller_rels = [r for r in alice_rels if r.get("type") == "taller_than"]
    assert len(taller_rels) > 0
    assert any(r.get("target") == "Bob" for r in taller_rels)
    
    # Check Bob's relationships (should have inverse)
    bob_rels = cache.get_entity_relationships("Bob")
    shorter_rels = [r for r in bob_rels if r.get("type") == "shorter_than"]
    assert len(shorter_rels) > 0
    assert any(r.get("target") == "Alice" for r in shorter_rels)


def test_consolidation_without_reextraction(pipeline, session_store):
    """Test consolidation when extract_entities=False."""
    session_id = "test_no_reextract"
    session_store.start(session_id)
    
    session_store.append_event(
        session_id,
        "interaction",
        {
            "user_message": "Alice is taller than Bob",
            "agent_response": "Noted."
        }
    )
    
    # Run consolidation with extract_entities=False
    result = pipeline.consolidate_session(session_id, extract_entities=False)
    
    # Should NOT have re-extracted relationships
    assert result.get("relationships_reextracted", 0) == 0


def test_end_to_end_alice_bob_carol_scenario(temp_workspace):
    """Full end-to-end test: ingest → consolidate → query."""
    # Setup
    session_store = SessionStore(temp_workspace)
    extractor = RelationExtractionEngine()
    cache = RelationalCache(temp_workspace)
    pipeline = ConsolidationPipeline(temp_workspace)
    engine = HypothesisEngine(temp_workspace)
    
    session_id = "e2e_test"
    session_store.start(session_id)
    
    # Step 1: Eager ingestion (simulating AgentLoop behavior)
    extractor.extract_and_ingest(
        "Alice is taller than Bob",
        cache,
        session_store=session_store,
        session_id=session_id
    )
    
    extractor.extract_and_ingest(
        "Bob is taller than Carol",
        cache,
        session_store=session_store,
        session_id=session_id
    )
    
    # Step 2: Query before consolidation (should work from eager ingestion)
    result_pre = engine.generate_hypotheses("Who is tallest?")
    assert not result_pre["requires_llm"]
    assert "Alice" in result_pre["hypotheses"][0]["result"]
    
    # Step 3: Consolidate session (self-healing)
    consolidation_result = pipeline.consolidate_session(session_id, extract_entities=True)
    
    # Step 4: Query after consolidation (should still work)
    result_post = engine.generate_hypotheses("Who is tallest?")
    assert not result_post["requires_llm"]
    assert "Alice" in result_post["hypotheses"][0]["result"]
    
    # Verify confidence is high
    assert result_post["hypotheses"][0]["confidence"] > 0.9
    assert result_post["entropy"] < 0.5
