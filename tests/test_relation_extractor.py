"""Tests for relation extraction engine."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from nanobot.memory.relation_extractor import RelationExtractionEngine
from nanobot.memory.relational_cache import RelationalCache


@pytest.fixture
def extractor():
    """Create a RelationExtractionEngine instance."""
    return RelationExtractionEngine()


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_workspace):
    """Create a RelationalCache instance."""
    return RelationalCache(temp_workspace)


def test_extract_taller_than(extractor):
    """Test extraction of 'taller than' relation."""
    text = "Alice taller than Bob"
    relations = extractor.extract(text)
    
    assert len(relations) == 2  # Primary + inverse
    
    # Check primary relation
    primary = next(r for r in relations if r["relation_type"] == "taller_than")
    assert primary["source"] == "Alice"
    assert primary["target"] == "Bob"
    assert primary["properties"]["confidence"] == 1.0
    
    # Check inverse relation
    inverse = next(r for r in relations if r["relation_type"] == "shorter_than")
    assert inverse["source"] == "Bob"
    assert inverse["target"] == "Alice"


def test_extract_with_is(extractor):
    """Test extraction with 'is' included."""
    text = "Alice is taller than Bob"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "taller_than")
    assert primary["source"] == "Alice"
    assert primary["target"] == "Bob"


def test_extract_shorter_than(extractor):
    """Test extraction of 'shorter than' relation."""
    text = "Bob is shorter than Carol"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    
    primary = next(r for r in relations if r["relation_type"] == "shorter_than")
    assert primary["source"] == "Bob"
    assert primary["target"] == "Carol"
    
    inverse = next(r for r in relations if r["relation_type"] == "taller_than")
    assert inverse["source"] == "Carol"
    assert inverse["target"] == "Bob"


def test_extract_older_than(extractor):
    """Test extraction of 'older than' relation."""
    text = "Dave older than Eve"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "older_than")
    assert primary["source"] == "Dave"
    assert primary["target"] == "Eve"


def test_extract_younger_than(extractor):
    """Test extraction of 'younger than' relation."""
    text = "Frank is younger than George"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "younger_than")
    assert primary["source"] == "Frank"
    assert primary["target"] == "George"


def test_extract_bigger_than(extractor):
    """Test extraction of 'bigger than' relation."""
    text = "The house is bigger than the shed"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "bigger_than")
    assert primary["source"] == "house"
    assert primary["target"] == "shed"


def test_extract_smaller_than(extractor):
    """Test extraction of 'smaller than' relation."""
    text = "The cat is smaller than the dog"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "smaller_than")
    assert primary["source"] == "cat"
    assert primary["target"] == "dog"


def test_extract_faster_than(extractor):
    """Test extraction of 'faster than' relation."""
    text = "The cheetah is faster than the lion"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "faster_than")
    assert primary["source"] == "cheetah"
    assert primary["target"] == "lion"


def test_extract_slower_than(extractor):
    """Test extraction of 'slower than' relation."""
    text = "The turtle is slower than the rabbit"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "slower_than")
    assert primary["source"] == "turtle"
    assert primary["target"] == "rabbit"


def test_extract_heavier_than(extractor):
    """Test extraction of 'heavier than' relation."""
    text = "The elephant is heavier than the horse"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "heavier_than")
    assert primary["source"] == "elephant"
    assert primary["target"] == "horse"


def test_extract_lighter_than(extractor):
    """Test extraction of 'lighter than' relation."""
    text = "The feather is lighter than the rock"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "lighter_than")
    assert primary["source"] == "feather"
    assert primary["target"] == "rock"


def test_extract_stronger_than(extractor):
    """Test extraction of 'stronger than' relation."""
    text = "Superman is stronger than Batman"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "stronger_than")
    assert primary["source"] == "Superman"
    assert primary["target"] == "Batman"


def test_extract_weaker_than(extractor):
    """Test extraction of 'weaker than' relation."""
    text = "The kitten is weaker than the cat"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "weaker_than")
    assert primary["source"] == "kitten"
    assert primary["target"] == "cat"


def test_case_insensitive(extractor):
    """Test that extraction is case-insensitive."""
    text = "ALICE TALLER THAN BOB"
    relations = extractor.extract(text)
    
    assert len(relations) == 2
    primary = next(r for r in relations if r["relation_type"] == "taller_than")
    # Entity casing should be preserved from the text
    assert primary["source"] == "ALICE"
    assert primary["target"] == "BOB"


def test_no_relations_in_unrelated_text(extractor):
    """Test that no relations are extracted from unrelated text."""
    text = "The weather is nice today."
    relations = extractor.extract(text)
    
    assert len(relations) == 0


def test_empty_text(extractor):
    """Test extraction from empty text."""
    relations = extractor.extract("")
    assert len(relations) == 0


def test_multiple_relations(extractor):
    """Test extraction of multiple relations from a single text."""
    text = "Alice is taller than Bob and Bob is older than Carol"
    relations = extractor.extract(text)
    
    # Should have 4 relations: 2 primary + 2 inverses
    assert len(relations) == 4
    
    # Check we have both relation types
    assert any(r["relation_type"] == "taller_than" for r in relations)
    assert any(r["relation_type"] == "older_than" for r in relations)


def test_same_entity_filtered(extractor):
    """Test that relations with same source and target are filtered."""
    text = "Alice is taller than Alice"
    relations = extractor.extract(text)
    
    # Should be filtered out
    assert len(relations) == 0


def test_extract_and_ingest_with_cache(extractor, cache):
    """Test extract_and_ingest with cache."""
    text = "Alice is taller than Bob"
    count = extractor.extract_and_ingest(text, cache)
    
    assert count == 2  # Primary + inverse
    
    # Verify relations were added to cache
    alice_rels = cache.get_entity_relationships("Alice")
    bob_rels = cache.get_entity_relationships("Bob")
    
    assert len(alice_rels) > 0
    assert len(bob_rels) > 0


def test_extract_and_ingest_with_session_store(extractor, cache, temp_workspace):
    """Test extract_and_ingest with cache and session store."""
    from nanobot.memory.session_store import SessionStore
    
    session_store = SessionStore(temp_workspace)
    session_id = "test_session"
    session_store.start(session_id)
    
    text = "Alice is taller than Bob"
    count = extractor.extract_and_ingest(
        text,
        cache,
        session_store=session_store,
        session_id=session_id
    )
    
    assert count == 2
    
    # Verify events were added to session
    session_data = session_store.load(session_id)
    events = session_data.get("events", [])
    
    # Should have entity_relation events
    relation_events = [e for e in events if e.get("type") == "entity_relation"]
    assert len(relation_events) == 2


def test_extract_and_ingest_no_relations(extractor, cache):
    """Test extract_and_ingest with text containing no relations."""
    text = "The weather is nice today"
    count = extractor.extract_and_ingest(text, cache)
    
    assert count == 0


def test_extract_and_ingest_without_session_store(extractor, cache):
    """Test extract_and_ingest without session store."""
    text = "Alice is taller than Bob"
    count = extractor.extract_and_ingest(text, cache, session_store=None, session_id=None)
    
    assert count == 2
    
    # Verify relations were still added to cache
    alice_rels = cache.get_entity_relationships("Alice")
    assert len(alice_rels) > 0


def test_inverse_relation_properties(extractor):
    """Test that inverse relations have proper properties."""
    text = "Alice is taller than Bob"
    relations = extractor.extract(text)
    
    inverse = next(r for r in relations if r["relation_type"] == "shorter_than")
    assert "inverse_of" in inverse["properties"]
    assert inverse["properties"]["inverse_of"] == "taller_than"


def test_context_snippet_stored(extractor):
    """Test that context snippet is stored in properties."""
    text = "Alice is taller than Bob in the class photo"
    relations = extractor.extract(text)
    
    for relation in relations:
        assert "extracted_from" in relation["properties"]
        assert len(relation["properties"]["extracted_from"]) <= 100
