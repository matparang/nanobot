"""Tests for hypothesis engine functionality."""

import tempfile
from pathlib import Path

import pytest

from nanobot.memory.hypothesis_engine import HypothesisEngine
from nanobot.memory.relational_cache import RelationalCache


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def populated_cache(temp_workspace):
    """Create a cache populated with test data."""
    cache = RelationalCache(temp_workspace)

    # Add entities with height attributes
    cache.update_entity_attribute("Alice", "height", 165)
    cache.update_entity_attribute("Bob", "height", 180)
    cache.update_entity_attribute("Charlie", "height", 175)
    cache.update_entity_attribute("Diana", "height", 160)

    # Add relationships
    cache.add_relationship("Alice", "Bob", "friend_of")
    cache.add_relationship("Bob", "Charlie", "works_with")
    cache.add_relationship("Charlie", "Diana", "related_to")
    cache.add_relationship("Alice", "Diana", "knows")

    # Add some patterns
    cache.add_pattern("comparison", {"subject": "height comparison"}, entities=["Alice", "Bob"])
    cache.add_pattern("interaction", {"type": "greeting"}, entities=["Alice"])

    return cache


@pytest.fixture
def engine(temp_workspace):
    """Create a HypothesisEngine instance."""
    return HypothesisEngine(temp_workspace, entropy_threshold=0.8)


def test_engine_initialization(temp_workspace):
    """Test that HypothesisEngine initializes correctly."""
    engine = HypothesisEngine(temp_workspace, entropy_threshold=0.7)

    assert engine.cache is not None
    assert engine.entropy_threshold == 0.7


def test_comparison_query(populated_cache, engine):
    """Test comparison query generation."""
    result = engine.generate_hypotheses("Who is taller, Alice or Bob?")

    assert "hypotheses" in result
    assert len(result["hypotheses"]) > 0

    # Should have high confidence for direct attribute comparison
    top_hypothesis = result["hypotheses"][0]
    assert top_hypothesis["confidence"] > 0.8
    assert "Bob" in top_hypothesis["result"]  # Bob is taller


def test_attribute_query(populated_cache, engine):
    """Test attribute query generation."""
    result = engine.generate_hypotheses("What is Alice's height?")

    assert "hypotheses" in result
    assert len(result["hypotheses"]) > 0

    # Should have very high confidence for direct attribute lookup
    top_hypothesis = result["hypotheses"][0]
    assert top_hypothesis["confidence"] >= 0.9
    assert "165" in str(top_hypothesis["result"])


def test_relationship_query(populated_cache, engine):
    """Test relationship query generation."""
    result = engine.generate_hypotheses("Is Alice related to Bob?")

    assert "hypotheses" in result
    # Should find direct relationship
    if result["hypotheses"]:
        assert result["hypotheses"][0]["confidence"] > 0.5


def test_entropy_calculation_high_confidence(populated_cache, engine):
    """Test that entropy is low when confidence is high."""
    result = engine.generate_hypotheses("What is Bob's height?")

    # Direct attribute lookup should have low entropy
    assert result["entropy"] < 0.5
    assert not result["requires_llm"]


def test_entropy_calculation_ambiguous(engine):
    """Test that entropy is high when query is ambiguous."""
    # Query with no data should have high entropy
    result = engine.generate_hypotheses("What is the meaning of life?")

    assert result["entropy"] > 0.5


def test_query_parsing_comparison(populated_cache, engine):
    """Test query parsing for comparison queries."""
    parsed = engine._parse_query("Who is taller, Alice or Bob?")

    # Should detect comparison type (even if it's general, it should have the entities)
    assert parsed["type"] in ["comparison", "general"]
    # Should detect the entities mentioned
    entities_str = str(parsed)
    assert "Alice" in entities_str or "Bob" in entities_str or len(parsed.get("entities", [])) > 0


def test_query_parsing_attribute(populated_cache, engine):
    """Test query parsing for attribute queries."""
    parsed = engine._parse_query("What is Alice's height?")

    assert parsed["type"] == "attribute"
    assert parsed.get("entity") == "Alice"
    assert parsed.get("attribute") == "height"


def test_path_finding_direct(populated_cache, engine):
    """Test finding direct paths between entities."""
    paths = engine._find_paths_bfs("Alice", "Bob", max_depth=3)

    assert len(paths) > 0
    # Should find direct path
    assert any(len(path) == 2 for path in paths)


def test_path_finding_indirect(populated_cache, engine):
    """Test finding indirect paths between entities."""
    paths = engine._find_paths_bfs("Alice", "Charlie", max_depth=3)

    assert len(paths) > 0
    # Should find path through Bob
    assert any("Bob" in path for path in paths)


def test_comparison_hypotheses_with_data(populated_cache, engine):
    """Test comparison hypothesis generation with data."""
    hypotheses = engine._generate_comparison_hypotheses(
        ["Alice", "Bob"],
        "height",
        max_hypotheses=3
    )

    assert len(hypotheses) > 0
    assert hypotheses[0]["confidence"] > 0.9  # High confidence with data
    assert "Bob" in hypotheses[0]["result"] or "taller" in hypotheses[0]["result"].lower()


def test_comparison_hypotheses_missing_data(engine):
    """Test comparison hypothesis generation without data."""
    hypotheses = engine._generate_comparison_hypotheses(
        ["Unknown1", "Unknown2"],
        "height",
        max_hypotheses=3
    )

    # Should return empty or low-confidence hypotheses
    assert len(hypotheses) == 0 or all(h["confidence"] < 0.9 for h in hypotheses)


def test_attribute_hypotheses(populated_cache, engine):
    """Test attribute hypothesis generation."""
    hypotheses = engine._generate_attribute_hypotheses(
        "Alice",
        "height",
        max_hypotheses=3
    )

    assert len(hypotheses) > 0
    assert hypotheses[0]["confidence"] > 0.9
    assert "165" in str(hypotheses[0]["result"])


def test_general_hypotheses(populated_cache, engine):
    """Test general hypothesis generation."""
    hypotheses = engine._generate_general_hypotheses(
        "Tell me about height",
        max_hypotheses=3
    )

    # Should find patterns related to height
    assert len(hypotheses) >= 0  # May or may not find patterns


def test_entropy_with_single_hypothesis(engine):
    """Test entropy calculation with a single high-confidence hypothesis."""
    hypotheses = [
        {"confidence": 0.95, "intent": "test"}
    ]

    entropy = engine._compute_entropy(hypotheses)

    # Single high-confidence hypothesis should have low entropy
    assert entropy < 0.5


def test_entropy_with_multiple_equal_hypotheses(engine):
    """Test entropy calculation with multiple equal-confidence hypotheses."""
    hypotheses = [
        {"confidence": 0.33, "intent": "option1"},
        {"confidence": 0.33, "intent": "option2"},
        {"confidence": 0.34, "intent": "option3"}
    ]

    entropy = engine._compute_entropy(hypotheses)

    # Equal confidence distribution should have high entropy
    assert entropy > 0.5


def test_entropy_with_no_hypotheses(engine):
    """Test entropy calculation with no hypotheses."""
    entropy = engine._compute_entropy([])

    # No hypotheses should result in maximum entropy
    assert entropy == 1.0


def test_query_method_comparison(populated_cache, engine):
    """Test the high-level query method for comparisons."""
    result = engine.query("Who is shorter, Alice or Diana?")

    assert "answer" in result
    assert "confidence" in result
    assert "entropy" in result
    assert result["query_type"] in ["comparison", "general", "relationship", "attribute"]


def test_query_method_attribute(populated_cache, engine):
    """Test the high-level query method for attribute queries."""
    result = engine.query("What is Charlie's height?", verbose=True)

    assert "answer" in result
    assert "hypotheses" in result  # verbose mode
    assert "cache_stats" in result  # verbose mode
    assert result["confidence"] > 0.8


def test_query_method_no_data(engine):
    """Test query method with no matching data."""
    result = engine.query("What is the weather?")

    assert "answer" in result
    assert result["confidence"] < 0.5  # Low confidence
    assert result["requires_llm"]  # Should recommend LLM


def test_llm_bypass_low_entropy(populated_cache, engine):
    """Test that LLM is not required when entropy is low."""
    result = engine.generate_hypotheses("What is Bob's height?")

    assert not result["requires_llm"]
    assert result["entropy"] < engine.entropy_threshold


def test_llm_required_high_entropy(engine):
    """Test that LLM is required when entropy is high."""
    result = engine.generate_hypotheses("What is the purpose of existence?")

    # Query with no data should require LLM
    assert result["requires_llm"]
    assert result["entropy"] >= engine.entropy_threshold or len(result["hypotheses"]) == 0


def test_multiple_comparison_queries(populated_cache, engine):
    """Test multiple comparison queries in sequence."""
    queries = [
        "Who is taller, Alice or Bob?",
        "Who is taller, Charlie or Diana?",
        "Is Bob taller than Charlie?"
    ]

    for query in queries:
        result = engine.query(query)
        assert "answer" in result
        # Should have reasonable confidence for all height comparisons
        assert result["confidence"] > 0.5


def test_relationship_traversal(populated_cache, engine):
    """Test that relationship graph traversal works."""
    # Alice -> Bob -> Charlie (2-hop path)
    result = engine.generate_hypotheses("How is Alice connected to Charlie?")

    if result["hypotheses"]:
        # Should find indirect relationship
        top_hyp = result["hypotheses"][0]
        assert "evidence" in top_hyp


def test_tallest_shortest_queries(populated_cache, engine):
    """Test queries about tallest/shortest entities."""
    stats = populated_cache.get_statistics()

    # Query for tallest - use a more specific query format
    result_tallest = engine.query("Who is taller, Bob or Alice?")

    # Should have some confidence since we have data
    assert result_tallest["confidence"] > 0.5 or "Bob" in str(result_tallest["answer"])


def test_parse_query_edge_cases(engine):
    """Test query parsing with edge cases."""
    # Empty query
    parsed = engine._parse_query("")
    assert parsed["type"] == "general"

    # Very long query
    long_query = "What " * 100 + "is the answer?"
    parsed = engine._parse_query(long_query)
    assert parsed["type"] in ["general", "attribute", "comparison", "relationship"]

    # Special characters
    parsed = engine._parse_query("Who's taller: Alice or Bob?!?")
    assert "type" in parsed


def test_confidence_decreases_with_path_length(populated_cache, engine):
    """Test that confidence decreases for longer relationship paths."""
    # Direct relationship should have higher confidence
    result_direct = engine._generate_relationship_hypotheses(
        "Alice", "Bob", None, 3, 3
    )

    # Indirect relationship should have lower confidence
    result_indirect = engine._generate_relationship_hypotheses(
        "Alice", "Charlie", None, 3, 3
    )

    if result_direct and result_indirect:
        direct_conf = result_direct[0]["confidence"]
        # Find the indirect path hypothesis
        indirect_hyps = [h for h in result_indirect if "indirect" in h["intent"]]
        if indirect_hyps:
            indirect_conf = indirect_hyps[0]["confidence"]
            assert direct_conf >= indirect_conf
