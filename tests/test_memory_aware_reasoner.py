"""Tests for memory-aware reasoner integration."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner, wrap_latent_reasoner_with_memory
from nanobot.memory.relational_cache import RelationalCache


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def populated_workspace(temp_workspace):
    """Create a workspace with test data."""
    cache = RelationalCache(temp_workspace)

    # Add test data
    cache.update_entity_attribute("Alice", "height", 165)
    cache.update_entity_attribute("Bob", "height", 180)
    cache.add_relationship("Alice", "Bob", "friend_of")

    return temp_workspace


@pytest.fixture
def memory_reasoner(populated_workspace):
    """Create a MemoryAwareReasoner instance."""
    return MemoryAwareReasoner(
        workspace=populated_workspace,
        memory_config={"clarify_entropy_threshold": 0.8}
    )


def test_memory_reasoner_initialization(temp_workspace):
    """Test that MemoryAwareReasoner initializes correctly."""
    reasoner = MemoryAwareReasoner(
        workspace=temp_workspace,
        memory_config={"clarify_entropy_threshold": 0.7}
    )

    assert reasoner.hypothesis_engine is not None
    assert reasoner.entropy_threshold == 0.7


def test_memory_reasoner_no_workspace():
    """Test initialization without workspace."""
    reasoner = MemoryAwareReasoner()

    assert reasoner.hypothesis_engine is None


def test_check_memory_high_confidence(memory_reasoner):
    """Test that memory check succeeds with high confidence query."""
    can_answer, state = memory_reasoner.check_memory_first("What is Bob's height?")

    assert can_answer
    assert state is not None
    assert isinstance(state, SuperpositionalState)
    assert len(state.hypotheses) > 0
    assert state.entropy < 0.5  # Low entropy


def test_check_memory_low_confidence(memory_reasoner):
    """Test that memory check fails with low confidence query."""
    can_answer, state = memory_reasoner.check_memory_first("What is the meaning of life?")

    # Should not be able to answer from cache
    assert not can_answer or state is None


def test_check_memory_comparison(memory_reasoner):
    """Test memory check with comparison query."""
    can_answer, state = memory_reasoner.check_memory_first("Who is taller, Alice or Bob?")

    assert can_answer
    assert state is not None
    assert state.entropy < 0.5
    # Should have high confidence
    assert state.hypotheses[0].confidence > 0.8


def test_check_memory_without_engine():
    """Test check_memory_first when engine is not initialized."""
    reasoner = MemoryAwareReasoner()  # No workspace

    can_answer, state = reasoner.check_memory_first("Any query")

    assert not can_answer
    assert state is None


def test_strategic_direction_comparison(memory_reasoner):
    """Test strategic direction generation for comparison queries."""
    result = {
        "hypotheses": [
            {
                "intent": "compare_height",
                "confidence": 0.95,
                "reasoning": "Direct comparison",
                "result": "Bob is taller than Alice"
            }
        ],
        "query_type": "comparison"
    }

    direction = memory_reasoner._get_strategic_direction(result)

    assert "comparison" in direction.lower()
    assert "cache" in direction.lower()


def test_strategic_direction_attribute(memory_reasoner):
    """Test strategic direction generation for attribute queries."""
    result = {
        "hypotheses": [
            {
                "intent": "get_attribute",
                "confidence": 0.95,
                "reasoning": "Direct lookup",
                "result": "Alice's height is 165"
            }
        ],
        "query_type": "attribute"
    }

    direction = memory_reasoner._get_strategic_direction(result)

    assert "attribute" in direction.lower()
    assert "cache" in direction.lower()


def test_strategic_direction_no_hypotheses(memory_reasoner):
    """Test strategic direction when no hypotheses are available."""
    result = {
        "hypotheses": [],
        "query_type": "general"
    }

    direction = memory_reasoner._get_strategic_direction(result)

    assert "no cached information" in direction.lower()


@pytest.mark.asyncio
async def test_wrap_latent_reasoner_cache_hit(populated_workspace):
    """Test wrapping LatentReasoner when cache can answer."""
    # Track if LLM was called
    llm_called = {"count": 0}

    async def mock_reason(user_message, context_summary):
        llm_called["count"] += 1
        return SuperpositionalState(
            hypotheses=[Hypothesis(intent="llm_answer", confidence=0.8, reasoning="LLM reasoning")],
            entropy=0.5,
            strategic_direction="LLM answer"
        )

    # Create a mock LatentReasoner
    mock_reasoner = Mock()
    mock_reasoner.reason = mock_reason

    # Wrap it
    wrapped = wrap_latent_reasoner_with_memory(
        mock_reasoner,
        workspace=populated_workspace,
        memory_config={"clarify_entropy_threshold": 0.8}
    )

    # Call with a query that cache can answer
    result = await wrapped.reason("What is Bob's height?", "context")

    # Should not have called original reason (cache answered)
    assert llm_called["count"] == 0

    # Should have result from cache
    assert result.entropy < 0.5
    assert len(result.hypotheses) > 0


@pytest.mark.asyncio
async def test_wrap_latent_reasoner_cache_miss(populated_workspace):
    """Test wrapping LatentReasoner when cache cannot answer."""
    # Track if LLM was called
    llm_called = {"count": 0}

    llm_state = SuperpositionalState(
        hypotheses=[Hypothesis(intent="llm_answer", confidence=0.8, reasoning="LLM reasoning")],
        entropy=0.5,
        strategic_direction="LLM answer"
    )

    async def mock_reason(user_message, context_summary):
        llm_called["count"] += 1
        return llm_state

    # Create a mock LatentReasoner
    mock_reasoner = Mock()
    mock_reasoner.reason = mock_reason

    # Wrap it
    wrapped = wrap_latent_reasoner_with_memory(
        mock_reasoner,
        workspace=populated_workspace,
        memory_config={"clarify_entropy_threshold": 0.8}
    )

    # Call with a query that cache cannot answer
    result = await wrapped.reason("What is the meaning of life?", "context")

    # Should have called original reason (cache missed)
    assert llm_called["count"] == 1

    # Should have result from LLM
    assert result == llm_state


@pytest.mark.asyncio
async def test_wrap_latent_reasoner_no_workspace():
    """Test wrapping LatentReasoner without workspace."""
    # Track if LLM was called
    llm_called = {"count": 0}

    llm_state = SuperpositionalState(
        hypotheses=[Hypothesis(intent="llm_answer", confidence=0.8, reasoning="LLM")],
        entropy=0.5,
        strategic_direction="LLM"
    )

    async def mock_reason(user_message, context_summary):
        llm_called["count"] += 1
        return llm_state

    # Create a mock LatentReasoner
    mock_reasoner = Mock()
    mock_reasoner.reason = mock_reason

    # Wrap without workspace
    wrapped = wrap_latent_reasoner_with_memory(mock_reasoner)

    # Should always call LLM since no cache
    result = await wrapped.reason("Any query", "context")

    assert llm_called["count"] == 1
    assert result == llm_state


def test_hypothesis_conversion(memory_reasoner):
    """Test that hypothesis engine hypotheses are converted correctly."""
    can_answer, state = memory_reasoner.check_memory_first("What is Alice's height?")

    if can_answer and state:
        # Check hypothesis structure
        assert len(state.hypotheses) > 0
        hyp = state.hypotheses[0]

        assert isinstance(hyp, Hypothesis)
        assert hasattr(hyp, "intent")
        assert hasattr(hyp, "confidence")
        assert hasattr(hyp, "reasoning")
        assert hyp.confidence > 0


def test_entropy_threshold_respected(populated_workspace):
    """Test that entropy threshold is respected."""
    # Create reasoner with low threshold
    reasoner = MemoryAwareReasoner(
        workspace=populated_workspace,
        memory_config={"clarify_entropy_threshold": 0.1}  # Very low threshold
    )

    # Even good queries might not meet the low threshold
    can_answer, state = reasoner.check_memory_first("What is Bob's height?")

    # Should either answer or not, but entropy logic should be consistent
    if can_answer:
        assert state.entropy < 0.1


def test_multiple_hypotheses(memory_reasoner):
    """Test that multiple hypotheses are returned when available."""
    can_answer, state = memory_reasoner.check_memory_first(
        "How are Alice and Bob related?"
    )

    if can_answer and state:
        # May have multiple hypotheses for relationships
        assert len(state.hypotheses) >= 1
        # All hypotheses should be valid
        for hyp in state.hypotheses:
            assert 0 <= hyp.confidence <= 1
            assert isinstance(hyp.intent, str)
            assert isinstance(hyp.reasoning, str)
