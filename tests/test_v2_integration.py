"""Integration tests for v2 memory-first reasoning flag toggling.

Tests that v2 can be enabled/disabled via configuration and that both modes work correctly.
"""

import pytest
from pathlib import Path
import tempfile

from nanobot.memory.memory_aware_reasoner import MemoryAwareReasoner
from nanobot.memory.consolidation import ConsolidationPipeline
from nanobot.memory.types_v2 import RelationType, IngestionStatus


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMemoryAwareReasonerV2Integration:
    """Test MemoryAwareReasoner with v2 flag."""
    
    def test_initialization_v1_mode(self, temp_workspace):
        """Test that v1 components are initialized when v2 flag is False."""
        memory_config = {"use_memory_v2": False}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        assert reasoner.use_memory_v2 is False
        assert reasoner.hypothesis_engine is not None
        assert reasoner.reasoner_v2 is None
        assert reasoner.cache_v2 is None
    
    def test_initialization_v2_mode(self, temp_workspace):
        """Test that v2 components are initialized when v2 flag is True."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        assert reasoner.use_memory_v2 is True
        assert reasoner.hypothesis_engine is None
        assert reasoner.reasoner_v2 is not None
        assert reasoner.cache_v2 is not None
    
    def test_initialization_default_mode(self, temp_workspace):
        """Test that v1 mode is default when flag is not specified."""
        memory_config = {}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        assert reasoner.use_memory_v2 is False
        assert reasoner.hypothesis_engine is not None
    
    def test_v2_superlative_query_tallest(self, temp_workspace):
        """Test v2 superlative query returns deterministic answer."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Add relations to v2 cache
        reasoner.cache_v2.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        reasoner.cache_v2.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        # Query for tallest
        can_answer, state = reasoner.check_memory_first("Who is tallest?")
        
        assert can_answer is True
        assert state is not None
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].intent == "superlative_tallest"
        assert state.hypotheses[0].confidence == 0.95
        assert "Alice" in state.hypotheses[0].reasoning
    
    def test_v2_superlative_query_shortest(self, temp_workspace):
        """Test v2 superlative query for shortest."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Add relations
        reasoner.cache_v2.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        reasoner.cache_v2.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        # Query for shortest
        can_answer, state = reasoner.check_memory_first("Who is shortest?")
        
        assert can_answer is True
        assert state is not None
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].intent == "superlative_shortest"
        assert "Carol" in state.hypotheses[0].reasoning
    
    def test_v2_superlative_query_ambiguous_returns_no_answer(self, temp_workspace):
        """Test v2 returns no answer for ambiguous superlatives."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Add relations that create ambiguity
        reasoner.cache_v2.add_relation("Alice", RelationType.TALLER_THAN, "Carol")
        reasoner.cache_v2.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        # Query for tallest - should return no answer because Alice and Bob are ambiguous
        can_answer, state = reasoner.check_memory_first("Who is tallest?")
        
        # V2 should return False for ambiguous queries (UNKNOWN outcome)
        assert can_answer is False
        assert state is None
    
    def test_v2_pairwise_query_true(self, temp_workspace):
        """Test v2 pairwise query returns TRUE for provable relation."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Add relation
        reasoner.cache_v2.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        # Query pairwise
        can_answer, state = reasoner.check_memory_first("Is Alice taller than Bob?")
        
        assert can_answer is True
        assert state is not None
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].confidence == 0.95
    
    def test_v2_pairwise_query_unknown_entity(self, temp_workspace):
        """Test v2 returns no answer for unknown entities."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Query unknown entities
        can_answer, state = reasoner.check_memory_first("Is Alice taller than Bob?")
        
        # Should return False because entities are unknown
        assert can_answer is False
        assert state is None
    
    def test_v2_unsupported_query_returns_no_answer(self, temp_workspace):
        """Test v2 returns no answer for unsupported query types."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Query something not supported by v2
        can_answer, state = reasoner.check_memory_first("What is the weather?")
        
        assert can_answer is False
        assert state is None
    
    def test_v1_mode_unaffected(self, temp_workspace):
        """Test that v1 mode works normally when v2 flag is False."""
        memory_config = {"use_memory_v2": False}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Should use v1 hypothesis engine
        assert reasoner.hypothesis_engine is not None
        
        # Query with no data - v1 should handle gracefully
        can_answer, state = reasoner.check_memory_first("Who is tallest?")
        
        # v1 returns high entropy when no data
        assert can_answer is False


class TestConsolidationPipelineV2Integration:
    """Test ConsolidationPipeline with v2 flag."""
    
    def test_initialization_v1_mode(self, temp_workspace):
        """Test that v1 components are initialized when v2 flag is False."""
        memory_config = {"use_memory_v2": False}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        assert pipeline.use_memory_v2 is False
        assert pipeline.relational_cache is not None
        assert pipeline.cache_v2 is None
        assert pipeline.extractor_v2 is None
    
    def test_initialization_v2_mode(self, temp_workspace):
        """Test that v2 components are initialized when v2 flag is True."""
        memory_config = {"use_memory_v2": True}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        assert pipeline.use_memory_v2 is True
        assert pipeline.relational_cache is None
        assert pipeline.cache_v2 is not None
        assert pipeline.extractor_v2 is not None
    
    def test_initialization_default_mode(self, temp_workspace):
        """Test that v1 mode is default when flag is not specified."""
        memory_config = {}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        assert pipeline.use_memory_v2 is False
        assert pipeline.relational_cache is not None
    
    def test_v2_confidence_threshold_configuration(self, temp_workspace):
        """Test that v2 confidence threshold can be configured."""
        memory_config = {"use_memory_v2": True, "confidence_threshold": 0.95}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        assert pipeline.extractor_v2.threshold == 0.95
    
    def test_v2_extraction_accepted(self, temp_workspace):
        """Test that v2 extraction accepts high-confidence relations."""
        memory_config = {"use_memory_v2": True}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        # Extract a high-confidence relation
        result = pipeline.extractor_v2.extract("Alice is taller than Bob")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence >= 0.9
        assert result.relation == ("Alice", RelationType.TALLER_THAN, "Bob")
    
    def test_v2_extraction_rejected_low_confidence(self, temp_workspace):
        """Test that v2 extraction rejects low-confidence relations."""
        memory_config = {"use_memory_v2": True, "confidence_threshold": 0.99}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        # Extract a medium-confidence relation (implicit pattern has 0.95 confidence)
        result = pipeline.extractor_v2.extract("Alice taller than Bob")
        
        # Should be rejected because 0.95 < 0.99
        assert result.status == IngestionStatus.REJECTED
        assert result.confidence == 0.95
    
    def test_v2_extraction_rejected_self_relation(self, temp_workspace):
        """Test that v2 extraction rejects self-relations."""
        memory_config = {"use_memory_v2": True}
        pipeline = ConsolidationPipeline(temp_workspace, memory_config=memory_config)
        
        # Try to extract self-relation
        result = pipeline.extractor_v2.extract("Alice is taller than Alice")
        
        assert result.status == IngestionStatus.REJECTED
        assert "self-relation" in result.reason.lower()


class TestDeterministicBehavior:
    """Test that v2 mode provides deterministic behavior."""
    
    def test_v2_repeated_queries_same_result(self, temp_workspace):
        """Test that repeated v2 queries return identical results."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Add data
        reasoner.cache_v2.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        reasoner.cache_v2.add_relation("Bob", RelationType.TALLER_THAN, "Carol")
        
        # Query multiple times
        results = [reasoner.check_memory_first("Who is tallest?") for _ in range(10)]
        
        # All results should be identical
        assert all(r[0] is True for r in results)  # All can answer
        assert all(r[1].hypotheses[0].reasoning == results[0][1].hypotheses[0].reasoning for r in results)
    
    def test_v2_no_numeric_hallucination(self, temp_workspace):
        """Test that v2 does not introduce numeric values not in the cache."""
        memory_config = {"use_memory_v2": True}
        reasoner = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config)
        
        # Add only relational data (no numeric heights)
        reasoner.cache_v2.add_relation("Alice", RelationType.TALLER_THAN, "Bob")
        
        # Query
        can_answer, state = reasoner.check_memory_first("Is Alice taller than Bob?")
        
        # Result should not contain any numeric height values
        assert can_answer is True
        reasoning = state.hypotheses[0].reasoning
        # Check that reasoning doesn't contain standalone numbers (like heights: 170, 180 cm)
        # This checks for sequences of 2+ digits which would indicate numeric hallucination
        import re
        assert not re.search(r'\d{2,}', reasoning), f"Found numeric values in: {reasoning}"


class TestV1V2Coexistence:
    """Test that v1 and v2 can coexist without interference."""
    
    def test_switching_modes(self, temp_workspace):
        """Test that modes can be switched by configuration."""
        # Start with v1
        memory_config_v1 = {"use_memory_v2": False}
        reasoner_v1 = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config_v1)
        assert reasoner_v1.use_memory_v2 is False
        assert reasoner_v1.hypothesis_engine is not None
        
        # Create new instance with v2
        memory_config_v2 = {"use_memory_v2": True}
        reasoner_v2 = MemoryAwareReasoner(workspace=temp_workspace, memory_config=memory_config_v2)
        assert reasoner_v2.use_memory_v2 is True
        assert reasoner_v2.reasoner_v2 is not None
        
        # Both should be independent
        assert reasoner_v1.use_memory_v2 is False
        assert reasoner_v2.use_memory_v2 is True
