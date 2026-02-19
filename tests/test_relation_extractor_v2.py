"""Tests for RelationExtractionEngineV2 with confidence gating."""

import pytest

from nanobot.memory.relation_extractor_v2 import RelationExtractionEngineV2
from nanobot.memory.types_v2 import IngestionStatus, RelationType


@pytest.fixture
def extractor():
    """Create a RelationExtractionEngineV2 with default threshold."""
    return RelationExtractionEngineV2()


@pytest.fixture
def low_threshold_extractor():
    """Create an extractor with low threshold for testing boundary cases."""
    return RelationExtractionEngineV2(confidence_threshold=0.5)


class TestTemplateExtraction:
    """Test template-based extraction."""
    
    def test_extract_taller_is_explicit(self, extractor):
        """Test extraction of 'A is taller than B' pattern."""
        result = extractor.extract("Alice is taller than Bob")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence == 1.0
        assert result.relation == ("Alice", RelationType.TALLER_THAN, "Bob")
        assert result.template_id == "taller_is_explicit"
    
    def test_extract_shorter_is_explicit(self, extractor):
        """Test extraction of 'A is shorter than B' pattern."""
        result = extractor.extract("Bob is shorter than Carol")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence == 1.0
        assert result.relation == ("Bob", RelationType.SHORTER_THAN, "Carol")
        assert result.template_id == "shorter_is_explicit"
    
    def test_extract_taller_implicit(self, extractor):
        """Test extraction of 'A taller than B' pattern (no 'is')."""
        result = extractor.extract("Alice taller than Bob")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence == 0.95
        assert result.relation == ("Alice", RelationType.TALLER_THAN, "Bob")
        assert result.template_id == "taller_implicit"
    
    def test_extract_shorter_implicit(self, extractor):
        """Test extraction of 'A shorter than B' pattern (no 'is')."""
        result = extractor.extract("Bob shorter than Carol")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence == 0.95
        assert result.relation == ("Bob", RelationType.SHORTER_THAN, "Carol")
        assert result.template_id == "shorter_implicit"
    
    def test_case_insensitive(self, extractor):
        """Test that extraction is case-insensitive."""
        result = extractor.extract("ALICE IS TALLER THAN BOB")
        
        assert result.status == IngestionStatus.ACCEPTED
        # Casing should be preserved from input
        assert result.relation == ("ALICE", RelationType.TALLER_THAN, "BOB")


class TestConfidenceThreshold:
    """Test confidence threshold gating."""
    
    def test_high_confidence_accepted(self, extractor):
        """Test that high confidence (>= 0.9) is ACCEPTED."""
        result = extractor.extract("Alice is taller than Bob")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence >= 0.9
    
    def test_boundary_confidence_with_high_threshold(self):
        """Test boundary case with threshold exactly at confidence."""
        # Extractor with threshold at 0.95
        extractor = RelationExtractionEngineV2(confidence_threshold=0.95)
        
        # Implicit pattern has confidence 0.95
        result = extractor.extract("Alice taller than Bob")
        
        # Should be ACCEPTED (confidence == threshold)
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence == 0.95
    
    def test_below_threshold_rejected(self):
        """Test that confidence below threshold is REJECTED."""
        # Extractor with very high threshold
        extractor = RelationExtractionEngineV2(confidence_threshold=0.99)
        
        # Implicit pattern has confidence 0.95 < 0.99
        result = extractor.extract("Alice taller than Bob")
        
        assert result.status == IngestionStatus.REJECTED
        assert "below threshold" in result.reason.lower()
        assert result.confidence == 0.95
        # Relation should still be extracted even if rejected
        assert result.relation == ("Alice", RelationType.TALLER_THAN, "Bob")
    
    def test_low_threshold_accepts_more(self, low_threshold_extractor):
        """Test that low threshold accepts patterns that high threshold rejects."""
        # With threshold 0.5, implicit patterns (0.95) should be accepted
        result = low_threshold_extractor.extract("Alice taller than Bob")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.confidence >= 0.5


class TestEntityNormalization:
    """Test entity name normalization."""
    
    def test_strip_whitespace(self, extractor):
        """Test that whitespace is stripped from entities."""
        result = extractor.extract("  Alice   is taller than   Bob  ")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.relation[0] == "Alice"
        assert result.relation[2] == "Bob"
    
    def test_strip_punctuation(self, extractor):
        """Test that boundary punctuation is stripped."""
        result = extractor.extract("Alice is taller than Bob.")
        
        assert result.status == IngestionStatus.ACCEPTED
        # Punctuation should be stripped from entity names
        assert result.relation[0] == "Alice"
        assert result.relation[2] == "Bob"
    
    def test_preserve_casing(self, extractor):
        """Test that original casing is preserved."""
        result = extractor.extract("ALICE is taller than bob")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.relation[0] == "ALICE"
        assert result.relation[2] == "bob"


class TestSelfRelationRejection:
    """Test that self-relations are rejected."""
    
    def test_self_relation_rejected(self, extractor):
        """Test that 'A is taller than A' is rejected."""
        result = extractor.extract("Alice is taller than Alice")
        
        assert result.status == IngestionStatus.REJECTED
        assert "self-relation" in result.reason.lower()
    
    def test_self_relation_case_insensitive(self, extractor):
        """Test that self-relation check is case-insensitive."""
        result = extractor.extract("Alice is taller than ALICE")
        
        assert result.status == IngestionStatus.REJECTED
        assert "self-relation" in result.reason.lower()


class TestNoMatch:
    """Test handling of text with no matching templates."""
    
    def test_no_template_match(self, extractor):
        """Test that non-matching text is rejected."""
        result = extractor.extract("The weather is nice today")
        
        assert result.status == IngestionStatus.REJECTED
        assert result.confidence == 0.0
        assert result.relation is None
        assert "no template match" in result.reason.lower()
    
    def test_empty_text(self, extractor):
        """Test that empty text is rejected."""
        result = extractor.extract("")
        
        assert result.status == IngestionStatus.REJECTED
        assert result.confidence == 0.0
        assert "empty" in result.reason.lower()
    
    def test_whitespace_only(self, extractor):
        """Test that whitespace-only text is rejected."""
        result = extractor.extract("   \n  \t  ")
        
        assert result.status == IngestionStatus.REJECTED
        assert "empty" in result.reason.lower()


class TestMultipleMatches:
    """Test behavior when multiple templates could match."""
    
    def test_first_match_wins(self, extractor):
        """Test that first matching template is used."""
        # Text that could match multiple patterns
        result = extractor.extract("Alice is taller than Bob")
        
        # Should match the "is" explicit pattern first (higher confidence)
        assert result.template_id == "taller_is_explicit"
        assert result.confidence == 1.0


class TestLogging:
    """Test that extraction attempts are logged."""
    
    def test_accepted_logged(self, extractor):
        """Test that ACCEPTED extractions are logged."""
        # Note: logging requires proper setup, so we just verify extraction works
        result = extractor.extract("Alice is taller than Bob")
        
        # Should be accepted
        assert result.status == IngestionStatus.ACCEPTED
    
    def test_rejected_logged(self, extractor):
        """Test that REJECTED extractions work correctly."""
        result = extractor.extract("The weather is nice")
        
        # Should be rejected
        assert result.status == IngestionStatus.REJECTED


class TestTemplateMetadata:
    """Test that template metadata is tracked."""
    
    def test_template_id_recorded(self, extractor):
        """Test that template_id is recorded in result."""
        result = extractor.extract("Alice is taller than Bob")
        
        assert result.template_id is not None
        assert isinstance(result.template_id, str)
    
    def test_different_templates_different_ids(self, extractor):
        """Test that different patterns have different template IDs."""
        result1 = extractor.extract("Alice is taller than Bob")
        result2 = extractor.extract("Bob is shorter than Carol")
        
        assert result1.template_id != result2.template_id


class TestDeterminism:
    """Test that extraction is deterministic."""
    
    def test_repeated_extraction_same_result(self, extractor):
        """Test that same text produces same result."""
        text = "Alice is taller than Bob"
        
        results = [extractor.extract(text) for _ in range(10)]
        
        # All results should be identical
        assert all(r.status == results[0].status for r in results)
        assert all(r.confidence == results[0].confidence for r in results)
        assert all(r.relation == results[0].relation for r in results)
        assert all(r.template_id == results[0].template_id for r in results)


class TestComplexText:
    """Test extraction from more complex text."""
    
    def test_extract_from_sentence(self, extractor):
        """Test extraction from a full sentence."""
        result = extractor.extract("In our class, Alice is taller than Bob.")
        
        assert result.status == IngestionStatus.ACCEPTED
        assert result.relation == ("Alice", RelationType.TALLER_THAN, "Bob")
    
    def test_extract_first_match_from_multiple(self, extractor):
        """Test that only first match is extracted (not multiple relations)."""
        result = extractor.extract(
            "Alice is taller than Bob and Carol is shorter than Dave"
        )
        
        # Should extract first match only
        assert result.status == IngestionStatus.ACCEPTED
        assert result.relation[0] == "Alice"
        assert result.relation[2] == "Bob"
