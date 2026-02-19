"""Confidence-gated template-based relation extraction for v2.

This module provides deterministic relation extraction using regex templates
with confidence scoring and threshold-based ingestion gating.
"""

import logging
import re
from typing import Optional

from nanobot.memory.config_v2 import CONF_THRESHOLD, LOGGER_NAME
from nanobot.memory.types_v2 import IngestionResult, IngestionStatus, RelationType

logger = logging.getLogger(LOGGER_NAME)


class RelationExtractionEngineV2:
    """Template-based relation extraction with confidence gating.
    
    Extracts height relations from text using regex templates and computes
    confidence scores. Only returns ACCEPTED results when confidence meets
    the threshold.
    
    Attributes:
        templates: List of (pattern, relation_type, template_id, confidence) tuples
        threshold: Minimum confidence for acceptance
    """
    
    def __init__(self, confidence_threshold: float = CONF_THRESHOLD):
        """Initialize extraction engine with templates.
        
        Args:
            confidence_threshold: Minimum confidence for ACCEPTED status (default: 0.9)
        """
        self.threshold = confidence_threshold
        
        # Define templates with base confidence scores
        # Template format: (regex_pattern, relation_type, template_id, base_confidence)
        # Higher confidence for more specific patterns (with "is")
        self.templates = [
            # High confidence: explicit "is" patterns
            (
                re.compile(r"(\w+)\s+is\s+taller\s+than\s+(\w+)", re.IGNORECASE),
                RelationType.TALLER_THAN,
                "taller_is_explicit",
                1.0
            ),
            (
                re.compile(r"(\w+)\s+is\s+shorter\s+than\s+(\w+)", re.IGNORECASE),
                RelationType.SHORTER_THAN,
                "shorter_is_explicit",
                1.0
            ),
            # Medium-high confidence: implicit patterns
            (
                re.compile(r"(\w+)\s+taller\s+than\s+(\w+)", re.IGNORECASE),
                RelationType.TALLER_THAN,
                "taller_implicit",
                0.95
            ),
            (
                re.compile(r"(\w+)\s+shorter\s+than\s+(\w+)", re.IGNORECASE),
                RelationType.SHORTER_THAN,
                "shorter_implicit",
                0.95
            ),
        ]
        
        logger.info(
            f"RelationExtractionEngineV2 initialized with {len(self.templates)} templates, "
            f"threshold={self.threshold}"
        )
    
    def extract(self, text: str) -> IngestionResult:
        """Extract relation from text with confidence gating.
        
        Args:
            text: Natural language text to extract from
            
        Returns:
            IngestionResult with status, confidence, and extracted relation if accepted
        """
        if not text or not text.strip():
            logger.debug("Empty text, returning REJECTED")
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                confidence=0.0,
                reason="Empty input"
            )
        
        # Try each template
        for pattern, relation_type, template_id, base_confidence in self.templates:
            match = pattern.search(text)
            
            if match:
                entity_a = self._normalize_entity(match.group(1))
                entity_b = self._normalize_entity(match.group(2))
                
                # Skip self-relations
                if entity_a.lower() == entity_b.lower():
                    logger.debug(
                        f"Self-relation {entity_a} {relation_type.value} {entity_b}, "
                        "returning REJECTED"
                    )
                    return IngestionResult(
                        status=IngestionStatus.REJECTED,
                        confidence=base_confidence,
                        reason="Self-relation not allowed",
                        template_id=template_id
                    )
                
                # Compute final confidence (could adjust based on text quality)
                confidence = base_confidence
                
                # Check threshold
                if confidence >= self.threshold:
                    logger.info(
                        f"ACCEPTED: {entity_a} {relation_type.value} {entity_b} "
                        f"(template={template_id}, confidence={confidence:.2f})"
                    )
                    return IngestionResult(
                        status=IngestionStatus.ACCEPTED,
                        confidence=confidence,
                        relation=(entity_a, relation_type, entity_b),
                        reason="Confidence meets threshold",
                        template_id=template_id
                    )
                else:
                    logger.info(
                        f"REJECTED: {entity_a} {relation_type.value} {entity_b} "
                        f"(template={template_id}, confidence={confidence:.2f} < {self.threshold})"
                    )
                    return IngestionResult(
                        status=IngestionStatus.REJECTED,
                        confidence=confidence,
                        relation=(entity_a, relation_type, entity_b),
                        reason=f"Confidence {confidence:.2f} below threshold {self.threshold}",
                        template_id=template_id
                    )
        
        # No template matched
        logger.debug(f"No template matched for text: {text[:50]}...")
        return IngestionResult(
            status=IngestionStatus.REJECTED,
            confidence=0.0,
            reason="No template match"
        )
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity name by stripping whitespace and punctuation.
        
        Preserves casing for consistency with original text.
        
        Args:
            entity: Raw entity string
            
        Returns:
            Normalized entity string
        """
        # Strip whitespace
        entity = entity.strip()
        
        # Remove common punctuation at boundaries
        entity = entity.strip('.,;:!?"\'')
        
        return entity
