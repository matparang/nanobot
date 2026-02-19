"""Type definitions for v2 deterministic relational reasoning.

This module provides enums and dataclasses for deterministic, memory-first
relational reasoning with strict UNKNOWN handling and confidence-gated extraction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TruthValue(Enum):
    """Truth value for deterministic queries."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class RelationType(Enum):
    """Supported relation types for comparative reasoning."""
    TALLER_THAN = "TALLER_THAN"
    SHORTER_THAN = "SHORTER_THAN"


class IngestionStatus(Enum):
    """Status of relation ingestion."""
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass
class QueryResult:
    """Result of a relational query.
    
    Attributes:
        value: Truth value (TRUE/FALSE/UNKNOWN)
        message: Human-readable explanation
        source: Source of the result (e.g., "direct", "transitive", "cache")
        details: Optional additional information (e.g., proof path, entities involved)
    """
    value: TruthValue
    message: str
    source: str
    details: Optional[dict[str, Any]] = None


@dataclass
class IngestionResult:
    """Result of relation extraction and ingestion.
    
    Attributes:
        status: Whether the relation was ACCEPTED or REJECTED
        confidence: Confidence score for the extraction
        relation: Optional tuple (entity_a, relation_type, entity_b) if extracted
        reason: Human-readable reason for the status
        template_id: Identifier of the template that matched (if any)
    """
    status: IngestionStatus
    confidence: float
    relation: Optional[tuple[str, RelationType, str]] = None
    reason: str = ""
    template_id: Optional[str] = None
