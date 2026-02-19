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
    # Comparative
    FASTER_THAN = "FASTER_THAN"
    SLOWER_THAN = "SLOWER_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    # Structural / Dependency
    DEPENDS_ON = "DEPENDS_ON"
    DEPENDENCY_OF = "DEPENDENCY_OF"
    IMPACTS = "IMPACTS"
    IMPACTED_BY = "IMPACTED_BY"
    CONTAINS = "CONTAINS"
    CONTAINED_IN = "CONTAINED_IN"
    SUPPLIES = "SUPPLIES"
    SUPPLIED_BY = "SUPPLIED_BY"
    # Generic
    RELATED_TO = "RELATED_TO"


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
class AggregateResult:
    """Result of an aggregate reasoning query.

    Attributes:
        value: Truth value (TRUE/FALSE/UNKNOWN)
        count: Number of entities satisfying the query
        entities: List of entity names in the result
        message: Human-readable explanation
        source: Source of the result
        details: Optional additional information
    """
    value: TruthValue
    count: int
    entities: list[str]
    message: str
    source: str
    details: Optional[dict] = None


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
