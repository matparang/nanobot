"""Data structures for Fractal Memory and Active Learning State."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContentType(str, Enum):
    """Types of content that can be stored in memory nodes."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    MIXED = "mixed"  # Contains multiple types


class FractalNode(BaseModel):
    """
    A fractal memory node representing a lesson or knowledge piece.
    
    Supports multi-modal content (text, code, images) and hierarchical relationships.
    Stored as lesson_X.json in archives directory.
    """
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list)
    content: str  # The core lesson/fact
    context_summary: str = ""  # Brief summary for the index
    embedding: list[float] | None = None  # Vector embeddings for semantic search
    
    # Multi-modal support
    content_type: ContentType = ContentType.TEXT
    mime_type: str | None = None  # For images: image/png, image/jpeg, etc.
    language: str | None = None  # For code: python, javascript, etc.
    binary_data: str | None = None  # Base64-encoded binary data for images
    
    # Hierarchical relationships
    parent_id: str | None = None  # Parent node ID
    children_ids: list[str] = Field(default_factory=list)  # Child node IDs
    depth: int = 0  # Depth in the hierarchy (0 = root)
    
    # QL-Bot properties
    entangled_ids: dict[str, float] = Field(default_factory=dict)
    spin: float = 0.0


class EntangledFractalNode(FractalNode):
    """
    Alias for QL-Bot specific interfaces.
    
    Use this name for APIs that explicitly communicate "entangled" capability while
    preserving wire-compatibility with existing FractalNode payloads.
    """


class Hypothesis(BaseModel):
    """A candidate strategy in superposition."""
    strategy: str
    confidence: float
    reasoning_trace: str
    tool_candidates: list[str] = Field(default_factory=list)


class SuperpositionalState(BaseModel):
    """Tracks competing hypotheses and uncertainty."""
    active_hypotheses: list[Hypothesis] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)
    evolution_stage: int = 1
    recent_reflections: list[str] = Field(default_factory=list)
    entropy: float = 0.0


class LatentGraph(BaseModel):
    """Topological representation of latent reasoning."""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    nodes: list[str] = Field(default_factory=list)
    edges: list[tuple[str, str, float]] = Field(default_factory=list)
    probabilities: dict[str, float] = Field(default_factory=dict)
    selected_hypothesis_index: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class ActiveLearningState(BaseModel):
    """
    Active Learning State tracking the agent's evolution and focus areas.
    
    Stored as ALS.json in memory directory.
    """
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    current_focus: str = "General Assistance"
    sparring_partners: list[str] = Field(default_factory=list)  # User IDs or personas
    evolution_stage: int = 1
    recent_reflections: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.now)
    superpositional_state: SuperpositionalState | None = None


class ContextBlock(BaseModel):
    """
    A block in the context-engineered workflow.
    
    Used for structured prompt building.
    """
    
    name: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
