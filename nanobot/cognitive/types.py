"""Shared dataclasses for the cognitive layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CognitivePhase(str, Enum):
    OBSERVATION = "observation"
    CONFIDENCE = "confidence"
    WORKING_MEMORY = "working_memory"
    CONTROLLER = "controller"


@dataclass
class ObservationRecord:
    timestamp: str  # ISO format
    query: str = ""
    reasoning_system_used: str = ""  # "system1", "system2", "hybrid"
    entropy: float = 0.0
    hypothesis_count: int = 0
    top_hypothesis_intent: str = ""
    top_hypothesis_confidence: float = 0.0
    memory_retrieval_count: int = 0
    memory_cache_hit: bool = False
    latent_reasoning_invoked: bool = False
    llm_invoked: bool = False
    tools_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    phase: CognitivePhase = CognitivePhase.OBSERVATION
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceScore:
    overall: float = 0.0
    memory_retrieval_strength: float = 0.0
    entanglement_score: float = 0.0
    entropy_level: float = 0.0
    reasoning_agreement: float = 0.0
    source_weights: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class WorkingMemoryState:
    query: str = ""
    session_id: str = ""
    retrieved_nodes: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    entropy: float = 0.0
    confidence: Optional[ConfidenceScore] = None
    reasoning_state: str = ""  # "pending", "system1", "system2", "complete"
    observations: list[ObservationRecord] = field(default_factory=list)
    context_summary: str = ""
    tools_invoked: list[str] = field(default_factory=list)
    start_time: str = ""  # ISO format


@dataclass
class CognitiveControllerState:
    query: str = ""
    working_memory: Optional[WorkingMemoryState] = None
    confidence: Optional[ConfidenceScore] = None
    observations: list[ObservationRecord] = field(default_factory=list)
    controller_mode: str = "passive"
