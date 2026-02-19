"""Phase 0: Observation Layer — non-invasive cognitive visibility."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from nanobot.cognitive.types import ObservationRecord, CognitivePhase

_log = logging.getLogger("nanobot.cognitive.observation")


class ObservationLayer:
    """Records reasoning and memory observations without altering any behavior."""

    def __init__(self, buffer_size: int = 100, enabled: bool = True) -> None:
        self.enabled = enabled
        self._buffer: deque[ObservationRecord] = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe_reasoning(
        self,
        query: str,
        reasoning_system_used: str = "",
        entropy: float = 0.0,
        hypothesis_count: int = 0,
        top_hypothesis_intent: str = "",
        top_hypothesis_confidence: float = 0.0,
        latent_reasoning_invoked: bool = False,
        llm_invoked: bool = False,
        tools_used: list[str] | None = None,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ObservationRecord | None:
        """Record a reasoning observation (read-only, no side effects)."""
        if not self.enabled:
            return None
        record = ObservationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            reasoning_system_used=reasoning_system_used,
            entropy=entropy,
            hypothesis_count=hypothesis_count,
            top_hypothesis_intent=top_hypothesis_intent,
            top_hypothesis_confidence=top_hypothesis_confidence,
            latent_reasoning_invoked=latent_reasoning_invoked,
            llm_invoked=llm_invoked,
            tools_used=tools_used or [],
            latency_ms=latency_ms,
            phase=CognitivePhase.OBSERVATION,
            metadata=metadata or {},
        )
        self._buffer.append(record)
        _log.debug(
            "observe_reasoning query=%r system=%s entropy=%.3f",
            query[:80],
            reasoning_system_used,
            entropy,
        )
        return record

    def observe_retrieval(
        self,
        query: str,
        memory_retrieval_count: int = 0,
        memory_cache_hit: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ObservationRecord | None:
        """Record a memory retrieval observation (read-only, no side effects)."""
        if not self.enabled:
            return None
        record = ObservationRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
            memory_retrieval_count=memory_retrieval_count,
            memory_cache_hit=memory_cache_hit,
            phase=CognitivePhase.OBSERVATION,
            metadata=metadata or {},
        )
        self._buffer.append(record)
        _log.debug(
            "observe_retrieval query=%r count=%d cache_hit=%s",
            query[:80],
            memory_retrieval_count,
            memory_cache_hit,
        )
        return record

    def get_recent_observations(self, count: int = 10) -> list[ObservationRecord]:
        """Return the most recent observations (up to *count*)."""
        if not self.enabled:
            return []
        items = list(self._buffer)
        return items[-count:] if count < len(items) else items

    def get_statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics over all buffered observations."""
        if not self.enabled or not self._buffer:
            return {}
        records = list(self._buffer)
        total = len(records)
        avg_entropy = sum(r.entropy for r in records) / total
        avg_confidence = sum(r.top_hypothesis_confidence for r in records) / total
        cache_hits = sum(1 for r in records if r.memory_cache_hit)
        llm_invocations = sum(1 for r in records if r.llm_invoked)
        return {
            "total_observations": total,
            "avg_entropy": avg_entropy,
            "avg_top_confidence": avg_confidence,
            "cache_hit_rate": cache_hits / total,
            "llm_invocation_rate": llm_invocations / total,
        }
