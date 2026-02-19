"""Phase 2: Working Memory Layer — ephemeral cognitive state storage."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from nanobot.cognitive.types import ConfidenceScore, ObservationRecord, WorkingMemoryState

_log = logging.getLogger("nanobot.cognitive.working_memory")


class WorkingMemory:
    """Manages a single ephemeral reasoning-cycle state object."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._state: WorkingMemoryState | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_cycle(self, query: str, session_id: str = "") -> WorkingMemoryState | None:
        """Start a new reasoning cycle. Any unfinished previous cycle is discarded."""
        if not self.enabled:
            return None
        if self._state is not None:
            _log.warning(
                "begin_cycle called before end_cycle for query=%r — discarding previous cycle",
                self._state.query[:80],
            )
        self._state = WorkingMemoryState(
            query=query,
            session_id=session_id,
            reasoning_state="pending",
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        return self._state

    def end_cycle(self) -> WorkingMemoryState | None:
        """Mark the current cycle complete, clear internal reference, return final state."""
        if not self.enabled or self._state is None:
            return None
        self._state.reasoning_state = "complete"
        final = self._state
        self._state = None
        return final

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_retrieved_nodes(self, nodes: list[dict[str, Any]]) -> None:
        if self.enabled and self._state is not None:
            self._state.retrieved_nodes = list(nodes)

    def set_hypotheses(self, hypotheses: list[dict[str, Any]]) -> None:
        if self.enabled and self._state is not None:
            self._state.hypotheses = list(hypotheses)

    def set_entropy(self, entropy: float) -> None:
        if self.enabled and self._state is not None:
            self._state.entropy = entropy

    def set_reasoning_state(self, state: str) -> None:
        if self.enabled and self._state is not None:
            self._state.reasoning_state = state

    def set_confidence(self, confidence: ConfidenceScore) -> None:
        if self.enabled and self._state is not None:
            self._state.confidence = confidence

    def set_context_summary(self, summary: str) -> None:
        if self.enabled and self._state is not None:
            self._state.context_summary = summary

    def add_observation(self, observation: ObservationRecord) -> None:
        if self.enabled and self._state is not None:
            self._state.observations.append(observation)

    def add_tool(self, tool_name: str) -> None:
        if self.enabled and self._state is not None:
            self._state.tools_invoked.append(tool_name)

    # ------------------------------------------------------------------
    # Accessor
    # ------------------------------------------------------------------

    def get_state(self) -> WorkingMemoryState | None:
        """Return a reference to the current state (not a copy)."""
        if not self.enabled:
            return None
        return self._state
