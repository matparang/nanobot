"""Phase 3: Passive Cognitive Controller — wraps DualReasoningOrchestrator non-invasively."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from nanobot.cognitive.confidence import ConfidenceCalculator
from nanobot.cognitive.observation import ObservationLayer
from nanobot.cognitive.types import CognitiveControllerState, ConfidenceScore
from nanobot.cognitive.working_memory import WorkingMemory

_log = logging.getLogger("nanobot.cognitive.controller")


class CognitiveController:
    """
    Passive wrapper around the reasoning pipeline.

    When *enabled* is False the controller adds zero overhead:
    ``await reason_fn(query, context_summary)`` is returned immediately.

    When enabled, it:
    1. Creates a working-memory cycle
    2. Calls the existing pipeline unchanged
    3. Extracts read-only metadata from the result
    4. Records an observation and computes confidence
    5. Returns the EXACT same result object — routing is never altered
    """

    # Mode is intentionally hardcoded — no routing decisions are ever made.
    mode: str = "passive"

    def __init__(
        self,
        observation_config: dict[str, Any] | None = None,
        confidence_config: dict[str, Any] | None = None,
        working_memory_config: dict[str, Any] | None = None,
        enabled: bool = False,
        mode: str | None = None,
    ) -> None:
        self.enabled = enabled
        resolved_mode = self._resolve_mode(mode, working_memory_config)

        if resolved_mode == "active":
            raise RuntimeError(
                "Active cognitive control is not yet supported. "
                "Use --cognitive-controller=passive or remove the active mode configuration."
            )
        self.mode = resolved_mode
        obs_cfg = observation_config or {}
        conf_cfg = confidence_config or {}
        wm_cfg = working_memory_config or {}

        self._observation = ObservationLayer(
            buffer_size=int(obs_cfg.get("buffer_size", 100)),
            enabled=enabled,
        )
        self._confidence = ConfidenceCalculator(
            weights=conf_cfg.get("weights"),
            enabled=enabled,
        )
        self._working_memory = WorkingMemory(enabled=enabled)

        # Snapshot of the most recent controller state (for introspection)
        self._last_state: CognitiveControllerState = CognitiveControllerState()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    async def process(
        self,
        query: str,
        context_summary: str,
        session_id: str,
        reason_fn: Callable[..., Awaitable[Any]],
        **kwargs: Any,
    ) -> Any:
        """
        Wrap *reason_fn* with observation and confidence metadata.

        SAFETY: Returns the EXACT same object that *reason_fn* returns.
        Routing is NEVER altered.
        """
        if not self.enabled:
            return await reason_fn(query, context_summary, **kwargs)

        self._working_memory.begin_cycle(query, session_id)
        self._working_memory.set_context_summary(context_summary)

        t0 = time.perf_counter()
        result = await reason_fn(query, context_summary, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # --- Read-only metadata extraction ---
        data = self._extract_reasoning_data(result)

        self._working_memory.set_entropy(data["entropy"])
        self._working_memory.set_hypotheses(data["hypotheses"])
        self._working_memory.set_reasoning_state(data["system_used"])

        observation = self._observation.observe_reasoning(
            query=query,
            reasoning_system_used=data["system_used"],
            entropy=data["entropy"],
            hypothesis_count=data["hypothesis_count"],
            top_hypothesis_intent=data["top_hypothesis_intent"],
            top_hypothesis_confidence=data["top_hypothesis_confidence"],
            latent_reasoning_invoked=data["latent_reasoning_invoked"],
            latency_ms=latency_ms,
        )
        if observation is not None:
            self._working_memory.add_observation(observation)

        confidence = self._confidence.calculate(
            entropy_level=data["entropy"],
        )
        self._working_memory.set_confidence(confidence)

        final_wm = self._working_memory.end_cycle()
        self._last_state = CognitiveControllerState(
            query=query,
            working_memory=final_wm,
            confidence=confidence,
            observations=list(self._observation.get_recent_observations(1)),
            controller_mode=self.mode,
        )

        _log.debug(
            "cognitive_controller query=%r system=%s entropy=%.3f confidence=%.3f latency_ms=%.1f",
            query[:80],
            data["system_used"],
            data["entropy"],
            confidence.overall,
            latency_ms,
        )

        # CRITICAL: return the unmodified result
        return result

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_state(self) -> CognitiveControllerState:
        """Return a snapshot of the last controller state."""
        return self._last_state

    def get_observation_statistics(self) -> dict[str, Any]:
        """Delegate to the observation layer."""
        return self._observation.get_statistics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_reasoning_data(self, result: Any) -> dict[str, Any]:
        """
        Extract read-only metadata from a DualReasoningResult or SuperpositionalState.

        Never raises — falls back to safe defaults.
        """
        data: dict[str, Any] = {
            "entropy": 0.0,
            "hypothesis_count": 0,
            "hypotheses": [],
            "top_hypothesis_intent": "",
            "top_hypothesis_confidence": 0.0,
            "system_used": "unknown",
            "latent_reasoning_invoked": False,
        }
        try:
            # DualReasoningResult
            if hasattr(result, "final_state") and hasattr(result, "system_used"):
                data["system_used"] = result.system_used or "unknown"
                data["latent_reasoning_invoked"] = result.escalated
                fs = result.final_state
                if fs is not None:
                    data["entropy"] = float(getattr(fs, "entropy", 0.0))
                    hyps = list(getattr(fs, "hypotheses", []) or [])
                    data["hypothesis_count"] = len(hyps)
                    data["hypotheses"] = [
                        {"intent": getattr(h, "intent", ""), "confidence": getattr(h, "confidence", 0.0)}
                        for h in hyps
                    ]
                    if hyps:
                        top = hyps[0]
                        data["top_hypothesis_intent"] = getattr(top, "intent", "")
                        data["top_hypothesis_confidence"] = float(getattr(top, "confidence", 0.0))
            # SuperpositionalState
            elif hasattr(result, "hypotheses") and hasattr(result, "entropy"):
                data["entropy"] = float(getattr(result, "entropy", 0.0))
                hyps = list(result.hypotheses or [])
                data["hypothesis_count"] = len(hyps)
                data["hypotheses"] = [
                    {"intent": getattr(h, "intent", ""), "confidence": getattr(h, "confidence", 0.0)}
                    for h in hyps
                ]
                if hyps:
                    top = hyps[0]
                    data["top_hypothesis_intent"] = getattr(top, "intent", "")
                    data["top_hypothesis_confidence"] = float(getattr(top, "confidence", 0.0))
        except Exception as exc:  # noqa: BLE001 — KeyboardInterrupt/SystemExit extend BaseException, not Exception
            _log.warning("_extract_reasoning_data failed (non-fatal): %s", exc)
        return data

    # ------------------------------------------------------------------
    # Internal configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_mode(mode: str | None, working_memory_config: dict[str, Any] | None) -> str:
        """
        Resolve controller mode with precedence:
        explicit mode parameter > working_memory_config > passive default.
        """
        wm_mode = (working_memory_config or {}).get("cognitive_controller_mode")
        return mode or wm_mode or "passive"
