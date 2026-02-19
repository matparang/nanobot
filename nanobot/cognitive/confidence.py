"""Phase 1: Confidence Layer — composite confidence scoring."""

from __future__ import annotations

from typing import Any

from nanobot.cognitive.types import ConfidenceScore

_DEFAULT_WEIGHTS: dict[str, float] = {
    "memory_retrieval_strength": 0.30,
    "entanglement_score": 0.20,
    "entropy_level": 0.25,
    "reasoning_agreement": 0.25,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class ConfidenceCalculator:
    """Calculates composite confidence scores without altering routing."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._weights = {**_DEFAULT_WEIGHTS, **(weights or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        memory_retrieval_strength: float = 0.0,
        entanglement_score: float = 0.0,
        entropy_level: float = 0.0,
        reasoning_agreement: float = 0.0,
    ) -> ConfidenceScore:
        """Return a ConfidenceScore. When disabled, returns a zeroed score."""
        if not self.enabled:
            return ConfidenceScore()

        mrs = _clamp(memory_retrieval_strength)
        es = _clamp(entanglement_score)
        # Entropy is inverted: low entropy → high confidence component
        el = _clamp(1.0 - entropy_level)
        ra = _clamp(reasoning_agreement)

        w = self._weights
        overall = (
            w["memory_retrieval_strength"] * mrs
            + w["entanglement_score"] * es
            + w["entropy_level"] * el
            + w["reasoning_agreement"] * ra
        )

        parts = [
            f"memory={mrs:.2f}×{w['memory_retrieval_strength']}",
            f"entanglement={es:.2f}×{w['entanglement_score']}",
            f"entropy_inv={el:.2f}×{w['entropy_level']}",
            f"agreement={ra:.2f}×{w['reasoning_agreement']}",
        ]
        return ConfidenceScore(
            overall=_clamp(overall),
            memory_retrieval_strength=mrs,
            entanglement_score=es,
            entropy_level=el,
            reasoning_agreement=ra,
            source_weights=dict(self._weights),
            explanation=" | ".join(parts),
        )

    def compute_reasoning_agreement(
        self,
        system1_intent: str,
        system1_confidence: float,
        system2_intents: list[str],
        system2_confidences: list[float],
    ) -> float:
        """
        Compute agreement between System 1 and System 2 results.

        Agreement is the average of:
        - keyword overlap between System 1 intent and each System 2 intent
        - confidence proximity (1 - |conf1 - conf2|)
        """
        if not self.enabled:
            return 0.0
        if not system2_intents:
            return 0.0

        s1_words = set(system1_intent.lower().split())
        scores: list[float] = []
        for intent, conf2 in zip(system2_intents, system2_confidences):
            s2_words = set(intent.lower().split())
            union = s1_words | s2_words
            overlap = len(s1_words & s2_words) / len(union) if union else 0.0
            proximity = 1.0 - abs(_clamp(system1_confidence) - _clamp(conf2))
            scores.append((overlap + proximity) / 2.0)

        return sum(scores) / len(scores)
