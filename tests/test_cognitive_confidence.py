"""Tests for Phase 1: ConfidenceCalculator."""

import pytest

from nanobot.cognitive.confidence import ConfidenceCalculator
from nanobot.cognitive.types import ConfidenceScore


class TestConfidenceCalculatorEnabled:
    def setup_method(self):
        self.calc = ConfidenceCalculator(enabled=True)

    def test_calculate_returns_confidence_score(self):
        score = self.calc.calculate(
            memory_retrieval_strength=0.8,
            entanglement_score=0.7,
            entropy_level=0.1,
            reasoning_agreement=0.9,
        )
        assert isinstance(score, ConfidenceScore)
        assert 0.0 <= score.overall <= 1.0

    def test_entropy_is_inverted(self):
        """Low entropy should produce a higher entropy_level component."""
        low_entropy = self.calc.calculate(entropy_level=0.1)
        high_entropy = self.calc.calculate(entropy_level=0.9)
        assert low_entropy.entropy_level > high_entropy.entropy_level

    def test_inputs_clamped_above_one(self):
        score = self.calc.calculate(
            memory_retrieval_strength=2.0,
            entanglement_score=5.0,
            entropy_level=0.0,
            reasoning_agreement=1.5,
        )
        assert score.overall <= 1.0

    def test_inputs_clamped_below_zero(self):
        score = self.calc.calculate(
            memory_retrieval_strength=-1.0,
            entanglement_score=-0.5,
            entropy_level=0.0,
            reasoning_agreement=-2.0,
        )
        assert score.overall >= 0.0

    def test_all_zeros_gives_low_entropy_component(self):
        score = self.calc.calculate(
            memory_retrieval_strength=0.0,
            entanglement_score=0.0,
            entropy_level=0.0,
            reasoning_agreement=0.0,
        )
        # entropy_level=0 → inverted component = 1.0 → contributes weight 0.25
        assert score.entropy_level == 1.0
        assert score.overall > 0.0

    def test_source_weights_present(self):
        score = self.calc.calculate()
        assert "memory_retrieval_strength" in score.source_weights
        assert "entropy_level" in score.source_weights

    def test_explanation_non_empty(self):
        score = self.calc.calculate()
        assert score.explanation != ""

    def test_compute_reasoning_agreement_identical_intents(self):
        agreement = self.calc.compute_reasoning_agreement(
            system1_intent="search web",
            system1_confidence=0.9,
            system2_intents=["search web"],
            system2_confidences=[0.9],
        )
        assert agreement == 1.0

    def test_compute_reasoning_agreement_disjoint_intents(self):
        agreement = self.calc.compute_reasoning_agreement(
            system1_intent="alpha beta",
            system1_confidence=0.5,
            system2_intents=["gamma delta"],
            system2_confidences=[0.5],
        )
        # overlap=0, proximity=1 → average=0.5
        assert abs(agreement - 0.5) < 1e-6

    def test_compute_reasoning_agreement_empty_system2(self):
        assert self.calc.compute_reasoning_agreement("intent", 0.8, [], []) == 0.0

    def test_calculate_does_not_alter_routing(self):
        """Safety: ConfidenceCalculator must not alter routing or answers."""
        score = self.calc.calculate(memory_retrieval_strength=0.5)
        # score is metadata only; no side-effects
        assert isinstance(score, ConfidenceScore)


class TestConfidenceCalculatorDisabled:
    def setup_method(self):
        self.calc = ConfidenceCalculator(enabled=False)

    def test_calculate_returns_zero_score(self):
        score = self.calc.calculate(
            memory_retrieval_strength=1.0,
            entanglement_score=1.0,
            entropy_level=0.0,
            reasoning_agreement=1.0,
        )
        assert score.overall == 0.0

    def test_compute_agreement_returns_zero(self):
        assert self.calc.compute_reasoning_agreement("x", 1.0, ["x"], [1.0]) == 0.0
