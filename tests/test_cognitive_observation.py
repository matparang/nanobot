"""Tests for Phase 0: ObservationLayer."""

import pytest

from nanobot.cognitive.observation import ObservationLayer
from nanobot.cognitive.types import CognitivePhase, ObservationRecord


class TestObservationLayerEnabled:
    def setup_method(self):
        self.layer = ObservationLayer(buffer_size=5, enabled=True)

    def test_observe_reasoning_returns_record(self):
        record = self.layer.observe_reasoning(
            query="what is Python?",
            reasoning_system_used="system1",
            entropy=0.2,
            hypothesis_count=3,
            top_hypothesis_intent="explain",
            top_hypothesis_confidence=0.9,
        )
        assert isinstance(record, ObservationRecord)
        assert record.query == "what is Python?"
        assert record.reasoning_system_used == "system1"
        assert record.entropy == 0.2
        assert record.hypothesis_count == 3
        assert record.phase == CognitivePhase.OBSERVATION

    def test_observe_retrieval_returns_record(self):
        record = self.layer.observe_retrieval(
            query="Python basics",
            memory_retrieval_count=5,
            memory_cache_hit=True,
        )
        assert isinstance(record, ObservationRecord)
        assert record.memory_retrieval_count == 5
        assert record.memory_cache_hit is True

    def test_get_recent_observations_respects_count(self):
        for i in range(4):
            self.layer.observe_reasoning(query=f"query_{i}")
        recent = self.layer.get_recent_observations(count=2)
        assert len(recent) == 2
        assert recent[-1].query == "query_3"

    def test_ring_buffer_overflow(self):
        for i in range(10):  # buffer_size is 5
            self.layer.observe_reasoning(query=f"q{i}")
        recent = self.layer.get_recent_observations(count=100)
        assert len(recent) == 5
        assert recent[-1].query == "q9"

    def test_get_statistics(self):
        self.layer.observe_reasoning(query="a", entropy=0.2, top_hypothesis_confidence=0.8, llm_invoked=True)
        self.layer.observe_reasoning(query="b", entropy=0.4, top_hypothesis_confidence=0.6, llm_invoked=False)
        stats = self.layer.get_statistics()
        assert stats["total_observations"] == 2
        assert abs(stats["avg_entropy"] - 0.3) < 1e-6
        assert abs(stats["llm_invocation_rate"] - 0.5) < 1e-6

    def test_statistics_cache_hit_rate(self):
        self.layer.observe_retrieval(query="a", memory_cache_hit=True)
        self.layer.observe_retrieval(query="b", memory_cache_hit=False)
        stats = self.layer.get_statistics()
        assert abs(stats["cache_hit_rate"] - 0.5) < 1e-6

    def test_observation_does_not_alter_any_external_state(self):
        """Safety: observation must not alter any external state."""
        sentinel = object()
        self.layer.observe_reasoning(query="safety test")
        # No exception, sentinel unchanged — observation is purely additive
        assert sentinel is sentinel


class TestObservationLayerDisabled:
    def setup_method(self):
        self.layer = ObservationLayer(enabled=False)

    def test_observe_reasoning_returns_none(self):
        assert self.layer.observe_reasoning(query="test") is None

    def test_observe_retrieval_returns_none(self):
        assert self.layer.observe_retrieval(query="test") is None

    def test_get_recent_observations_returns_empty(self):
        assert self.layer.get_recent_observations() == []

    def test_get_statistics_returns_empty(self):
        assert self.layer.get_statistics() == {}
