"""Tests for Phase 3: CognitiveController."""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.cognitive.cognitive_controller import CognitiveController
from nanobot.cognitive.types import CognitiveControllerState


# ---------------------------------------------------------------------------
# Minimal stand-ins for DualReasoningResult / SuperpositionalState
# ---------------------------------------------------------------------------

@dataclass
class _FakeHypothesis:
    intent: str = "explain"
    confidence: float = 0.8


@dataclass
class _FakeSuperState:
    hypotheses: list = field(default_factory=lambda: [_FakeHypothesis()])
    entropy: float = 0.2
    strategic_direction: str = ""


@dataclass
class _FakeDualResult:
    final_state: _FakeSuperState = field(default_factory=_FakeSuperState)
    system_used: str = "system1"
    system1_result: Any = None
    system2_state: Any = None
    total_chi_cost: float = 0.1
    escalated: bool = False
    reasoning_trace: list = field(default_factory=list)
    latency_ms: float = 5.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCognitiveControllerPassThrough:
    @pytest.mark.asyncio
    async def test_pass_through_returns_unmodified_result(self):
        """SAFETY: must return the EXACT same object as reason_fn."""
        sentinel = _FakeDualResult()
        mock_reason = AsyncMock(return_value=sentinel)

        controller = CognitiveController(enabled=True)
        result = await controller.process(
            query="test query",
            context_summary="some context",
            session_id="s1",
            reason_fn=mock_reason,
        )
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_reason_fn_called_exactly_once(self):
        mock_reason = AsyncMock(return_value=_FakeDualResult())
        controller = CognitiveController(enabled=True)
        await controller.process(
            query="q", context_summary="ctx", session_id="s", reason_fn=mock_reason
        )
        mock_reason.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mode_is_passive(self):
        controller = CognitiveController(enabled=True)
        assert controller.mode == "passive"

    @pytest.mark.asyncio
    async def test_get_state_after_process(self):
        controller = CognitiveController(enabled=True)
        await controller.process(
            query="hello",
            context_summary="ctx",
            session_id="s",
            reason_fn=AsyncMock(return_value=_FakeDualResult()),
        )
        state = controller.get_state()
        assert isinstance(state, CognitiveControllerState)
        assert state.query == "hello"
        assert state.controller_mode == "passive"

    @pytest.mark.asyncio
    async def test_statistics_available_after_process(self):
        controller = CognitiveController(enabled=True)
        await controller.process(
            query="q",
            context_summary="ctx",
            session_id="s",
            reason_fn=AsyncMock(return_value=_FakeDualResult()),
        )
        stats = controller.get_observation_statistics()
        assert "total_observations" in stats
        assert stats["total_observations"] >= 1

    @pytest.mark.asyncio
    async def test_extract_dual_reasoning_result(self):
        controller = CognitiveController(enabled=True)
        fake = _FakeDualResult(system_used="system2", escalated=True)
        fake.final_state.entropy = 0.6
        result = await controller.process(
            query="q", context_summary="ctx", session_id="s",
            reason_fn=AsyncMock(return_value=fake),
        )
        assert result is fake

    @pytest.mark.asyncio
    async def test_extract_superpositional_state(self):
        """Controller handles plain SuperpositionalState too."""
        controller = CognitiveController(enabled=True)
        state = _FakeSuperState(entropy=0.4)
        result = await controller.process(
            query="q", context_summary="ctx", session_id="s",
            reason_fn=AsyncMock(return_value=state),
        )
        assert result is state


class TestCognitiveControllerDisabled:
    @pytest.mark.asyncio
    async def test_disabled_returns_reason_fn_result(self):
        sentinel = _FakeDualResult()
        mock_reason = AsyncMock(return_value=sentinel)
        controller = CognitiveController(enabled=False)
        result = await controller.process(
            query="q", context_summary="ctx", session_id="s", reason_fn=mock_reason
        )
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_disabled_calls_reason_fn_once(self):
        mock_reason = AsyncMock(return_value=_FakeDualResult())
        controller = CognitiveController(enabled=False)
        await controller.process(
            query="q", context_summary="ctx", session_id="s", reason_fn=mock_reason
        )
        mock_reason.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_statistics_empty(self):
        controller = CognitiveController(enabled=False)
        await controller.process(
            query="q", context_summary="ctx", session_id="s",
            reason_fn=AsyncMock(return_value=_FakeDualResult()),
        )
        assert controller.get_observation_statistics() == {}
