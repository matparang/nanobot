from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.dual_reasoning import DualReasoningOrchestrator
from nanobot.agent.memory_types import Hypothesis, SuperpositionalState
from nanobot.providers.base import LLMResponse
from nanobot.runtime.chi_tracker import ChiTracker
from nanobot.runtime.state import state


@pytest.mark.asyncio
async def test_high_confidence_uses_system1_only():
    previous = state.get_all_toggles()
    try:
        state.dual_layer_enabled = True
        state.light_reasoner_enabled = True
        state.latent_reasoning_enabled = True
        state.chi_tracking_enabled = True

        provider = SimpleNamespace(
            chat=AsyncMock(
                return_value=LLMResponse(
                    content='{"intent":"search","confidence":0.9,"reasoning":"clear intent"}'
                )
            )
        )
        orchestrator = DualReasoningOrchestrator(
            provider=provider,
            model="test-model",
            chi_tracker=ChiTracker(),
            memory_config={},
        )
        result = await orchestrator.reason("Search Python tutorials", "docs context")
        assert result.system_used == "system1"
        assert result.escalated is False
        assert result.total_chi_cost < 0.5
    finally:
        state.restore_toggles(previous)


@pytest.mark.asyncio
async def test_low_confidence_escalates_to_system2():
    previous = state.get_all_toggles()
    try:
        state.dual_layer_enabled = True
        state.light_reasoner_enabled = True
        state.latent_reasoning_enabled = True

        provider = SimpleNamespace(
            chat=AsyncMock(
                return_value=LLMResponse(content='{"intent":"unknown","confidence":0.2,"reasoning":"ambiguous"}')
            )
        )
        orchestrator = DualReasoningOrchestrator(
            provider=provider,
            model="test-model",
            chi_tracker=ChiTracker(),
            memory_config={},
        )
        orchestrator.latent_reasoner.reason = AsyncMock(
            return_value=SuperpositionalState(
                hypotheses=[Hypothesis(intent="do thing", confidence=0.6, reasoning="needs deeper pass")],
                entropy=0.3,
                strategic_direction="use system2",
            )
        )
        result = await orchestrator.reason("Do the thing", "minimal context")
        assert result.escalated is True
        assert result.system_used in ["system2", "hybrid"]
    finally:
        state.restore_toggles(previous)


@pytest.mark.asyncio
async def test_pattern_cache_reduces_latency():
    previous = state.get_all_toggles()
    try:
        state.dual_layer_enabled = True
        state.light_reasoner_enabled = True
        state.latent_reasoning_enabled = True

        provider = SimpleNamespace(
            chat=AsyncMock(
                return_value=LLMResponse(
                    content='{"intent":"search","confidence":0.95,"reasoning":"repeatable intent"}'
                )
            )
        )
        orchestrator = DualReasoningOrchestrator(
            provider=provider,
            model="test-model",
            chi_tracker=ChiTracker(),
            memory_config={},
        )
        await orchestrator.reason("Search Python tutorials", "docs context")
        result = await orchestrator.reason("Search Python tutorials", "docs context")
        assert result.system1_result is not None
        assert result.system1_result.pattern_hit is True
        assert result.total_chi_cost < 0.05
    finally:
        state.restore_toggles(previous)
