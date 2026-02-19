"""Integration test for LLM enforcement in AgentLoop."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock
import tempfile

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.runtime.state import state


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=LLMProvider)
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=LLMResponse(
        content="Test response from LLM",
        finish_reason="stop"
    ))
    return provider


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset runtime state before and after each test."""
    original_llm = state.llm_enabled
    state.llm_enabled = True
    yield
    state.llm_enabled = original_llm


@pytest.mark.asyncio
async def test_agent_loop_with_llm_enabled(mock_provider, temp_workspace):
    """Test that AgentLoop works normally with LLM enabled."""
    state.llm_enabled = True
    
    bus = MessageBus()
    agent = AgentLoop(
        bus=bus,
        provider=mock_provider,
        workspace=temp_workspace,
        model="test-model",
    )
    
    response = await agent.process_direct("Hello, agent!")
    
    # Should get response from LLM
    assert response is not None
    assert len(response) > 0
    mock_provider.chat.assert_called()


@pytest.mark.asyncio
async def test_agent_loop_with_llm_disabled(mock_provider, temp_workspace):
    """Test that AgentLoop handles LLM disabled gracefully."""
    state.llm_enabled = False
    
    bus = MessageBus()
    agent = AgentLoop(
        bus=bus,
        provider=mock_provider,
        workspace=temp_workspace,
        model="test-model",
    )
    
    response = await agent.process_direct("Hello, agent!")
    
    # Should get error message about LLM being disabled
    assert response is not None
    assert "Memory-Only Mode" in response
    assert "LLM access is currently disabled" in response
    # Note: Internal components (light_reasoner, etc.) may still call provider
    # but the main agent loop should handle the denial correctly


@pytest.mark.asyncio
async def test_agent_loop_llm_adapter_usage(mock_provider, temp_workspace):
    """Test that AgentLoop uses LLM adapter correctly."""
    state.llm_enabled = True
    
    bus = MessageBus()
    agent = AgentLoop(
        bus=bus,
        provider=mock_provider,
        workspace=temp_workspace,
        model="test-model",
    )
    
    # Verify adapter was created
    assert hasattr(agent, 'llm_adapter')
    assert agent.llm_adapter is not None
    assert agent.llm_adapter.provider == mock_provider
    
    # Verify adapter is used for chat
    await agent.process_direct("Test message")
    
    # Verify the underlying provider was called through the adapter
    mock_provider.chat.assert_called()
