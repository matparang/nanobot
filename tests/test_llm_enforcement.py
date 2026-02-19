"""Tests for LLM adapter and enforcement."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from nanobot.llm_adapter import LLMAdapter, LLMAccessDeniedError
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.runtime.state import state


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=LLMProvider)
    provider.get_default_model.return_value = "test-model"
    provider.chat = AsyncMock(return_value=LLMResponse(
        content="Test response",
        finish_reason="stop"
    ))
    return provider


@pytest.fixture
def llm_adapter(mock_provider):
    """Create an LLM adapter with a mock provider."""
    return LLMAdapter(mock_provider)


@pytest.fixture(autouse=True)
def reset_llm_state():
    """Reset LLM state before and after each test."""
    original_state = state.llm_enabled
    state.llm_enabled = True  # Default to enabled
    yield
    state.llm_enabled = original_state


class TestLLMAdapter:
    """Tests for LLM adapter enforcement."""

    @pytest.mark.asyncio
    async def test_chat_when_llm_enabled(self, llm_adapter, mock_provider):
        """Test that chat works normally when LLM is enabled."""
        state.llm_enabled = True
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await llm_adapter.chat(messages=messages)
        
        assert response.content == "Test response"
        mock_provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_when_llm_disabled(self, llm_adapter, mock_provider):
        """Test that chat raises error when LLM is disabled."""
        state.llm_enabled = False
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(LLMAccessDeniedError) as exc_info:
            await llm_adapter.chat(messages=messages)
        
        assert "LLM access denied" in str(exc_info.value)
        assert "globally disabled" in str(exc_info.value)
        mock_provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_with_tools_when_enabled(self, llm_adapter, mock_provider):
        """Test that tools are passed through when LLM is enabled."""
        state.llm_enabled = True
        
        messages = [{"role": "user", "content": "Use a tool"}]
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        await llm_adapter.chat(messages=messages, tools=tools)
        
        mock_provider.chat.assert_called_once()
        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_parameters_passed_through(self, llm_adapter, mock_provider):
        """Test that all parameters are correctly passed to provider."""
        state.llm_enabled = True
        
        messages = [{"role": "user", "content": "Test"}]
        await llm_adapter.chat(
            messages=messages,
            model="custom-model",
            max_tokens=2048,
            temperature=0.5
        )
        
        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.5

    def test_is_enabled_when_llm_enabled(self, llm_adapter):
        """Test is_enabled returns True when LLM is enabled."""
        state.llm_enabled = True
        assert llm_adapter.is_enabled() is True

    def test_is_enabled_when_llm_disabled(self, llm_adapter):
        """Test is_enabled returns False when LLM is disabled."""
        state.llm_enabled = False
        assert llm_adapter.is_enabled() is False

    def test_get_default_model(self, llm_adapter, mock_provider):
        """Test get_default_model delegates to provider."""
        model = llm_adapter.get_default_model()
        assert model == "test-model"
        mock_provider.get_default_model.assert_called_once()


class TestLLMStateToggling:
    """Tests for LLM state toggling via RuntimeState."""

    def test_llm_enabled_default(self):
        """Test that LLM is enabled by default."""
        # Reset to fresh state
        from nanobot.runtime.state import RuntimeState
        fresh_state = RuntimeState()
        assert fresh_state.llm_enabled is True

    def test_llm_enabled_can_be_disabled(self):
        """Test that LLM can be disabled via state."""
        state.llm_enabled = False
        assert state.llm_enabled is False
        
        state.llm_enabled = True
        assert state.llm_enabled is True

    def test_llm_state_persists_across_accesses(self):
        """Test that LLM state is consistent."""
        state.llm_enabled = False
        assert state.llm_enabled is False
        assert state.llm_enabled is False  # Should remain disabled
        
        state.llm_enabled = True
        assert state.llm_enabled is True
        assert state.llm_enabled is True  # Should remain enabled


class TestProviderHardBlock:
    """Tests for hard block enforcement in LiteLLM provider."""

    @pytest.mark.asyncio
    async def test_provider_blocks_when_llm_disabled(self):
        """Test that provider hard-blocks direct calls when LLM disabled."""
        from nanobot.providers.litellm_provider import LiteLLMProvider
        
        provider = LiteLLMProvider(
            api_key="test-key",
            default_model="test-model"
        )
        
        state.llm_enabled = False
        
        messages = [{"role": "user", "content": "Test"}]
        response = await provider.chat(messages=messages)
        
        # Should return error response, not raise
        assert response.finish_reason == "error"
        assert "HARD BLOCK" in response.content
        assert "globally disabled" in response.content

    @pytest.mark.asyncio
    async def test_provider_allows_when_llm_enabled(self):
        """Test that provider allows calls when LLM enabled."""
        from nanobot.providers.litellm_provider import LiteLLMProvider
        
        state.llm_enabled = True
        
        # Note: This test would make actual API calls without mocking
        # We'll just verify the state check passes
        provider = LiteLLMProvider(
            api_key="test-key",
            default_model="test-model"
        )
        
        # Mock the actual LiteLLM call to avoid real API calls
        with patch('nanobot.providers.litellm_provider.acompletion') as mock_acompletion:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = None
            mock_acompletion.return_value = mock_response
            
            messages = [{"role": "user", "content": "Test"}]
            response = await provider.chat(messages=messages)
            
            # Should get normal response
            assert response.content == "Test response"
            assert response.finish_reason == "stop"
            mock_acompletion.assert_called_once()
