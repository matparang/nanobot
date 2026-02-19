"""
LLM Adapter Module - Single point of LLM access enforcement.

This module acts as the sole gateway to LLM providers, enforcing the global
LLM_ENABLED flag and providing comprehensive logging for the execution chain.
"""

from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.runtime.state import state


class LLMAccessDeniedError(Exception):
    """Raised when LLM access is attempted while globally disabled."""
    pass


class LLMAdapter:
    """
    Adapter that wraps LLM provider calls with enforcement and logging.
    
    This is the single entry point for all LLM access, ensuring that:
    1. LLM calls are blocked when globally disabled
    2. All LLM calls are logged for audit trail
    3. Errors are clear and actionable
    """
    
    def __init__(self, provider: LLMProvider):
        """
        Initialize the LLM adapter.
        
        Args:
            provider: The underlying LLM provider to use when enabled.
        """
        self.provider = provider
        logger.info(f"[LLM Adapter] Initialized with provider: {provider.__class__.__name__}")
    
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request through the LLM adapter.
        
        This method enforces the global LLM_ENABLED flag before allowing
        any call to the underlying provider.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
            
        Raises:
            LLMAccessDeniedError: If LLM access is globally disabled.
        """
        # Check global LLM enabled state
        if not state.llm_enabled:
            error_msg = (
                "LLM access denied: LLM calls are globally disabled. "
                "Use --enable-llm flag or enable LLM through configuration."
            )
            logger.warning(f"[LLM Adapter] {error_msg}")
            raise LLMAccessDeniedError(error_msg)
        
        # Log the LLM call
        model_name = model or self.provider.get_default_model()
        logger.info(f"[LLM Adapter] Calling LLM provider (model={model_name})")
        logger.debug(f"[LLM Adapter] Request details: messages={len(messages)}, "
                    f"tools={len(tools) if tools else 0}, max_tokens={max_tokens}")
        
        try:
            # Make the actual LLM call
            response = await self.provider.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            logger.info(f"[LLM Adapter] LLM call completed (finish_reason={response.finish_reason})")
            logger.debug(f"[LLM Adapter] Response details: content_length={len(response.content) if response.content else 0}, "
                        f"tool_calls={len(response.tool_calls)}")
            
            return response
            
        except Exception as e:
            logger.error(f"[LLM Adapter] Error during LLM call: {str(e)}")
            raise
    
    def get_default_model(self) -> str:
        """Get the default model from the underlying provider."""
        return self.provider.get_default_model()
    
    def is_enabled(self) -> bool:
        """Check if LLM access is currently enabled."""
        return state.llm_enabled
