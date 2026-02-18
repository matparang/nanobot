"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import copy
import json
import os
import subprocess
import ipaddress
from typing import Any
from urllib.parse import urlparse

import litellm
from litellm import acompletion

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


def _latent_enabled() -> bool:
    """
    Check if latent reasoning is enabled.
    
    This is a local helper to avoid circular imports.
    The actual gate logic is in nanobot.agent.latent.latent_enabled().
    """
    from nanobot.runtime.state import state
    
    # Baseline mode always disables latent reasoning
    if state.baseline_active:
        return False
    
    # Check the runtime toggle
    return state.latent_reasoning_enabled


class LiteLLMProvider(LLMProvider):
    CURL_TIMEOUT = 120
    PROCESS_TIMEOUT_BUFFER = 5
    LOCAL_MODEL_PREFIXES = ("hosted_vllm/", "ollama/")

    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, api_key, api_base)

        # Configure environment variables
        if api_key:
            self._setup_env(api_key, api_base, default_model)

        if api_base:
            litellm.api_base = api_base

        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True

    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model

        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        model = self._resolve_model(model or self.default_model)
        prepared_messages = copy.deepcopy(messages)

        # Use centralized latent gate for reasoning mode
        if _latent_enabled():
            max_tokens = max(max_tokens, 1024)
            for msg in prepared_messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if "step-by-step reasoning" not in content.lower():
                        msg["content"] = (
                            f"{content}\n\nUse step-by-step reasoning. Think carefully before answering."
                        ).strip()
                    break
        else:
            max_tokens = min(max_tokens, 512)
            for msg in prepared_messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if "respond concisely" not in content.lower():
                        msg["content"] = f"{content}\n\nRespond concisely.".strip()
                    break

        # Special handling for local endpoints (Ollama, vLLM)
        if self.api_base:
            parsed_base = urlparse(self.api_base)
            if not parsed_base.scheme:
                parsed_base = urlparse(f"http://{self.api_base}")
            hostname = parsed_base.hostname
            is_loopback = False
            if hostname:
                try:
                    ip_obj = ipaddress.ip_address(hostname)
                    is_loopback = ip_obj.is_loopback
                except ValueError:
                    if hostname == "localhost" or hostname.endswith(".localhost"):
                        is_loopback = True

            if is_loopback:
                return await self._chat_via_curl(prepared_messages, model, max_tokens, temperature, tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": prepared_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)

        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key

        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # Return error as content for graceful handling
            return LLMResponse(
                content=f"Error calling LLM: {str(e)}",
                finish_reason="error",
            )

    async def _chat_via_curl(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Fallback: Use curl for local endpoints (Ollama, vLLM).
        
        Args:
            messages: Conversation messages in OpenAI format.
            model: Resolved model name to send.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            tools: Optional list of tool definitions in OpenAI format.
        
        Returns:
            LLMResponse containing the model content or an error message.
        
        Workaround for Python HTTP client incompatibility with Ollama.
        curl works reliably where httpx/requests hang. HTTPS endpoints rely on system
        trust; self-signed certificates require adding a trusted CA (we deliberately
        avoid --insecure).
        """
        # Strip prefixes for curl request
        model_name = model
        for prefix in self.LOCAL_MODEL_PREFIXES:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):]
                break
        base_url = self.api_base.rstrip("/")
        if base_url.endswith("/v1"):
            endpoint = f"{base_url}/chat/completions"
        else:
            endpoint = f"{base_url}/v1/chat/completions"
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme not in ("http", "https"):
            return LLMResponse(
                content="Local endpoint error: invalid api_base scheme",
                finish_reason="error",
            )
        curl_timeout = self.CURL_TIMEOUT
        process_timeout = curl_timeout + self.PROCESS_TIMEOUT_BUFFER
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    'curl', '-X', 'POST',
                    endpoint,
                    '-H', 'Content-Type: application/json',
                    '-d', json.dumps(payload),
                    '--fail',
                    '--max-time', str(curl_timeout),
                    '--silent',
                ],
                capture_output=True,
                text=True,
                timeout=process_timeout,
            )
            
            if result.returncode != 0:
                return LLMResponse(
                    content=f"Local endpoint error: {result.stderr}",
                    finish_reason="error",
                )

            if not result.stdout:
                return LLMResponse(
                    content="Local endpoint error: empty response",
                    finish_reason="error",
                )

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as parse_err:
                return LLMResponse(
                    content=f"Local endpoint error: invalid JSON response ({parse_err})",
                    finish_reason="error",
                )

            # Safely extract choices with defaults
            if not isinstance(data, dict):
                return LLMResponse(
                    content="Local endpoint error: response is not a dict",
                    finish_reason="error",
                )
            
            choices = data.get('choices')
            if not choices or not isinstance(choices, list):
                return LLMResponse(
                    content="Local endpoint error: missing or invalid choices in response",
                    finish_reason="error",
                )

            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get('message')
            if not isinstance(message, dict):
                message = {}
            
            content = message.get('content')
            tool_calls = []
            
            # Safely parse tool_calls
            raw_tool_calls = message.get('tool_calls')
            if isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    
                    function = tc.get('function')
                    if not isinstance(function, dict):
                        function = {}
                    
                    arguments = function.get('arguments', '{}')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments) if arguments.strip() else {}
                        except json.JSONDecodeError:
                            arguments = {"raw": arguments}
                    elif not isinstance(arguments, dict):
                        arguments = {}
                    
                    tool_calls.append(
                        ToolCallRequest(
                            id=tc.get('id', ''),
                            name=function.get('name', ''),
                            arguments=arguments,
                        )
                    )

            # Allow empty content if tool_calls are present
            if content is None and not tool_calls:
                return LLMResponse(
                    content="Local endpoint error: missing content and tool_calls in response",
                    finish_reason="error",
                )
            
            return LLMResponse(
                # OpenAI-compatible tool calls can legitimately return null content.
                content=content if content is not None else "",
                tool_calls=tool_calls,
                finish_reason=choice.get('finish_reason', 'stop'),
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error calling local endpoint: {str(e)}",
                finish_reason="error",
            )

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        reasoning_content = getattr(message, "reasoning_content", None)

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )

    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
