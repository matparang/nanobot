import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.providers.base import LLMResponse
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.runtime.state import state


@pytest.mark.asyncio
async def test_chat_passes_tools_to_local_curl() -> None:
    previous = state.latent_reasoning_enabled
    try:
        state.latent_reasoning_enabled = False
        provider = LiteLLMProvider(api_base="http://localhost:11434")
        provider._chat_via_curl = AsyncMock(return_value=LLMResponse(content="ok"))

        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

        await provider.chat(messages=messages, tools=tools, model="demo-model")

        provider._chat_via_curl.assert_awaited_once_with(messages, "demo-model", 512, 0.7, tools)
    finally:
        state.latent_reasoning_enabled = previous


@pytest.mark.asyncio
async def test_chat_via_curl_includes_and_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = LiteLLMProvider(api_base="http://localhost:11434")
    captured_command: list[str] = []

    async def fake_to_thread(function, command, **kwargs):
        captured_command.extend(command)
        payload = json.loads(command[command.index("-d") + 1])
        assert payload["tools"][0]["function"]["name"] == "ping"
        assert payload["tool_choice"] == "auto"
        assert command[3].endswith("/v1/chat/completions")
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {
                                            "name": "ping",
                                            "arguments": "{\"value\": 1}",
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr("nanobot.providers.litellm_provider.asyncio.to_thread", fake_to_thread)

    response = await provider._chat_via_curl(
        messages=[{"role": "user", "content": "hi"}],
        model="hosted_vllm/demo-model",
        max_tokens=128,
        temperature=0.3,
        tools=[{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}],
    )

    assert captured_command
    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "ping"
    assert response.tool_calls[0].arguments == {"value": 1}


@pytest.mark.asyncio
async def test_chat_via_curl_handles_api_base_with_v1_and_ollama_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(api_base="http://localhost:11434/v1")

    async def fake_to_thread(function, command, **kwargs):
        payload = json.loads(command[command.index("-d") + 1])
        assert command[3] == "http://localhost:11434/v1/chat/completions"
        assert payload["model"] == "qwen2.5:7b-instruct"
        assert payload["stream"] is False
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "ok"},
                        }
                    ]
                }
            ),
        )

    monkeypatch.setattr("nanobot.providers.litellm_provider.asyncio.to_thread", fake_to_thread)

    response = await provider._chat_via_curl(
        messages=[{"role": "user", "content": "hi"}],
        model="ollama/qwen2.5:7b-instruct",
        max_tokens=128,
        temperature=0.3,
    )

    assert response.content == "ok"


@pytest.mark.asyncio
async def test_chat_adds_concise_instruction_when_latent_reasoning_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = state.latent_reasoning_enabled
    try:
        state.latent_reasoning_enabled = False
        provider = LiteLLMProvider()
        captured_kwargs = {}

        async def fake_acompletion(**kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")]
            )

        monkeypatch.setattr("nanobot.providers.litellm_provider.acompletion", fake_acompletion)
        await provider.chat(messages=[{"role": "system", "content": "Base prompt"}], model="demo-model")

        assert captured_kwargs["max_tokens"] == 512
        assert captured_kwargs["messages"][0]["content"].endswith("Respond concisely.")
    finally:
        state.latent_reasoning_enabled = previous


@pytest.mark.asyncio
async def test_chat_adds_deep_reasoning_instruction_when_latent_reasoning_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = state.latent_reasoning_enabled
    try:
        state.latent_reasoning_enabled = True
        provider = LiteLLMProvider()
        captured_kwargs = {}

        async def fake_acompletion(**kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")]
            )

        monkeypatch.setattr("nanobot.providers.litellm_provider.acompletion", fake_acompletion)
        await provider.chat(
            messages=[{"role": "system", "content": "Base prompt"}],
            model="demo-model",
            max_tokens=128,
        )

        assert captured_kwargs["max_tokens"] == 1024
        assert "Use step-by-step reasoning." in captured_kwargs["messages"][0]["content"]
    finally:
        state.latent_reasoning_enabled = previous
