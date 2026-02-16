import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.providers.base import LLMResponse
from nanobot.providers.litellm_provider import LiteLLMProvider


@pytest.mark.asyncio
async def test_chat_passes_tools_to_local_curl() -> None:
    provider = LiteLLMProvider(api_base="http://localhost:11434")
    provider._chat_via_curl = AsyncMock(return_value=LLMResponse(content="ok"))

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

    await provider.chat(messages=messages, tools=tools, model="demo-model")

    provider._chat_via_curl.assert_awaited_once_with(messages, "demo-model", 4096, 0.7, tools)


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
