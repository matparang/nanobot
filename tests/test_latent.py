"""Tests for latent reasoning module."""

import pytest

from nanobot.agent.latent import LatentReasoner
from nanobot.providers.base import LLMResponse


class _MockProvider:
    def __init__(self, content: str):
        self._content = content

    async def chat(self, messages, tools=None, model=None, max_tokens=4096, temperature=0.7):  # noqa: ANN001
        return LLMResponse(content=self._content)


@pytest.mark.asyncio
async def test_latent_reasoner_parses_graph_json():
    provider = _MockProvider(
        '{"nodes":["h1","h2"],"edges":[["h1","h2",0.8]],'
        '"probabilities":{"h1":0.6,"h2":0.4},"selected_hypothesis_index":1}'
    )
    reasoner = LatentReasoner(provider, model="test-model")

    graph, strategy = await reasoner.reason("test context")

    assert graph.nodes == ["h1", "h2"]
    assert graph.selected_hypothesis_index == 1
    assert strategy == "Selected Hypothesis 1"


@pytest.mark.asyncio
async def test_latent_reasoner_falls_back_on_invalid_json():
    provider = _MockProvider("not-json")
    reasoner = LatentReasoner(provider, model="test-model")

    graph, strategy = await reasoner.reason("test context")

    assert graph.nodes == []
    assert strategy == "Linear Fallback"
