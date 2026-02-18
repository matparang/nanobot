"""Tests for context builder."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.context import ContextBuilder
from nanobot.agent.latent import LatentReasoner
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import FractalNode, SuperpositionalState
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Config
from nanobot.gateway import build_health_payload
from nanobot.providers.base import LLMResponse
from nanobot.runtime.state import state


def test_cognitive_directive_in_system_prompt():
    """Test that cognitive directive is included in system prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create a context builder
        context = ContextBuilder(workspace)

        # Build system prompt
        system_prompt = context.build_system_prompt()

        # Verify cognitive directive is present
        assert "# COGNITIVE DIRECTIVE" in system_prompt
        assert "Memory retrieved in the RESOURCES & MEMORY section is authoritative internal knowledge" in system_prompt
        assert "You must use retrieved memory as primary reasoning substrate" in system_prompt
        assert "First consult retrieved memory" in system_prompt
        assert "Prefer memory over tools" in system_prompt
        assert "Prefer memory over assumptions" in system_prompt
        assert "Use tools only if memory does not contain the answer" in system_prompt
        assert "Do not ignore relevant memory" in system_prompt


def test_cognitive_directive_placement():
    """Test that cognitive directive appears after bootstrap and before resources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create a context builder
        context = ContextBuilder(workspace)

        # Build system prompt
        system_prompt = context.build_system_prompt(user_query="test query")

        # Find positions
        cognitive_pos = system_prompt.find("# COGNITIVE DIRECTIVE")
        resources_pos = system_prompt.find("# RESOURCES & MEMORY")

        # Cognitive directive should be present
        assert cognitive_pos != -1, "Cognitive directive not found in system prompt"

        # If resources section exists, cognitive directive should come before it
        if resources_pos != -1:
            assert cognitive_pos < resources_pos, "Cognitive directive should appear before resources section"


def test_system_prompt_respects_runtime_memory_toggles(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        context = ContextBuilder(workspace)
        latent_state = SuperpositionalState.model_validate_json(
            '{"hypotheses":[{"intent":"test","confidence":1.0,"reasoning":"test"}],"entropy":0.1,"strategic_direction":"go"}'
        )

        monkeypatch.setattr(context, "_load_bootstrap_files", lambda: "BOOTSTRAP")
        monkeypatch.setattr(context.memory, "get_als_context", lambda: "ALS")
        monkeypatch.setattr(context.memory, "read_long_term", lambda: "CORE")
        monkeypatch.setattr(context.memory, "retrieve_relevant_nodes", lambda *args, **kwargs: "FRACTAL")
        monkeypatch.setattr(context.memory, "get_entangled_context", lambda *args, **kwargs: [])

        previous = (
            state.triune_memory_enabled,
            state.latent_reasoning_enabled,
            state.mem0_enabled,
            state.fractal_memory_enabled,
            state.entangled_memory_enabled,
        )
        previous_stages = state.get_context_stages()
        try:
            state.triune_memory_enabled = False
            state.latent_reasoning_enabled = False
            state.mem0_enabled = False
            state.fractal_memory_enabled = False
            state.entangled_memory_enabled = False
            prompt = context.build_system_prompt(user_query="hello", latent_state=latent_state)
            assert "BOOTSTRAP" not in prompt
            assert "<latent_state>" not in prompt
            assert "FRACTAL" not in prompt

            state.triune_memory_enabled = True
            state.latent_reasoning_enabled = True
            state.fractal_memory_enabled = True
            state.entangled_memory_enabled = True
            # Also ensure context stages are enabled
            state.enable_context_stage("bootstrap")
            state.enable_context_stage("latent_state")
            state.enable_context_stage("fractal_memory")
            prompt = context.build_system_prompt(user_query="hello", latent_state=latent_state)
            assert "BOOTSTRAP" in prompt
            assert "<latent_state>" in prompt
            assert "FRACTAL" in prompt
        finally:
            (
                state.triune_memory_enabled,
                state.latent_reasoning_enabled,
                state.mem0_enabled,
                state.fractal_memory_enabled,
                state.entangled_memory_enabled,
            ) = previous
            # Restore context stages
            for stage_name, enabled in previous_stages.items():
                if enabled:
                    state.enable_context_stage(stage_name)
                else:
                    state.disable_context_stage(stage_name)


def test_fractal_node_serialization():
    node = FractalNode(
        content="Test content",
        context_summary="Summary",
        entangled_ids={"node_abc": 0.8},
    )
    json_str = node.model_dump_json()
    loaded = FractalNode.model_validate_json(json_str)
    assert loaded.entangled_ids["node_abc"] == 0.8


def test_latent_state_parsing():
    raw_json = """
    {
        "hypotheses": [],
        "entropy": 0.5,
        "strategic_direction": "go"
    }
    """
    state = SuperpositionalState.model_validate_json(raw_json)
    assert state.entropy == 0.5
    assert state.hypotheses == []


def test_memory_normalization_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(Path(tmpdir))

        node_a = memory.save_fractal_node(
            content="alpha vector score candidate",
            tags=["alpha"],
            summary="alpha summary",
        )
        node_b = memory.save_fractal_node(
            content="beta",
            tags=["beta"],
            summary="beta summary",
        )
        node_c = memory.save_fractal_node(
            content="gamma",
            tags=["gamma"],
            summary="gamma summary",
        )

        node_b.entangled_ids[node_a.id] = 1.0
        node_c.entangled_ids[node_a.id] = 1.0
        memory._update_node(node_b)
        memory._update_node(node_c)

        ranked = memory.get_entangled_context("alpha", top_k=2)
        assert ranked
        assert ranked[0].id == node_a.id


def test_retrieve_relevant_nodes_respects_runtime_toggles():
    previous = state.fractal_memory_enabled, state.mem0_enabled
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryStore(Path(tmpdir))
            memory.save_fractal_node(content="alpha", tags=["alpha"], summary="alpha summary")

            state.fractal_memory_enabled = False
            state.mem0_enabled = False
            assert memory.retrieve_relevant_nodes("alpha", k=1) == ""

            state.fractal_memory_enabled = True
            result = memory.retrieve_relevant_nodes("alpha", k=1)
            assert "Relevant Resources" in result
    finally:
        state.fractal_memory_enabled, state.mem0_enabled = previous


def test_entanglement_cycle_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(Path(tmpdir))
        node_a = memory.save_fractal_node(content="A", tags=["alpha"], summary="alpha")
        node_b = memory.save_fractal_node(content="B", tags=["beta"], summary="beta")

        node_a.entangled_ids[node_b.id] = 0.9
        node_b.entangled_ids[node_a.id] = 0.9
        memory._update_node(node_a)
        memory._update_node(node_b)

        context = memory.get_entangled_context(query="alpha", top_k=5)
        ids = [node.id for node in context]

        assert node_a.id in ids
        assert node_b.id in ids
        assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_latent_reasoning_pipeline():
    mock_provider = AsyncMock()
    mock_provider.chat.return_value = LLMResponse(
        content='{"hypotheses":[{"intent":"search","confidence":0.9,"reasoning":"direct lookup"}]}'
    )

    reasoner = LatentReasoner(
        provider=mock_provider,
        model="test-model",
        memory_config={"clarify_entropy_threshold": 0.5, "latent_max_depth": 1, "beam_width": 3},
    )
    state = await reasoner.reason("Find weather in Tokyo", context_summary="")

    assert state.hypotheses[0].intent == "search"
    assert state.hypotheses[0].confidence == 0.9
    assert state.entropy == 0.0
    assert mock_provider.chat.call_count >= 1


@pytest.mark.asyncio
async def test_latent_reasoning_retries_transient_errors():
    mock_provider = AsyncMock()
    mock_provider.chat.side_effect = [
        ConnectionError("transient failure"),
        ConnectionError("transient failure"),
        LLMResponse(content='{"hypotheses":[{"intent":"search","confidence":0.8}]}'),
    ]

    reasoner = LatentReasoner(provider=mock_provider, model="test-model")
    state = await reasoner.reason("retry please", context_summary="")

    assert state.hypotheses
    assert mock_provider.chat.call_count == 3


@pytest.mark.asyncio
async def test_latent_reasoning_iterative_beam_pruning():
    provider = AsyncMock()
    provider.chat.return_value = LLMResponse(
        content='{"hypotheses":[{"intent":"a","confidence":0.5,"reasoning":"a"},{"intent":"b","confidence":0.5,"reasoning":"b"}]}'
    )

    reasoner = LatentReasoner(
        provider=provider,
        model="test-model",
        memory_config={
            "clarify_entropy_threshold": 0.9,
            "latent_max_depth": 1,
            "beam_width": 2,
            "monte_carlo_samples": 1,
        },
    )
    reasoner._monte_carlo_expand = AsyncMock(
        return_value=[
            reasoner._parse_hypotheses(
                '{"hypotheses":[{"intent":"c","confidence":0.9,"reasoning":"c"}]}'
            )[0]
        ]
    )

    state = await reasoner.reason("ambiguous", context_summary="ctx")
    assert len(state.hypotheses) == 2
    assert any(h.intent == "c" for h in state.hypotheses)
    reasoner._monte_carlo_expand.assert_awaited_once()


@pytest.mark.asyncio
async def test_monte_carlo_expand_uses_sampling_count():
    provider = AsyncMock()
    provider.chat.side_effect = [
        LLMResponse(content='{"hypotheses":[{"intent":"h1","confidence":0.6,"reasoning":"r1"}]}'),
        LLMResponse(content='{"hypotheses":[{"intent":"h2","confidence":0.7,"reasoning":"r2"}]}'),
    ]
    reasoner = LatentReasoner(
        provider=provider,
        model="test-model",
        memory_config={"monte_carlo_samples": 2},
    )

    base_hypotheses = reasoner._parse_hypotheses(
        '{"hypotheses":[{"intent":"seed","confidence":0.5,"reasoning":"seed"}]}'
    )
    expanded = await reasoner._monte_carlo_expand(base_hypotheses, "q", "ctx")

    assert [h.intent for h in expanded] == ["h1", "h2"]
    assert provider.chat.await_count == 2


@pytest.mark.asyncio
async def test_latent_retry_attempts_are_configurable():
    provider = AsyncMock()
    provider.chat.side_effect = ConnectionError("always fails")
    reasoner = LatentReasoner(
        provider=provider,
        model="test-model",
        memory_config={"latent_retry_attempts": 2},
    )

    state = await reasoner.reason("retry", context_summary="")
    assert state.hypotheses == []
    assert provider.chat.await_count == 2


@pytest.mark.asyncio
async def test_agent_loop_skips_latent_reasoning_when_disabled():
    previous = state.latent_reasoning_enabled
    try:
        state.latent_reasoning_enabled = False
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = SimpleNamespace(
                get_default_model=lambda: "test-model",
                chat=AsyncMock(return_value=LLMResponse(content="ok")),
            )
            loop = AgentLoop(
                bus=MessageBus(),
                provider=provider,
                workspace=Path(tmpdir),
                enable_latent_reasoning=False,
            )
            loop.latent_engine.reason = AsyncMock()

            response = await loop._process_message(
                InboundMessage(channel="cli", sender_id="user", chat_id="test", content="hello")
            )

            assert response is not None
            assert response.content == "ok"
            loop.latent_engine.reason.assert_not_awaited()
    finally:
        state.latent_reasoning_enabled = previous


def test_agent_loop_enable_latent_reasoning_reads_runtime_state():
    previous = state.latent_reasoning_enabled
    try:
        state.latent_reasoning_enabled = True
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = SimpleNamespace(get_default_model=lambda: "test-model", chat=AsyncMock())
            loop = AgentLoop(
                bus=MessageBus(),
                provider=provider,
                workspace=Path(tmpdir),
                enable_latent_reasoning=True,
            )
            assert loop.enable_latent_reasoning is True
            state.latent_reasoning_enabled = False
            assert loop.enable_latent_reasoning is False
    finally:
        state.latent_reasoning_enabled = previous


def test_entanglement_hop_limit_prevents_deep_chain():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(Path(tmpdir), config={"max_entanglement_hops": 1})
        node_a = memory.save_fractal_node(content="A", tags=["alpha"], summary="alpha")
        node_b = memory.save_fractal_node(content="B", tags=["beta"], summary="beta")
        node_c = memory.save_fractal_node(content="C", tags=["gamma"], summary="gamma")

        node_a.entangled_ids[node_b.id] = 1.0
        node_b.entangled_ids[node_c.id] = 1.0
        node_c.entangled_ids[node_a.id] = 1.0
        memory._update_node(node_a)
        memory._update_node(node_b)
        memory._update_node(node_c)

        context = memory.get_entangled_context(query="alpha", top_k=5)
        ids = {node.id for node in context}
        assert node_a.id in ids
        assert node_b.id in ids
        assert node_c.id not in ids


def test_importance_decay_penalizes_old_nodes():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(
            Path(tmpdir),
            config={
                "semantic_weight": 0.0,
                "entanglement_weight": 0.0,
                "importance_weight": 1.0,
                "importance_decay_rate": 0.05,
            },
        )
        old_node = memory.save_fractal_node(content="old", tags=["alpha"], summary="alpha")
        new_node = memory.save_fractal_node(content="new", tags=["alpha"], summary="alpha")

        old_node.importance = 1.0
        new_node.importance = 0.5
        old_node.timestamp = datetime.now() - timedelta(hours=100)
        memory._update_node(old_node)
        memory._update_node(new_node)

        context = memory.get_entangled_context(query="alpha", top_k=2)
        assert [node.id for node in context] == [new_node.id, old_node.id]


class DummyChannel(BaseChannel):
    name = "dummy"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg) -> None:
        return None


@pytest.mark.asyncio
async def test_base_channel_blocks_sender_not_in_allow_from():
    bus = MessageBus()
    config = SimpleNamespace(allow_from=["allowed-user"])
    channel = DummyChannel(config=config, bus=bus)

    await channel._handle_message(sender_id="blocked-user", chat_id="chat1", content="hello")
    assert bus.inbound_size == 0


def test_health_payload_schema(monkeypatch):
    monkeypatch.setattr("nanobot.gateway.load_config", lambda: Config())
    monkeypatch.setattr("nanobot.gateway._version", lambda: "vtest")

    payload = build_health_payload()

    assert payload["status"] == "ok"
    assert payload["version"] == "vtest"
    # Triune field is optional depending on workspace setup
    assert "status" in payload
    assert "version" in payload
    assert "memory" in payload
    assert "latent_reasoning" in payload
    assert "enabled" in payload["memory"]
    assert "entanglement_weight" in payload["memory"]
    assert "enabled" in payload["latent_reasoning"]
    assert "depth" in payload["latent_reasoning"]
