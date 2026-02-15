"""Tests for context builder."""

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.latent import LatentReasoner
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import FractalNode, SuperpositionalState
from nanobot.providers.base import LLMResponse


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


class _FakeProvider:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.call_count = 0

    async def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        index = min(self.call_count, len(self.payloads) - 1)
        self.call_count += 1
        return LLMResponse(content=json.dumps(self.payloads[index]))

    def get_default_model(self) -> str:
        return "fake-model"


def test_hybrid_scoring_correctness():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(
            Path(tmpdir),
            config={
                "semantic_weight": 0.5,
                "entanglement_weight": 0.3,
                "importance_weight": 0.2,
            },
        )
        score = memory._score_candidate(vec_score=1.0, normalized_entanglement=0.5, importance=0.5)
        assert score == pytest.approx(0.75)


def test_beam_pruning_preserves_score_ordering():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(Path(tmpdir), config={"beam_prune_k": 2})

        node_a = memory.save_fractal_node("alpha", ["alpha"], "alpha")
        memory.save_fractal_node("alpha 2", ["alpha"], "alpha")
        memory.save_fractal_node("alpha 3", ["alpha"], "alpha")

        ranked = memory.get_entangled_context("alpha", top_k=3)
        assert len(ranked) == 2
        assert ranked[0].id == node_a.id
        assert len({node.id for node in ranked}) == 2


def test_importance_decay_adjusts_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(
            Path(tmpdir),
            config={
                "semantic_weight": 0.0,
                "entanglement_weight": 0.0,
                "importance_weight": 1.0,
                "importance_decay_rate": 0.2,
            },
        )
        node = memory.save_fractal_node("important", ["important"], "important")
        decayed_importance = memory._get_decayed_importance(node)
        assert decayed_importance == pytest.approx(0.8)


def test_memory_retrieval_with_importance_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryStore(Path(tmpdir))
        memory.save_fractal_node("retrieval works", ["retrieve"], "retrieval summary")
        context = memory.retrieve_relevant_nodes("retrieve", k=1)
        assert "retrieval works" in context


@pytest.mark.asyncio
async def test_iterative_deepening_stops_at_entropy_threshold():
    provider = _FakeProvider(
        [
            {
                "hypotheses": [{"intent": "first", "confidence": 0.5, "reasoning": "a"}],
                "entropy": 0.9,
                "strategic_direction": "continue",
            },
            {
                "hypotheses": [{"intent": "second", "confidence": 0.9, "reasoning": "b"}],
                "entropy": 0.2,
                "strategic_direction": "stop",
            },
            {
                "hypotheses": [{"intent": "third", "confidence": 0.9, "reasoning": "c"}],
                "entropy": 0.1,
                "strategic_direction": "unused",
            },
        ]
    )
    reasoner = LatentReasoner(
        provider=provider,  # type: ignore[arg-type]
        model="fake",
        max_depth=3,
        entropy_threshold=0.5,
    )
    state = await reasoner.reason("msg", "ctx")
    assert provider.call_count == 2
    assert state.strategic_direction == "stop"


@pytest.mark.asyncio
async def test_monte_carlo_sampling_reduces_hypotheses():
    provider = _FakeProvider(
        [
            {
                "hypotheses": [
                    {"intent": "a", "confidence": 0.7, "reasoning": "a"},
                    {"intent": "b", "confidence": 0.2, "reasoning": "b"},
                    {"intent": "c", "confidence": 0.1, "reasoning": "c"},
                ],
                "entropy": 0.6,
                "strategic_direction": "sample",
            }
        ]
    )
    reasoner = LatentReasoner(
        provider=provider,  # type: ignore[arg-type]
        model="fake",
        monte_carlo_samples=50,
        monte_carlo_top_k=2,
    )
    state = await reasoner.reason("msg", "ctx")
    assert len(state.hypotheses) <= 2
