"""Tests for entanglement expansion in fractal memory retrieval."""

import tempfile
from pathlib import Path

from nanobot.agent.memory import MemoryStore


def test_retrieve_relevant_nodes_includes_entangled_nodes():
    """Querying node A should also pull strongly entangled node B."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        memory = MemoryStore(workspace)

        node_a = memory.save_fractal_node(
            content="alpha topic content",
            tags=["alpha"],
            summary="alpha summary",
        )
        node_b = memory.save_fractal_node(
            content="entangled companion context",
            tags=["beta"],
            summary="beta summary",
        )

        node_a.entangled_ids[node_b.id] = 0.9
        memory._update_node(node_a)

        result = memory.retrieve_relevant_nodes("alpha", k=1)
        assert "alpha topic content" in result
        assert "entangled companion context" in result
