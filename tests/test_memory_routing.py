"""Tests for entropy-based routing and memory operations."""

import json
import tempfile
from pathlib import Path

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_types import ActiveLearningState, FractalNode


class TestMemoryRouting:
    """Test suite for entropy-based routing and memory updates."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def memory_store(self, temp_workspace):
        """Create a MemoryStore instance for testing."""
        config = {
            "clarify_entropy_threshold": 0.8,
        }
        return MemoryStore(temp_workspace, config)

    def test_append_long_term_creates_file(self, memory_store):
        """Test that append_long_term creates file if it doesn't exist."""
        memory_store.append_long_term("First entry")
        
        assert memory_store.memory_file.exists()
        content = memory_store.memory_file.read_text()
        assert "First entry" in content

    def test_append_long_term_preserves_existing(self, memory_store):
        """Test that append_long_term preserves existing content."""
        memory_store.write_long_term("Initial content\n")
        memory_store.append_long_term("Additional entry")
        
        content = memory_store.memory_file.read_text()
        assert "Initial content" in content
        assert "Additional entry" in content

    def test_append_long_term_multiple_times(self, memory_store):
        """Test multiple appends build up content."""
        memory_store.append_long_term("Entry 1")
        memory_store.append_long_term("Entry 2")
        memory_store.append_long_term("Entry 3")
        
        content = memory_store.memory_file.read_text()
        assert "Entry 1" in content
        assert "Entry 2" in content
        assert "Entry 3" in content

    def test_route_latent_state_low_entropy_to_fractal(self, memory_store):
        """Test that low entropy routes to Fractal Memory."""
        hypotheses = [
            {"intent": "create file", "confidence": 0.9, "reasoning": "clear intent"}
        ]
        
        memory_store.route_latent_state(
            user_message="Please create a file",
            hypotheses=hypotheses,
            entropy=0.3,  # Low entropy
            strategic_direction="Execute file creation",
        )
        
        # Check that fractal node was created
        index_data = json.loads(memory_store.index_file.read_text())
        assert len(index_data) > 0
        
        # Should contain latent-reasoning tag
        assert any("latent-reasoning" in entry.get("tags", []) for entry in index_data)
        
        # Pattern cache should not be created for low entropy
        assert not memory_store.pattern_cache_file.exists()

    def test_route_latent_state_high_entropy_to_cache(self, memory_store):
        """Test that high entropy routes to Pattern Cache."""
        hypotheses = [
            {"intent": "h1", "confidence": 0.5, "reasoning": "r1"},
            {"intent": "h2", "confidence": 0.5, "reasoning": "r2"}
        ]
        
        memory_store.route_latent_state(
            user_message="What do you mean?",
            hypotheses=hypotheses,
            entropy=0.95,  # High entropy
            strategic_direction="Need clarification",
        )
        
        # Check that pattern cache was created
        assert memory_store.pattern_cache_file.exists()
        
        entries = memory_store.get_pattern_cache_entries()
        assert len(entries) == 1
        assert entries[0]["entropy"] == 0.95
        assert entries[0]["user_message"] == "What do you mean?"

    def test_route_latent_state_updates_memory_md(self, memory_store):
        """Test that routing updates MEMORY.md."""
        hypotheses = [{"intent": "test", "confidence": 0.8, "reasoning": "test"}]
        
        memory_store.route_latent_state(
            user_message="Test message",
            hypotheses=hypotheses,
            entropy=0.5,
            strategic_direction="Test strategy",
        )
        
        content = memory_store.memory_file.read_text()
        assert "Teaching Flow" in content
        assert "Test message" in content
        assert "entropy=0.5" in content
        assert "Test strategy" in content

    def test_route_latent_state_updates_history_md(self, memory_store):
        """Test that routing updates HISTORY.md."""
        hypotheses = [{"intent": "test", "confidence": 0.8, "reasoning": "test"}]
        
        # Use a longer message to trigger entropy in history
        long_message = "This is a longer test message that exceeds one hundred characters to ensure the entropy information is included in the history entry."
        
        memory_store.route_latent_state(
            user_message=long_message,
            hypotheses=hypotheses,
            entropy=0.5,
            strategic_direction="Test strategy",
        )
        
        content = memory_store.history_file.read_text()
        assert long_message[:50] in content  # Check for message prefix
        assert "entropy=0.5" in content

    def test_route_latent_state_updates_als(self, memory_store):
        """Test that routing updates ALS with reflection."""
        hypotheses = [{"intent": "test", "confidence": 0.8, "reasoning": "test"}]
        
        memory_store.route_latent_state(
            user_message="Test message",
            hypotheses=hypotheses,
            entropy=0.5,
            strategic_direction="Test strategy",
        )
        
        als = ActiveLearningState.model_validate_json(
            memory_store.als_file.read_text()
        )
        
        assert len(als.recent_reflections) > 0
        assert "Latent routing" in als.recent_reflections[0]
        assert "entropy=0.5" in als.recent_reflections[0]

    def test_route_latent_state_with_model_metadata(self, memory_store):
        """Test that model metadata is stored in pattern cache."""
        hypotheses = [{"intent": "test", "confidence": 0.5, "reasoning": "test"}]
        
        memory_store.route_latent_state(
            user_message="Test",
            hypotheses=hypotheses,
            entropy=0.9,  # High entropy -> cache
            strategic_direction="clarify",
            model="gpt-4",
            provider="openai",
        )
        
        entries = memory_store.get_pattern_cache_entries()
        assert entries[0]["model"] == "gpt-4"
        assert entries[0]["provider"] == "openai"

    def test_get_pattern_cache_entries_empty(self, memory_store):
        """Test getting entries from non-existent cache."""
        entries = memory_store.get_pattern_cache_entries()
        assert entries == []

    def test_get_pattern_cache_entries_limit(self, memory_store):
        """Test that limit works correctly."""
        # Add 5 entries
        for i in range(5):
            memory_store._route_to_pattern_cache(
                user_message=f"Message {i}",
                hypotheses=[{"intent": "test", "confidence": 0.5, "reasoning": "test"}],
                entropy=0.9,
                strategic_direction="test",
                model=None,
                provider=None,
            )
        
        # Get only 3
        entries = memory_store.get_pattern_cache_entries(limit=3)
        assert len(entries) == 3
        
        # Should be most recent first
        assert "Message 4" in entries[0]["user_message"]
        assert "Message 3" in entries[1]["user_message"]
        assert "Message 2" in entries[2]["user_message"]

    def test_get_pattern_cache_entries_invalid_json_skipped(self, memory_store):
        """Test that invalid JSON lines are skipped gracefully."""
        # Write some entries, including invalid JSON
        with open(memory_store.pattern_cache_file, "w") as f:
            f.write('{"valid": "entry1", "entropy": 0.9}\n')
            f.write('invalid json line\n')
            f.write('{"valid": "entry2", "entropy": 0.8}\n')
        
        entries = memory_store.get_pattern_cache_entries()
        assert len(entries) == 2  # Only valid entries
        assert entries[0]["valid"] == "entry2"
        assert entries[1]["valid"] == "entry1"

    def test_teaching_flow_format(self, memory_store):
        """Test teaching flow format structure."""
        from datetime import datetime
        
        hypotheses = [
            {"intent": "intent1", "confidence": 0.8, "reasoning": "reason1"},
            {"intent": "intent2", "confidence": 0.6, "reasoning": "reason2"}
        ]
        
        entry = memory_store._format_teaching_flow(
            timestamp=datetime(2024, 1, 15, 10, 30),
            user_message="Test message",
            hypotheses=hypotheses,
            entropy=0.5,
            strategic_direction="Test strategy"
        )
        
        assert "## Teaching Flow - 2024-01-15 10:30" in entry
        assert "User: Test message" in entry
        assert "entropy=0.500" in entry
        assert "intent1" in entry
        assert "confidence=0.80" in entry
        assert "reason1" in entry
        assert "Strategic Direction" in entry
        assert "Test strategy" in entry

    def test_multiple_routing_calls_preserve_history(self, memory_store):
        """Test that multiple routing calls preserve all history."""
        for i in range(3):
            memory_store.route_latent_state(
                user_message=f"Message {i}",
                hypotheses=[{"intent": f"intent{i}", "confidence": 0.8, "reasoning": "test"}],
                entropy=0.5,
                strategic_direction=f"Strategy {i}",
            )
        
        # Check MEMORY.md has all teaching flows
        memory_content = memory_store.memory_file.read_text()
        assert "Message 0" in memory_content
        assert "Message 1" in memory_content
        assert "Message 2" in memory_content
        
        # Check HISTORY.md has all entries
        history_content = memory_store.history_file.read_text()
        assert "Message 0" in history_content
        assert "Message 1" in history_content
        assert "Message 2" in history_content

    def test_routing_respects_custom_threshold(self, temp_workspace):
        """Test that custom entropy threshold is respected."""
        # Set low threshold
        config = {"clarify_entropy_threshold": 0.5}
        memory = MemoryStore(temp_workspace, config)
        
        # Entropy 0.6 should go to cache (> 0.5)
        memory.route_latent_state(
            user_message="Test",
            hypotheses=[{"intent": "test", "confidence": 0.7, "reasoning": "test"}],
            entropy=0.6,
            strategic_direction="test",
        )
        
        assert memory.pattern_cache_file.exists()

    def test_routing_default_threshold(self, temp_workspace):
        """Test default threshold of 0.8 when not configured."""
        memory = MemoryStore(temp_workspace, {})
        
        # Entropy 0.7 should go to fractal (< 0.8)
        memory.route_latent_state(
            user_message="Test",
            hypotheses=[{"intent": "test", "confidence": 0.9, "reasoning": "test"}],
            entropy=0.7,
            strategic_direction="test",
        )
        
        # Fractal node should be created
        index_data = json.loads(memory.index_file.read_text())
        assert len(index_data) > 0
        
        # Pattern cache should not be created
        assert not memory.pattern_cache_file.exists()
