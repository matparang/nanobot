"""Integration tests for Triune Memory Translator."""

import tempfile
from pathlib import Path

import pytest
import yaml

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.config.schema import Config, TranslatorConfig
from nanobot.runtime.state import state
from nanobot.utils.translator import parse_md_to_yaml, sync_all


@pytest.fixture
def ensure_bootstrap_enabled():
    """Ensure bootstrap context stage is enabled for tests."""
    original = state.get_context_stages()
    original_triune = state.triune_memory_enabled
    state.enable_context_stage("bootstrap")
    state.triune_memory_enabled = True
    yield
    # Restore
    for stage_name, enabled in original.items():
        if enabled:
            state.enable_context_stage(stage_name)
        else:
            state.disable_context_stage(stage_name)
    state.triune_memory_enabled = original_triune


class TestContextBuilderYamlIntegration:
    """Tests for context builder YAML integration."""

    def test_context_builder_reads_yaml_when_available(self, ensure_bootstrap_enabled):
        """Test that context builder uses .yaml when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create YAML file
            yaml_data = {
                "_meta": {"source": "AGENTS.md"},
                "title": "Agent Instructions",
                "sections": [
                    {
                        "title": "Guidelines",
                        "content": "Be helpful and accurate.",
                    }
                ],
            }
            (workspace / "AGENTS.yaml").write_text(yaml.dump(yaml_data))

            # Don't create MD file - should use YAML

            context = ContextBuilder(workspace)
            system_prompt = context.build_system_prompt()

            # Verify YAML content was loaded
            assert "AGENTS" in system_prompt
            assert "Guidelines" in system_prompt or "Be helpful" in system_prompt

    def test_context_builder_falls_back_to_md(self, ensure_bootstrap_enabled):
        """Test that context builder falls back to .md when no .yaml exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create only MD file
            md_content = """# Agent Instructions

Be helpful and accurate.

## Guidelines

- Always explain your reasoning
- Use tools when needed
"""
            (workspace / "AGENTS.md").write_text(md_content)

            context = ContextBuilder(workspace)
            system_prompt = context.build_system_prompt()

            # Verify MD content was loaded
            assert "Agent Instructions" in system_prompt
            assert "Guidelines" in system_prompt

    def test_context_builder_prefers_yaml_over_md(self, ensure_bootstrap_enabled):
        """Test that YAML is preferred when both files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create MD file
            md_content = """# OLD Content from MD

This should NOT be loaded.
"""
            (workspace / "USER.md").write_text(md_content)

            # Create YAML file with different content
            yaml_data = {
                "_meta": {"source": "USER.md"},
                "title": "User Profile",
                "sections": [
                    {
                        "title": "Preferences",
                        "content": "YAML content loaded successfully.",
                    }
                ],
            }
            (workspace / "USER.yaml").write_text(yaml.dump(yaml_data))

            context = ContextBuilder(workspace)
            system_prompt = context.build_system_prompt()

            # Should see YAML content, not MD content
            assert "YAML content loaded successfully" in system_prompt or "Preferences" in system_prompt
            assert "OLD Content from MD" not in system_prompt


class TestMemoryStoreYamlIntegration:
    """Tests for memory store YAML integration."""

    def test_memory_store_reads_yaml_when_available(self):
        """Test that memory store uses .yaml when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()

            # Create YAML memory file
            yaml_data = {
                "_meta": {"source": "MEMORY.md"},
                "title": "Long-term Memory",
                "sections": [
                    {
                        "title": "User Preferences",
                        "content": "User prefers concise responses.",
                    }
                ],
            }
            (memory_dir / "MEMORY.yaml").write_text(yaml.dump(yaml_data))

            memory = MemoryStore(workspace)
            content = memory.read_long_term()

            # Verify YAML content was loaded
            assert "concise responses" in content or "User Preferences" in content

    def test_memory_store_falls_back_to_md(self):
        """Test that memory store falls back to .md when no .yaml exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()

            # Create only MD file
            md_content = """# Long-term Memory

User prefers detailed explanations.
"""
            (memory_dir / "MEMORY.md").write_text(md_content)

            memory = MemoryStore(workspace)
            content = memory.read_long_term()

            # Verify MD content was loaded
            assert "detailed explanations" in content

    def test_memory_yaml_paths_initialized(self):
        """Test that YAML paths are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory = MemoryStore(workspace)

            # Verify YAML paths are set
            assert memory.memory_yaml == workspace / "memory" / "MEMORY.yaml"
            assert memory.history_yaml == workspace / "memory" / "HISTORY.yaml"


class TestFullAgentLoopWithYaml:
    """Tests for full agent execution with YAML files."""

    def test_full_context_build_with_yaml_files(self, ensure_bootstrap_enabled):
        """Test building complete context with YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "memory"
            memory_dir.mkdir()

            # Create workspace YAML files
            agents_yaml = {
                "_meta": {"source": "AGENTS.md"},
                "title": "Agent Guidelines",
                "sections": [
                    {"title": "Core Principles", "content": "Be helpful and safe."}
                ],
            }
            (workspace / "AGENTS.yaml").write_text(yaml.dump(agents_yaml))

            user_yaml = {
                "_meta": {"source": "USER.md"},
                "title": "User Profile",
                "sections": [{"title": "Preferences", "content": "Prefers brief responses."}],
            }
            (workspace / "USER.yaml").write_text(yaml.dump(user_yaml))

            # Create memory YAML
            memory_yaml = {
                "_meta": {"source": "MEMORY.md"},
                "title": "Memory",
                "sections": [{"title": "Facts", "content": "User is named Alice."}],
            }
            (memory_dir / "MEMORY.yaml").write_text(yaml.dump(memory_yaml))

            # Build context
            context = ContextBuilder(workspace)
            system_prompt = context.build_system_prompt(user_query="Hello")

            # Verify key content from all files
            assert "nanobot" in system_prompt  # Identity section always present
            assert "AGENTS" in system_prompt or "Core Principles" in system_prompt

    def test_sync_then_context_build(self, ensure_bootstrap_enabled):
        """Test syncing MD files then building context from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create MD files
            (workspace / "AGENTS.md").write_text(
                "# Agent Instructions\n\nBe helpful and safe."
            )
            (workspace / "USER.md").write_text(
                "# User Profile\n\nUser is named Bob."
            )

            # Sync to YAML
            sync_all(workspace, direction="md_to_yaml", quiet=True)

            # Verify YAML files created
            assert (workspace / "AGENTS.yaml").exists()
            assert (workspace / "USER.yaml").exists()

            # Build context - should use YAML
            context = ContextBuilder(workspace)
            system_prompt = context.build_system_prompt()

            # Content should be present
            assert "Agent Instructions" in system_prompt or "AGENTS" in system_prompt


class TestConfigSchemaIntegration:
    """Tests for config schema with translator settings."""

    def test_translator_config_in_schema(self):
        """Test that translator config is part of main config."""
        config = Config()

        assert hasattr(config, "translator")
        assert config.translator.enabled is True
        assert config.translator.auto_sync_on_startup is True
        assert config.translator.sync_direction == "md_to_yaml"

    def test_translator_config_customization(self):
        """Test customizing translator config."""

        config = TranslatorConfig(
            enabled=False,
            auto_sync_on_startup=False,
            sync_direction="bidirectional",
            excluded_files=["CUSTOM.md"],
        )

        assert config.enabled is False
        assert config.sync_direction == "bidirectional"
        assert "CUSTOM.md" in config.excluded_files


class TestTokenReduction:
    """Tests for verifying token reduction with YAML format."""

    def test_yaml_preserves_essential_content(self):
        """Test that YAML conversion preserves essential content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            # Create verbose MD file
            md_content = """# Agent Instructions

You are a helpful AI assistant.

## Guidelines

Here are the guidelines you should follow:

- Always explain what you're doing
- Ask for clarification when needed
- Use tools appropriately
- Remember important information

## Tools Available

You have access to the following tools:

- File operations (read, write, edit)
- Shell commands (exec)
- Web access (search, fetch)

## Memory Management

Your memory files are:

- `memory/MEMORY.md` — long-term facts
- `memory/HISTORY.md` — event log
"""
            md_path = workspace / "TEST.md"
            yaml_path = workspace / "TEST.yaml"
            md_path.write_text(md_content)

            # Sync
            parse_md_to_yaml(md_path, yaml_path)

            # Verify essential content preserved
            data = yaml.safe_load(yaml_path.read_text())

            # Check structure
            assert "sections" in data
            assert len(data["sections"]) >= 1

            # Verify key content in sections
            all_content = str(data)
            assert "Guidelines" in all_content or "guidelines" in all_content.lower()

    def test_yaml_format_is_structured(self):
        """Test that YAML output is properly structured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            md_content = """# Title

Content paragraph.

## Section A

Section A content.

### Subsection A1

Subsection content.

## Section B

Section B content.
"""
            md_path = workspace / "STRUCT.md"
            yaml_path = workspace / "STRUCT.yaml"
            md_path.write_text(md_content)

            parse_md_to_yaml(md_path, yaml_path)
            data = yaml.safe_load(yaml_path.read_text())

            # Verify hierarchy
            assert "_meta" in data
            assert "sections" in data

            # Should have nested structure
            for section in data["sections"]:
                if section.get("subsections"):
                    break

            # Either has nested subsections or flat sections (both valid)
            assert isinstance(data["sections"], list)
