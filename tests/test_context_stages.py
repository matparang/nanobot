"""Tests for context stage control system."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_types import SuperpositionalState
from nanobot.runtime.state import state


@pytest.fixture
def temp_workspace():
    """Provide a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def save_and_restore_state():
    """Save runtime state before test and restore after."""
    original_state = state.get_context_stages()
    yield
    # Restore original state
    for stage_name, enabled in original_state.items():
        if enabled:
            state.enable_context_stage(stage_name)
        else:
            state.disable_context_stage(stage_name)


class TestContextStageRegistry:
    """Tests for context stage registry in RuntimeState."""

    def test_all_stages_enabled_by_default(self, save_and_restore_state):
        """Test that all context stages are enabled by default for backward compatibility."""
        stages = state.get_context_stages()
        assert all(stages.values()), "All context stages should be enabled by default"

    def test_enable_context_stage(self, save_and_restore_state):
        """Test enabling a context stage."""
        state.disable_context_stage("identity")
        assert not state.is_context_stage_enabled("identity")

        state.enable_context_stage("identity")
        assert state.is_context_stage_enabled("identity")

    def test_disable_context_stage(self, save_and_restore_state):
        """Test disabling a context stage."""
        assert state.is_context_stage_enabled("bootstrap")

        state.disable_context_stage("bootstrap")
        assert not state.is_context_stage_enabled("bootstrap")

    def test_toggle_context_stage(self, save_and_restore_state):
        """Test toggling a context stage."""
        original = state.is_context_stage_enabled("ALS")

        new_state = state.toggle_context_stage("ALS")
        assert new_state != original
        assert state.is_context_stage_enabled("ALS") == new_state

        # Toggle back
        new_state = state.toggle_context_stage("ALS")
        assert new_state == original

    def test_enable_all_context_stages(self, save_and_restore_state):
        """Test enabling all context stages."""
        state.disable_all_context_stages()
        stages = state.get_context_stages()
        assert not any(stages.values()), "All stages should be disabled"

        state.enable_all_context_stages()
        stages = state.get_context_stages()
        assert all(stages.values()), "All stages should be enabled"

    def test_disable_all_context_stages(self, save_and_restore_state):
        """Test disabling all context stages."""
        state.enable_all_context_stages()
        stages = state.get_context_stages()
        assert all(stages.values()), "All stages should be enabled"

        state.disable_all_context_stages()
        stages = state.get_context_stages()
        assert not any(stages.values()), "All stages should be disabled"

    def test_invalid_stage_name_raises_error(self, save_and_restore_state):
        """Test that invalid stage names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown context stage"):
            state.enable_context_stage("invalid_stage")

        with pytest.raises(ValueError, match="Unknown context stage"):
            state.disable_context_stage("invalid_stage")

        with pytest.raises(ValueError, match="Unknown context stage"):
            state.toggle_context_stage("invalid_stage")

    def test_get_enabled_context_stages(self, save_and_restore_state):
        """Test getting list of enabled context stages."""
        state.disable_all_context_stages()
        state.enable_context_stage("identity")
        state.enable_context_stage("user_message")

        enabled = state.get_enabled_context_stages()
        assert set(enabled) == {"identity", "user_message"}


class TestContextBuilderRespectStages:
    """Tests for ContextBuilder respecting context stage toggles."""

    def test_identity_stage_toggle(self, temp_workspace, save_and_restore_state):
        """Test that identity stage can be disabled."""
        context = ContextBuilder(temp_workspace)

        # With identity enabled
        state.enable_context_stage("identity")
        prompt = context.build_system_prompt()
        assert "# nanobot" in prompt
        assert "You are nanobot" in prompt

        # With identity disabled
        state.disable_context_stage("identity")
        prompt = context.build_system_prompt()
        assert "# nanobot" not in prompt
        assert "You are nanobot" not in prompt

    def test_bootstrap_stage_toggle(self, temp_workspace, save_and_restore_state):
        """Test that bootstrap stage can be disabled."""
        # Create a bootstrap file
        (temp_workspace / "AGENTS.md").write_text("# Test Bootstrap")

        context = ContextBuilder(temp_workspace)

        # Save original triune state
        original_triune = state.triune_memory_enabled

        try:
            # With bootstrap enabled
            state.enable_context_stage("bootstrap")
            state.triune_memory_enabled = True
            prompt = context.build_system_prompt()
            assert "Test Bootstrap" in prompt

            # With bootstrap disabled
            state.disable_context_stage("bootstrap")
            prompt = context.build_system_prompt()
            assert "Test Bootstrap" not in prompt
        finally:
            state.triune_memory_enabled = original_triune

    def test_latent_state_stage_toggle(self, temp_workspace, save_and_restore_state):
        """Test that latent_state stage can be disabled."""
        context = ContextBuilder(temp_workspace)
        latent_state = SuperpositionalState.model_validate_json(
            '{"hypotheses":[{"intent":"test","confidence":1.0,"reasoning":"test"}],'
            '"entropy":0.1,"strategic_direction":"go"}'
        )

        # Save original latent reasoning state
        original_latent = state.latent_reasoning_enabled

        try:
            # With latent_state enabled
            state.enable_context_stage("latent_state")
            state.latent_reasoning_enabled = True
            prompt = context.build_system_prompt(latent_state=latent_state)
            assert "<latent_state>" in prompt

            # With latent_state disabled
            state.disable_context_stage("latent_state")
            prompt = context.build_system_prompt(latent_state=latent_state)
            assert "<latent_state>" not in prompt
        finally:
            state.latent_reasoning_enabled = original_latent

    def test_core_memory_stage_toggle(self, temp_workspace, save_and_restore_state, monkeypatch):
        """Test that core_memory stage can be disabled."""
        context = ContextBuilder(temp_workspace)
        monkeypatch.setattr(context.memory, "read_long_term", lambda: "CORE MEMORY")

        # With core_memory enabled
        state.enable_context_stage("core_memory")
        prompt = context.build_system_prompt()
        assert "CORE MEMORY" in prompt

        # With core_memory disabled
        state.disable_context_stage("core_memory")
        prompt = context.build_system_prompt()
        assert "CORE MEMORY" not in prompt

    def test_skills_stage_toggle(self, temp_workspace, save_and_restore_state, monkeypatch):
        """Test that skills stage can be disabled."""
        context = ContextBuilder(temp_workspace)

        # Mock skills
        monkeypatch.setattr(context.skills, "get_always_skills", lambda: ["test_skill"])
        monkeypatch.setattr(context.skills, "load_skills_for_context", lambda skills: "ALWAYS SKILL")

        # With skills enabled
        state.enable_context_stage("skills")
        prompt = context.build_system_prompt()
        assert "ALWAYS SKILL" in prompt

        # With skills disabled
        state.disable_context_stage("skills")
        prompt = context.build_system_prompt()
        assert "ALWAYS SKILL" not in prompt

    def test_skills_summary_stage_toggle(self, temp_workspace, save_and_restore_state, monkeypatch):
        """Test that skills_summary stage can be disabled."""
        context = ContextBuilder(temp_workspace)

        # Mock skills summary
        monkeypatch.setattr(context.skills, "build_skills_summary", lambda: "SKILLS SUMMARY")

        # With skills_summary enabled
        state.enable_context_stage("skills_summary")
        prompt = context.build_system_prompt()
        assert "SKILLS SUMMARY" in prompt

        # With skills_summary disabled
        state.disable_context_stage("skills_summary")
        prompt = context.build_system_prompt()
        assert "SKILLS SUMMARY" not in prompt

    def test_conversation_history_stage_toggle(self, temp_workspace, save_and_restore_state):
        """Test that conversation_history stage can be disabled."""
        context = ContextBuilder(temp_workspace)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # With conversation_history enabled
        state.enable_context_stage("conversation_history")
        messages = context.build_messages(history, "Current message")
        # System message + history + user message
        assert len(messages) == 4
        assert messages[1] == history[0]
        assert messages[2] == history[1]

        # With conversation_history disabled
        state.disable_context_stage("conversation_history")
        messages = context.build_messages(history, "Current message")
        # System message + user message (no history)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_user_message_stage_toggle(self, temp_workspace, save_and_restore_state):
        """Test that user_message stage can be disabled."""
        context = ContextBuilder(temp_workspace)

        # With user_message enabled
        state.enable_context_stage("user_message")
        messages = context.build_messages([], "Test message")
        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test message"

        # With user_message disabled
        state.disable_context_stage("user_message")
        messages = context.build_messages([], "Test message")
        assert len(messages) == 1
        assert messages[0]["role"] == "system"


class TestBaselineModeIntegration:
    """Tests for baseline mode integration with context stages."""

    @pytest.fixture(autouse=True)
    def cleanup_baseline_mode(self):
        """Ensure baseline mode is exited after each test."""
        yield
        # Clean up after test
        if state.baseline_active:
            state.exit_baseline_mode(restore=False)

    def test_baseline_mode_disables_stages(self, save_and_restore_state):
        """Test that baseline mode disables all stages except identity and user_message."""
        # Enable all stages first
        state.enable_all_context_stages()

        # Enter baseline mode
        state.enter_baseline_mode()

        stages = state.get_context_stages()
        assert stages["identity"] is True
        assert stages["user_message"] is True
        assert stages["bootstrap"] is False
        assert stages["latent_state"] is False
        assert stages["ALS"] is False
        assert stages["core_memory"] is False
        assert stages["fractal_memory"] is False
        assert stages["entangled_memory"] is False
        assert stages["skills"] is False
        assert stages["skills_summary"] is False
        assert stages["conversation_history"] is False

    def test_baseline_mode_restores_stages(self, save_and_restore_state):
        """Test that exiting baseline mode restores context stages."""
        # Set custom stage configuration
        state.enable_all_context_stages()
        state.disable_context_stage("bootstrap")
        state.disable_context_stage("ALS")
        original = state.get_context_stages()

        # Enter baseline mode
        state.enter_baseline_mode()

        # Exit baseline mode and restore
        state.exit_baseline_mode(restore=True)

        restored = state.get_context_stages()
        assert restored == original

    def test_baseline_mode_no_restore(self, save_and_restore_state):
        """Test that exiting baseline mode without restore doesn't restore stages."""
        # Set custom stage configuration
        state.enable_all_context_stages()
        state.disable_context_stage("bootstrap")

        # Enter baseline mode
        state.enter_baseline_mode()
        baseline_stages = state.get_context_stages()

        # Exit baseline mode without restore
        state.exit_baseline_mode(restore=False)

        # Stages should remain as they were in baseline mode
        current_stages = state.get_context_stages()
        assert current_stages == baseline_stages


class TestDebugLogging:
    """Tests for debug logging of active context stages."""

    def test_build_system_prompt_logs_active_stages(self, temp_workspace, save_and_restore_state):
        """Test that build_system_prompt logs active context stages."""
        context = ContextBuilder(temp_workspace)

        # Disable some stages
        state.disable_context_stage("bootstrap")
        state.disable_context_stage("ALS")

        # Build system prompt and capture logs
        with patch("nanobot.agent.context.logger") as mock_logger:
            context.build_system_prompt()

            # Check that debug was called with active stages
            mock_logger.debug.assert_called()
            call_args = str(mock_logger.debug.call_args)
            assert "Active context stages" in call_args
