import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.runtime.state import state

runner = CliRunner()


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.utils.helpers.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_latent_command_toggles_runtime_state():
    previous = state.latent_reasoning_enabled
    try:
        result = runner.invoke(app, ["latent"], input="1\n")
        assert result.exit_code == 0
        assert state.latent_reasoning_enabled is True

        result = runner.invoke(app, ["latent"], input="2\n")
        assert result.exit_code == 0
        assert state.latent_reasoning_enabled is False
    finally:
        state.latent_reasoning_enabled = previous


def test_latent_command_rejects_invalid_choice():
    result = runner.invoke(app, ["latent"], input="99\n")
    assert result.exit_code == 1


def test_memory_toggle_commands_update_runtime_state():
    previous_mem0 = state.mem0_enabled
    previous_fractal = state.fractal_memory_enabled
    previous_entangled = state.entangled_memory_enabled
    previous_triune = state.triune_memory_enabled
    try:
        result = runner.invoke(app, ["mem0"], input="1\n")
        assert result.exit_code == 0
        assert state.mem0_enabled is True

        result = runner.invoke(app, ["fractal"], input="1\n")
        assert result.exit_code == 0
        assert state.fractal_memory_enabled is True

        result = runner.invoke(app, ["entangled"], input="2\n")
        assert result.exit_code == 0
        assert state.entangled_memory_enabled is False

        result = runner.invoke(app, ["triune"], input="2\n")
        assert result.exit_code == 0
        assert state.triune_memory_enabled is False
    finally:
        state.mem0_enabled = previous_mem0
        state.fractal_memory_enabled = previous_fractal
        state.entangled_memory_enabled = previous_entangled
        state.triune_memory_enabled = previous_triune


def test_toggle_commands_accept_direct_action_arguments():
    previous = state.latent_reasoning_enabled, state.mem0_enabled, state.heartbeat_enabled
    try:
        result = runner.invoke(app, ["latent", "on"])
        assert result.exit_code == 0
        assert state.latent_reasoning_enabled is True

        result = runner.invoke(app, ["mem0", "off"])
        assert result.exit_code == 0
        assert state.mem0_enabled is False

        result = runner.invoke(app, ["heartbeat", "off"])
        assert result.exit_code == 0
        assert state.heartbeat_enabled is False
    finally:
        (
            state.latent_reasoning_enabled,
            state.mem0_enabled,
            state.heartbeat_enabled,
        ) = previous


def test_baseline_command_enter_and_exit():
    previous = state.get_all_toggles()
    try:
        result = runner.invoke(app, ["baseline", "enter", "--force"])
        assert result.exit_code == 0
        assert state.baseline_active is True
        assert all(v is False for v in state.get_all_toggles().values())

        result = runner.invoke(app, ["baseline", "exit"])
        assert result.exit_code == 0
        assert state.baseline_active is False
        assert state.get_all_toggles() == previous
    finally:
        if state.baseline_active:
            state.exit_baseline_mode(restore=False)
        state.restore_toggles(previous)


def test_state_commands_inspect_and_clear(tmp_path):
    with patch("nanobot.utils.helpers.get_workspace_path") as mock_ws, patch(
        "nanobot.config.loader.get_data_dir"
    ) as mock_data_dir:
        workspace = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        (workspace / "memory").mkdir(parents=True)
        (workspace / "memory" / "ALS.json").write_text("{}", encoding="utf-8")
        mock_ws.return_value = workspace
        mock_data_dir.return_value = data_dir

        inspect_result = runner.invoke(app, ["state", "inspect", "als"])
        assert inspect_result.exit_code == 0
        assert "ALS" in inspect_result.stdout

        clear_result = runner.invoke(app, ["state", "clear", "als", "--dry-run", "--force"])
        assert clear_result.exit_code == 0
        assert "Would clear" in clear_result.stdout
