"""Smoke tests for CLI memory commands."""


import pytest
from typer.testing import CliRunner

from nanobot.agent.memory import MemoryStore
from nanobot.cli.memory_commands import app

runner = CliRunner()


@pytest.fixture
def setup_memory_dir(monkeypatch, tmp_path):
    """Setup a test memory directory with some data."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    memory = MemoryStore(workspace_path, {"clarify_entropy_threshold": 0.8})

    # Add some test data
    memory.route_latent_state(
        user_message="Test message 1",
        hypotheses=[
            {"intent": "intent1", "confidence": 0.9, "reasoning": "reason1"}
        ],
        entropy=0.3,  # Low entropy
        strategic_direction="Clear strategy",
    )

    memory.route_latent_state(
        user_message="Ambiguous message",
        hypotheses=[
            {"intent": "intent2a", "confidence": 0.5, "reasoning": "reason2a"},
            {"intent": "intent2b", "confidence": 0.5, "reasoning": "reason2b"}
        ],
        entropy=0.9,  # High entropy
        strategic_direction="Need clarification",
    )

    # Mock config to return our test workspace
    # Use a local variable to avoid closure issues
    ws_str = str(workspace_path)

    def mock_load_config():
        class MockConfig:
            class MockAgents:
                class MockDefaults:
                    workspace = ws_str
                defaults = MockDefaults()
            agents = MockAgents()
        return MockConfig()

    monkeypatch.setattr("nanobot.cli.memory_commands.load_config", mock_load_config)

    return workspace_path


def test_memory_status_command(setup_memory_dir):
    """Test that memory status command runs without error."""
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Active Learning State" in result.stdout
    assert "Current Focus" in result.stdout


def test_memory_list_hypotheses_command(setup_memory_dir):
    """Test that list-hypotheses command runs and shows data."""
    result = runner.invoke(app, ["list-hypotheses"])

    assert result.exit_code == 0
    assert "Hypothesis Entries" in result.stdout or "No hypothesis entries" in result.stdout


def test_memory_fractal_command(setup_memory_dir):
    """Test that fractal command runs without error."""
    result = runner.invoke(app, ["fractal"])

    assert result.exit_code == 0
    # Should show nodes or "No fractal nodes found"
    assert "Fractal Memory" in result.stdout or "No fractal" in result.stdout


def test_memory_cache_command(setup_memory_dir):
    """Test that cache command runs and shows pattern cache."""
    result = runner.invoke(app, ["cache"])

    assert result.exit_code == 0
    # Should show cache entries or "No cache entries"
    assert "Pattern Cache" in result.stdout or "No cache" in result.stdout


def test_memory_history_command(setup_memory_dir):
    """Test that history command runs without error."""
    result = runner.invoke(app, ["history"])

    assert result.exit_code == 0
    assert "Recent History" in result.stdout or "No history" in result.stdout


def test_memory_list_hypotheses_with_high_entropy_filter(setup_memory_dir):
    """Test filtering high entropy entries."""
    result = runner.invoke(app, ["list-hypotheses", "--high-entropy"])

    assert result.exit_code == 0


def test_memory_fractal_with_tag_filter(setup_memory_dir):
    """Test filtering fractal nodes by tag."""
    result = runner.invoke(app, ["fractal", "--tag", "latent-reasoning"])

    assert result.exit_code == 0


def test_memory_commands_with_custom_limits():
    """Test that limit options work."""
    # These should at least not crash even with no data
    result1 = runner.invoke(app, ["list-hypotheses", "--limit", "5"])
    result2 = runner.invoke(app, ["fractal", "--limit", "3"])
    result3 = runner.invoke(app, ["cache", "--limit", "7"])
    result4 = runner.invoke(app, ["history", "--limit", "15"])

    # All should succeed (exit code 0) or gracefully show no data
    assert result1.exit_code == 0
    assert result2.exit_code == 0
    assert result3.exit_code == 0
    assert result4.exit_code == 0


def test_memory_status_no_als_file(tmp_path, monkeypatch):
    """Test status command shows default ALS when file is auto-created."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "memory").mkdir()

    ws_str = str(workspace_path)

    def mock_load_config():
        class MockConfig:
            class MockAgents:
                class MockDefaults:
                    workspace = ws_str
                defaults = MockDefaults()
            agents = MockAgents()
        return MockConfig()

    monkeypatch.setattr("nanobot.cli.memory_commands.load_config", mock_load_config)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    # MemoryStore auto-creates ALS with defaults
    assert "Active Learning State" in result.stdout
    assert "General Assistance" in result.stdout  # Default focus


def test_memory_list_hypotheses_no_cache(tmp_path, monkeypatch):
    """Test list-hypotheses when no pattern cache exists."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "memory").mkdir()

    ws_str = str(workspace_path)

    def mock_load_config():
        class MockConfig:
            class MockAgents:
                class MockDefaults:
                    workspace = ws_str
                defaults = MockDefaults()
            agents = MockAgents()
        return MockConfig()

    monkeypatch.setattr("nanobot.cli.memory_commands.load_config", mock_load_config)

    result = runner.invoke(app, ["list-hypotheses"])

    assert result.exit_code == 0
    assert "No pattern cache found" in result.stdout
