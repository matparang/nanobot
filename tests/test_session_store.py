"""Tests for session store functionality."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from nanobot.memory.session_store import SessionStore


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session_store(temp_workspace):
    """Create a SessionStore instance."""
    return SessionStore(temp_workspace)


def test_session_store_initialization(session_store, temp_workspace):
    """Test that SessionStore initializes directories correctly."""
    assert session_store.workspace == temp_workspace
    assert session_store.sessions_dir == temp_workspace / "sessions"
    assert session_store.archive_dir == temp_workspace / "sessions" / "archive"

    # Check directories exist
    assert session_store.sessions_dir.exists()
    assert session_store.archive_dir.exists()


def test_start_session_generates_id(session_store):
    """Test starting a session without providing an ID."""
    session_path = session_store.start()

    assert session_path.exists()
    assert session_path.suffix == ".yaml"
    assert session_path.name.startswith("session_")

    # Load and verify content
    data = yaml.safe_load(session_path.read_text())
    assert data["schema_version"] == 1
    assert "session_id" in data
    assert "created_at" in data
    assert data["events"] == []


def test_start_session_with_custom_id(session_store):
    """Test starting a session with a custom ID."""
    custom_id = "test_session_123"
    session_path = session_store.start(session_id=custom_id)

    assert session_path.name == f"{custom_id}.yaml"

    data = yaml.safe_load(session_path.read_text())
    assert data["session_id"] == custom_id


def test_start_session_with_metadata(session_store):
    """Test starting a session with custom metadata."""
    metadata = {"user": "testuser", "context": "unit_test"}
    session_path = session_store.start(session_id="meta_test", metadata=metadata)

    data = yaml.safe_load(session_path.read_text())
    assert data["metadata"] == metadata


def test_resume_existing_session(session_store):
    """Test that starting an existing session doesn't overwrite it."""
    session_id = "resume_test"

    # Start first time
    session_store.start(session_id=session_id)
    session_store.append_event(session_id, "test", {"msg": "first"})

    # Start again (should not overwrite)
    session_store.start(session_id=session_id)

    # Verify event still exists
    data = session_store.load(session_id)
    assert len(data["events"]) == 1
    assert data["events"][0]["payload"]["msg"] == "first"


def test_load_session(session_store):
    """Test loading session data."""
    session_id = "load_test"
    session_store.start(session_id=session_id, metadata={"key": "value"})

    data = session_store.load(session_id)

    assert data["session_id"] == session_id
    assert data["metadata"]["key"] == "value"
    assert isinstance(data["events"], list)


def test_load_nonexistent_session_raises_error(session_store):
    """Test that loading a nonexistent session raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        session_store.load("nonexistent_session")


def test_load_archived_session(session_store):
    """Test that load can find archived sessions."""
    session_id = "archive_load_test"
    session_store.start(session_id=session_id)
    session_store.archive(session_id)

    # Should still be able to load from archive
    data = session_store.load(session_id)
    assert data["session_id"] == session_id


def test_save_session(session_store):
    """Test saving session data."""
    session_id = "save_test"
    session_store.start(session_id=session_id)

    data = session_store.load(session_id)
    data["events"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "custom",
        "payload": {"test": "data"}
    })

    session_path = session_store.save(session_id, data)

    assert session_path.exists()

    # Verify saved data
    reloaded = session_store.load(session_id)
    assert len(reloaded["events"]) == 1
    assert reloaded["events"][0]["payload"]["test"] == "data"


def test_append_event(session_store):
    """Test appending events to a session."""
    session_id = "append_test"
    session_store.start(session_id=session_id)

    # Append first event
    session_store.append_event(
        session_id,
        "interaction",
        {"user_message": "Hello", "agent_response": "Hi"}
    )

    # Append second event
    session_store.append_event(
        session_id,
        "reasoning",
        {"hypotheses": [{"intent": "greeting"}], "entropy": 0.2}
    )

    # Verify both events
    data = session_store.load(session_id)
    assert len(data["events"]) == 2
    assert data["events"][0]["type"] == "interaction"
    assert data["events"][1]["type"] == "reasoning"


def test_inspect_session(session_store):
    """Test inspecting session events."""
    session_id = "inspect_test"
    session_store.start(session_id=session_id)

    # Add multiple events
    for i in range(10):
        session_store.append_event(
            session_id,
            "test_event",
            {"index": i}
        )

    # Inspect with limit
    events = session_store.inspect(session_id, limit=5)
    assert len(events) == 5
    assert events[-1]["payload"]["index"] == 9  # Most recent


def test_inspect_with_type_filter(session_store):
    """Test inspecting session with type filtering."""
    session_id = "filter_test"
    session_store.start(session_id=session_id)

    # Add different event types
    session_store.append_event(session_id, "type_a", {"data": "a1"})
    session_store.append_event(session_id, "type_b", {"data": "b1"})
    session_store.append_event(session_id, "type_a", {"data": "a2"})
    session_store.append_event(session_id, "type_c", {"data": "c1"})

    # Filter by type_a
    events = session_store.inspect(session_id, types=["type_a"])
    assert len(events) == 2
    assert all(e["type"] == "type_a" for e in events)


def test_trim_session(session_store):
    """Test trimming session events by type."""
    session_id = "trim_test"
    session_store.start(session_id=session_id)

    # Add various events
    session_store.append_event(session_id, "keep", {"data": "k1"})
    session_store.append_event(session_id, "remove", {"data": "r1"})
    session_store.append_event(session_id, "keep", {"data": "k2"})
    session_store.append_event(session_id, "remove", {"data": "r2"})

    # Trim to keep only "keep" type
    session_store.trim(session_id, keep_types=["keep"])

    # Verify only "keep" events remain
    data = session_store.load(session_id)
    assert len(data["events"]) == 2
    assert all(e["type"] == "keep" for e in data["events"])


def test_archive_session(session_store):
    """Test archiving a session."""
    session_id = "archive_test"
    session_path = session_store.start(session_id=session_id)

    # Verify session exists in main directory
    assert session_path.exists()

    # Archive the session
    archive_path = session_store.archive(session_id)

    # Verify moved to archive
    assert archive_path.exists()
    assert archive_path.parent == session_store.archive_dir
    assert not session_path.exists()


def test_archive_nonexistent_session_raises_error(session_store):
    """Test that archiving a nonexistent session raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        session_store.archive("nonexistent_session")


def test_list_sessions(session_store):
    """Test listing all sessions."""
    # Create multiple sessions
    session_store.start(session_id="session_1")
    session_store.start(session_id="session_2")
    session_store.append_event("session_1", "test", {"data": "test"})

    sessions = session_store.list_sessions()

    assert len(sessions) == 2
    assert any(s["session_id"] == "session_1" for s in sessions)
    assert any(s["session_id"] == "session_2" for s in sessions)

    # Check event count
    session_1 = next(s for s in sessions if s["session_id"] == "session_1")
    assert session_1["event_count"] == 1


def test_list_sessions_includes_archived(session_store):
    """Test listing sessions with archived flag."""
    # Create and archive a session
    session_store.start(session_id="active_session")
    session_store.start(session_id="archived_session")
    session_store.archive("archived_session")

    # List without archived
    active_only = session_store.list_sessions(include_archived=False)
    assert len(active_only) == 1
    assert active_only[0]["session_id"] == "active_session"

    # List with archived
    all_sessions = session_store.list_sessions(include_archived=True)
    assert len(all_sessions) == 2
    assert any(s["archived"] for s in all_sessions)
    assert any(not s["archived"] for s in all_sessions)


def test_persistence_across_instances(temp_workspace):
    """Test that sessions persist across SessionStore instances."""
    session_id = "persist_test"

    # Create session with first instance
    store1 = SessionStore(temp_workspace)
    store1.start(session_id=session_id)
    store1.append_event(session_id, "test", {"data": "persistent"})

    # Load with second instance
    store2 = SessionStore(temp_workspace)
    data = store2.load(session_id)

    assert data["session_id"] == session_id
    assert len(data["events"]) == 1
    assert data["events"][0]["payload"]["data"] == "persistent"
