"""Session store for episodic memory persistence.

This module provides local-only YAML-based session storage for tracking
episodic interactions and events. Sessions are stored in workspace/sessions/
and can be archived, inspected, and consolidated into the relational cache.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class SessionStore:
    """Manages episodic session storage in YAML format.

    Sessions are stored in workspace/sessions/ with one YAML file per session.
    Each session contains a chronological log of events (interactions, state changes, etc.).

    Attributes:
        workspace: Path to the workspace directory
        sessions_dir: Path to workspace/sessions/
        archive_dir: Path to workspace/sessions/archive/
    """

    SCHEMA_VERSION = 1

    def __init__(self, workspace: Path):
        """Initialize session store.

        Args:
            workspace: Path to workspace directory
        """
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / "sessions"
        self.archive_dir = self.sessions_dir / "archive"

        # Ensure directories exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> Path:
        """Start a new session or resume an existing one.

        Args:
            session_id: Optional session ID. If None, generates one with timestamp.
            metadata: Optional metadata dictionary to store with session.

        Returns:
            Path to the session file.
        """
        if session_id is None:
            # Generate session ID with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp}"

        session_path = self.sessions_dir / f"{session_id}.yaml"

        # If session doesn't exist, create it
        if not session_path.exists():
            session_data = {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
                "events": []
            }
            self._write_yaml(session_path, session_data)

        return session_path

    def load(self, session_id: str) -> dict[str, Any]:
        """Load session data from file.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary

        Raises:
            FileNotFoundError: If session doesn't exist
        """
        session_path = self.sessions_dir / f"{session_id}.yaml"

        if not session_path.exists():
            # Check if it's in archive
            archive_path = self.archive_dir / f"{session_id}.yaml"
            if archive_path.exists():
                session_path = archive_path
            else:
                raise FileNotFoundError(f"Session '{session_id}' not found")

        return self._read_yaml(session_path)

    def save(self, session_id: str, data: dict[str, Any]) -> Path:
        """Save session data to file.

        Args:
            session_id: Session identifier
            data: Complete session data dictionary

        Returns:
            Path to the session file
        """
        session_path = self.sessions_dir / f"{session_id}.yaml"
        self._write_yaml(session_path, data)
        return session_path

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: dict[str, Any]
    ) -> Path:
        """Append an event to the session log.

        Args:
            session_id: Session identifier
            event_type: Type of event (e.g., 'interaction', 'state_change', 'reasoning')
            payload: Event data dictionary

        Returns:
            Path to the session file
        """
        session_data = self.load(session_id)

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": payload
        }

        session_data["events"].append(event)
        return self.save(session_id, session_data)

    def inspect(
        self,
        session_id: str,
        limit: int = 50,
        types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Inspect session events with optional filtering.

        Args:
            session_id: Session identifier
            limit: Maximum number of events to return (most recent first)
            types: Optional list of event types to filter by

        Returns:
            List of event dictionaries
        """
        session_data = self.load(session_id)
        events = session_data.get("events", [])

        # Filter by types if specified
        if types:
            events = [e for e in events if e.get("type") in types]

        # Return most recent events up to limit
        return events[-limit:] if len(events) > limit else events

    def trim(self, session_id: str, keep_types: list[str]) -> Path:
        """Remove events not matching specified types.

        Args:
            session_id: Session identifier
            keep_types: List of event types to keep (all others removed)

        Returns:
            Path to the session file
        """
        session_data = self.load(session_id)

        # Filter events to keep only specified types
        filtered_events = [
            e for e in session_data.get("events", [])
            if e.get("type") in keep_types
        ]

        session_data["events"] = filtered_events
        return self.save(session_id, session_data)

    def archive(self, session_id: str) -> Path:
        """Move session to archive directory.

        Args:
            session_id: Session identifier

        Returns:
            Path to the archived session file
        """
        session_path = self.sessions_dir / f"{session_id}.yaml"

        if not session_path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")

        archive_path = self.archive_dir / f"{session_id}.yaml"
        shutil.move(str(session_path), str(archive_path))

        return archive_path

    def list_sessions(self, include_archived: bool = False) -> list[dict[str, Any]]:
        """List all sessions with basic metadata.

        Args:
            include_archived: Whether to include archived sessions

        Returns:
            List of session info dictionaries with id, created_at, event_count
        """
        sessions = []

        # Active sessions
        for session_file in sorted(self.sessions_dir.glob("*.yaml")):
            try:
                data = self._read_yaml(session_file)
                sessions.append({
                    "session_id": data.get("session_id", session_file.stem),
                    "created_at": data.get("created_at", ""),
                    "event_count": len(data.get("events", [])),
                    "archived": False
                })
            except Exception:
                # Skip malformed files
                pass

        # Archived sessions
        if include_archived:
            for session_file in sorted(self.archive_dir.glob("*.yaml")):
                try:
                    data = self._read_yaml(session_file)
                    sessions.append({
                        "session_id": data.get("session_id", session_file.stem),
                        "created_at": data.get("created_at", ""),
                        "event_count": len(data.get("events", [])),
                        "archived": True
                    })
                except Exception:
                    # Skip malformed files
                    pass

        return sessions

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """Read YAML file safely."""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _write_yaml(self, path: Path, data: dict[str, Any]) -> None:
        """Write YAML file safely."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
