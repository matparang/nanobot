import json

from nanobot.cli import toggle_utils
from nanobot.runtime.state import state


def test_batch_toggle_updates_mapped_toggles_and_logs(tmp_path, monkeypatch):
    log_path = tmp_path / "toggle.log"
    monkeypatch.setattr(toggle_utils, "get_toggle_log_path", lambda: log_path)
    monkeypatch.setattr(toggle_utils, "audit_log", lambda *args, **kwargs: None)

    previous = state.get_all_toggles()
    try:
        result = toggle_utils.batch_toggle(
            state,
            {"latent": True, "mem0": True, "heartbeat": False, "light_reasoner": True},
            log_source="test",
        )
        assert result["latent"] == (previous["latent_reasoning"], True)
        assert result["mem0"] == (previous["mem0"], True)
        assert result["heartbeat"] == (previous["heartbeat"], False)
        assert result["light_reasoner"] == (previous["light_reasoner"], True)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        assert {json.loads(line)["feature"] for line in lines} == {
            "latent",
            "mem0",
            "heartbeat",
            "light_reasoner",
        }
    finally:
        state.restore_toggles(previous)
