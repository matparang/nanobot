from nanobot.cli import state_manager


def test_check_persistent_state_filters_category(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    (workspace / "memory").mkdir(parents=True)
    (workspace / "memory" / "ALS.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(state_manager, "get_workspace_path", lambda: workspace)
    monkeypatch.setattr(state_manager, "get_data_dir", lambda: data_dir)

    result = state_manager.check_persistent_state("als")
    assert list(result) == ["ALS"]
    assert result["ALS"]["exists"] is True
