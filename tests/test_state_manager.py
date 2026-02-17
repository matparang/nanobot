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


def test_check_persistent_state_supports_triune_and_soul(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    data_dir = tmp_path / "data"
    (workspace / ".triune").mkdir(parents=True)
    (workspace / ".triune" / "checksums.json").write_text("{}", encoding="utf-8")
    (workspace / "SOUL.md").write_text("# soul", encoding="utf-8")

    monkeypatch.setattr(state_manager, "get_workspace_path", lambda: workspace)
    monkeypatch.setattr(state_manager, "get_data_dir", lambda: data_dir)

    triune = state_manager.check_persistent_state("triune")
    assert list(triune) == ["triune_checksums"]
    assert triune["triune_checksums"]["exists"] is True

    soul = state_manager.check_persistent_state("soul")
    assert list(soul) == ["soul_config"]
    assert soul["soul_config"]["exists"] is True
