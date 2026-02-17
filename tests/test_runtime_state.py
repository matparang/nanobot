from nanobot.runtime.state import state


def test_runtime_state_baseline_and_restore():
    previous = state.get_all_toggles()
    try:
        state.latent_reasoning_enabled = True
        state.mem0_enabled = True
        state.fractal_memory_enabled = True
        state.entangled_memory_enabled = True
        state.triune_memory_enabled = True
        state.heartbeat_enabled = True

        before_baseline = state.get_all_toggles()
        saved = state.set_baseline_mode()
        after_baseline = state.get_all_toggles()

        assert saved == before_baseline
        assert all(value is False for value in after_baseline.values())

        state.restore_toggles(saved)
        assert state.get_all_toggles() == before_baseline
    finally:
        state.restore_toggles(previous)


def test_runtime_state_enter_exit_baseline_tracking():
    previous = state.get_all_toggles()
    try:
        entered = state.enter_baseline_mode()
        assert state.baseline_active is True
        assert entered == previous
        assert all(value is False for value in state.get_all_toggles().values())

        state.register_suspended_service("heartbeat")
        assert "heartbeat" in state.suspended_services

        state.exit_baseline_mode(restore=True)
        assert state.baseline_active is False
        assert state.get_all_toggles() == previous
        assert state.suspended_services == set()
    finally:
        if state.baseline_active:
            state.exit_baseline_mode(restore=False)
        state.restore_toggles(previous)
