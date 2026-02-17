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
