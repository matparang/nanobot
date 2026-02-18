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
    previous_stages = state.get_context_stages()
    try:
        entered = state.enter_baseline_mode()
        assert state.baseline_active is True
        # entered dict includes toggles + context_stages
        assert entered["latent_reasoning"] == previous["latent_reasoning"]
        assert "context_stages" in entered
        assert all(value is False for value in state.get_all_toggles().values())

        state.register_suspended_service("heartbeat")
        assert "heartbeat" in state.suspended_services

        state.exit_baseline_mode(restore=True)
        assert state.baseline_active is False
        assert state.get_all_toggles() == previous
        assert state.get_context_stages() == previous_stages
        assert state.suspended_services == set()
    finally:
        if state.baseline_active:
            state.exit_baseline_mode(restore=False)
        state.restore_toggles(previous)
        # Restore context stages
        for stage_name, enabled in previous_stages.items():
            if enabled:
                state.enable_context_stage(stage_name)
            else:
                state.disable_context_stage(stage_name)


def test_runtime_state_reasoning_modes():
    previous = state.get_all_toggles()
    try:
        state.set_reasoning_mode("system1_only")
        assert state.light_reasoner_enabled is True
        assert state.latent_reasoning_enabled is False
        assert state.dual_layer_enabled is False

        state.set_reasoning_mode("system2_only")
        assert state.light_reasoner_enabled is False
        assert state.latent_reasoning_enabled is True
        assert state.dual_layer_enabled is False

        state.set_reasoning_mode("hybrid")
        assert state.light_reasoner_enabled is True
        assert state.latent_reasoning_enabled is True
        assert state.dual_layer_enabled is True
    finally:
        state.restore_toggles(previous)


def test_latent_enabled_gate_with_baseline():
    """Test that latent reasoning is disabled in baseline mode."""
    previous = state.get_all_toggles()
    previous_baseline = state.baseline_active
    try:
        # Test when latent is enabled
        state.latent_reasoning_enabled = True
        assert state.latent_reasoning_enabled is True
        
        # Test when latent is disabled
        state.latent_reasoning_enabled = False
        assert state.latent_reasoning_enabled is False
        
        # Test that baseline mode disables latent
        state.latent_reasoning_enabled = True
        assert state.latent_reasoning_enabled is True
        
        state.enter_baseline_mode()
        assert state.latent_reasoning_enabled is False, "Baseline mode should disable latent"
        assert state.baseline_active is True
        
        state.exit_baseline_mode(restore=True)
        assert state.latent_reasoning_enabled is True, "Should restore after baseline exit"
    finally:
        if state.baseline_active:
            state.exit_baseline_mode(restore=False)
        state.restore_toggles(previous)
