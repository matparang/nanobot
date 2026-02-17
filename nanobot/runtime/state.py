"""Global runtime state for Nanobot."""

from threading import Lock


class RuntimeState:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RuntimeState, cls).__new__(cls)
                cls._instance._latent_reasoning_enabled = False
                cls._instance._mem0_enabled = False
                cls._instance._fractal_memory_enabled = False
                cls._instance._entangled_memory_enabled = False
                cls._instance._triune_memory_enabled = True
                cls._instance._heartbeat_enabled = True
                cls._instance._light_reasoner_enabled = True
                cls._instance._dual_layer_enabled = True
                cls._instance._chi_tracking_enabled = True
                cls._instance._reasoning_audit_enabled = True
                cls._instance._baseline_active = False
                cls._instance._pre_baseline_state = None
                cls._instance._suspended_services = set()
        return cls._instance

    @property
    def latent_reasoning_enabled(self) -> bool:
        with self._lock:
            return self._latent_reasoning_enabled

    @latent_reasoning_enabled.setter
    def latent_reasoning_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._latent_reasoning_enabled = bool(enabled)

    @property
    def mem0_enabled(self) -> bool:
        with self._lock:
            return self._mem0_enabled

    @mem0_enabled.setter
    def mem0_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._mem0_enabled = bool(enabled)

    @property
    def fractal_memory_enabled(self) -> bool:
        with self._lock:
            return self._fractal_memory_enabled

    @fractal_memory_enabled.setter
    def fractal_memory_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._fractal_memory_enabled = bool(enabled)

    @property
    def entangled_memory_enabled(self) -> bool:
        with self._lock:
            return self._entangled_memory_enabled

    @entangled_memory_enabled.setter
    def entangled_memory_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._entangled_memory_enabled = bool(enabled)

    @property
    def triune_memory_enabled(self) -> bool:
        with self._lock:
            return self._triune_memory_enabled

    @triune_memory_enabled.setter
    def triune_memory_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._triune_memory_enabled = bool(enabled)

    @property
    def heartbeat_enabled(self) -> bool:
        with self._lock:
            return self._heartbeat_enabled

    @heartbeat_enabled.setter
    def heartbeat_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._heartbeat_enabled = bool(enabled)

    @property
    def light_reasoner_enabled(self) -> bool:
        with self._lock:
            return self._light_reasoner_enabled

    @light_reasoner_enabled.setter
    def light_reasoner_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._light_reasoner_enabled = bool(enabled)

    @property
    def dual_layer_enabled(self) -> bool:
        with self._lock:
            return self._dual_layer_enabled

    @dual_layer_enabled.setter
    def dual_layer_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._dual_layer_enabled = bool(enabled)

    @property
    def chi_tracking_enabled(self) -> bool:
        with self._lock:
            return self._chi_tracking_enabled

    @chi_tracking_enabled.setter
    def chi_tracking_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._chi_tracking_enabled = bool(enabled)

    @property
    def reasoning_audit_enabled(self) -> bool:
        with self._lock:
            return self._reasoning_audit_enabled

    @reasoning_audit_enabled.setter
    def reasoning_audit_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._reasoning_audit_enabled = bool(enabled)

    def get_all_toggles(self) -> dict[str, bool]:
        """Return all runtime toggle values keyed by stable CLI-friendly names."""
        with self._lock:
            return {
                "latent_reasoning": self._latent_reasoning_enabled,
                "mem0": self._mem0_enabled,
                "fractal_memory": self._fractal_memory_enabled,
                "entangled_memory": self._entangled_memory_enabled,
                "triune_memory": self._triune_memory_enabled,
                "heartbeat": self._heartbeat_enabled,
                "light_reasoner": self._light_reasoner_enabled,
                "dual_layer": self._dual_layer_enabled,
                "chi_tracking": self._chi_tracking_enabled,
                "reasoning_audit": self._reasoning_audit_enabled,
            }

    def set_baseline_mode(self) -> dict[str, bool]:
        with self._lock:
            previous = {
                "latent_reasoning": self._latent_reasoning_enabled,
                "mem0": self._mem0_enabled,
                "fractal_memory": self._fractal_memory_enabled,
                "entangled_memory": self._entangled_memory_enabled,
                "triune_memory": self._triune_memory_enabled,
                "heartbeat": self._heartbeat_enabled,
                "light_reasoner": self._light_reasoner_enabled,
                "dual_layer": self._dual_layer_enabled,
                "chi_tracking": self._chi_tracking_enabled,
                "reasoning_audit": self._reasoning_audit_enabled,
            }
            self._latent_reasoning_enabled = False
            self._mem0_enabled = False
            self._fractal_memory_enabled = False
            self._entangled_memory_enabled = False
            self._triune_memory_enabled = False
            self._heartbeat_enabled = False
            self._light_reasoner_enabled = False
            self._dual_layer_enabled = False
            self._chi_tracking_enabled = False
            self._reasoning_audit_enabled = False
            return previous

    @property
    def baseline_active(self) -> bool:
        with self._lock:
            return self._baseline_active

    @property
    def suspended_services(self) -> set[str]:
        with self._lock:
            return set(self._suspended_services)

    def enter_baseline_mode(self) -> dict[str, bool]:
        with self._lock:
            if self._baseline_active:
                raise RuntimeError("Already in baseline mode")
            self._pre_baseline_state = {
                "latent_reasoning": self._latent_reasoning_enabled,
                "mem0": self._mem0_enabled,
                "fractal_memory": self._fractal_memory_enabled,
                "entangled_memory": self._entangled_memory_enabled,
                "triune_memory": self._triune_memory_enabled,
                "heartbeat": self._heartbeat_enabled,
                "light_reasoner": self._light_reasoner_enabled,
                "dual_layer": self._dual_layer_enabled,
                "chi_tracking": self._chi_tracking_enabled,
                "reasoning_audit": self._reasoning_audit_enabled,
            }
            self._latent_reasoning_enabled = False
            self._mem0_enabled = False
            self._fractal_memory_enabled = False
            self._entangled_memory_enabled = False
            self._triune_memory_enabled = False
            self._heartbeat_enabled = False
            self._light_reasoner_enabled = False
            self._dual_layer_enabled = False
            self._chi_tracking_enabled = False
            self._reasoning_audit_enabled = False
            self._baseline_active = True
            return dict(self._pre_baseline_state)

    def exit_baseline_mode(self, restore: bool = True) -> None:
        with self._lock:
            if not self._baseline_active:
                raise RuntimeError("Not in baseline mode")
            previous = self._pre_baseline_state
            self._baseline_active = False
            self._pre_baseline_state = None
            self._suspended_services.clear()
            if restore and previous:
                self._latent_reasoning_enabled = bool(previous.get("latent_reasoning", False))
                self._mem0_enabled = bool(previous.get("mem0", False))
                self._fractal_memory_enabled = bool(previous.get("fractal_memory", False))
                self._entangled_memory_enabled = bool(previous.get("entangled_memory", False))
                self._triune_memory_enabled = bool(previous.get("triune_memory", True))
                self._heartbeat_enabled = bool(previous.get("heartbeat", True))
                self._light_reasoner_enabled = bool(previous.get("light_reasoner", True))
                self._dual_layer_enabled = bool(previous.get("dual_layer", True))
                self._chi_tracking_enabled = bool(previous.get("chi_tracking", True))
                self._reasoning_audit_enabled = bool(previous.get("reasoning_audit", True))

    def register_suspended_service(self, name: str) -> None:
        with self._lock:
            self._suspended_services.add(name)

    def unregister_suspended_service(self, name: str) -> None:
        with self._lock:
            self._suspended_services.discard(name)

    def restore_toggles(self, states: dict[str, bool]) -> None:
        with self._lock:
            self._latent_reasoning_enabled = bool(states.get("latent_reasoning", False))
            self._mem0_enabled = bool(states.get("mem0", False))
            self._fractal_memory_enabled = bool(states.get("fractal_memory", False))
            self._entangled_memory_enabled = bool(states.get("entangled_memory", False))
            self._triune_memory_enabled = bool(states.get("triune_memory", True))
            self._heartbeat_enabled = bool(states.get("heartbeat", True))
            self._light_reasoner_enabled = bool(states.get("light_reasoner", True))
            self._dual_layer_enabled = bool(states.get("dual_layer", True))
            self._chi_tracking_enabled = bool(states.get("chi_tracking", True))
            self._reasoning_audit_enabled = bool(states.get("reasoning_audit", True))

    def set_reasoning_mode(self, mode: str) -> dict[str, bool]:
        """Set predefined reasoning modes."""
        with self._lock:
            previous = {
                "latent_reasoning": self._latent_reasoning_enabled,
                "light_reasoner": self._light_reasoner_enabled,
                "dual_layer": self._dual_layer_enabled,
                "chi_tracking": self._chi_tracking_enabled,
                "reasoning_audit": self._reasoning_audit_enabled,
            }
            if mode == "system1_only":
                self._light_reasoner_enabled = True
                self._latent_reasoning_enabled = False
                self._dual_layer_enabled = False
            elif mode == "system2_only":
                self._light_reasoner_enabled = False
                self._latent_reasoning_enabled = True
                self._dual_layer_enabled = False
            elif mode == "hybrid":
                self._light_reasoner_enabled = True
                self._latent_reasoning_enabled = True
                self._dual_layer_enabled = True
            elif mode == "baseline":
                self._light_reasoner_enabled = False
                self._latent_reasoning_enabled = False
                self._dual_layer_enabled = False
            else:
                raise ValueError(f"Unknown reasoning mode: {mode}")
            return previous


state = RuntimeState()
