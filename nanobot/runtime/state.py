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
            }
            self._latent_reasoning_enabled = False
            self._mem0_enabled = False
            self._fractal_memory_enabled = False
            self._entangled_memory_enabled = False
            self._triune_memory_enabled = False
            self._heartbeat_enabled = False
            return previous

    def restore_toggles(self, states: dict[str, bool]) -> None:
        with self._lock:
            self._latent_reasoning_enabled = bool(states.get("latent_reasoning", False))
            self._mem0_enabled = bool(states.get("mem0", False))
            self._fractal_memory_enabled = bool(states.get("fractal_memory", False))
            self._entangled_memory_enabled = bool(states.get("entangled_memory", False))
            self._triune_memory_enabled = bool(states.get("triune_memory", True))
            self._heartbeat_enabled = bool(states.get("heartbeat", True))


state = RuntimeState()
