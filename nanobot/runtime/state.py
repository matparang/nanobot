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


state = RuntimeState()
