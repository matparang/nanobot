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
        return cls._instance

    @property
    def latent_reasoning_enabled(self) -> bool:
        with self._lock:
            return self._latent_reasoning_enabled

    @latent_reasoning_enabled.setter
    def latent_reasoning_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._latent_reasoning_enabled = bool(enabled)


state = RuntimeState()
