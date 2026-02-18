"""Async-safe file locking utilities for concurrent memory updates."""

import asyncio
from pathlib import Path
from typing import Any


class AsyncFileLock:
    """
    Async-safe file locking to prevent races when multiple tasks update memory.

    Uses asyncio.Lock for async-safe locking within a single process.
    For multi-process scenarios, consider using fcntl (Unix) or msvcrt (Windows).
    """

    _locks: dict[str, asyncio.Lock] = {}
    _locks_lock: asyncio.Lock | None = None

    def __init__(self, file_path: Path | str):
        """
        Initialize lock for a specific file.

        Args:
            file_path: Path to the file to lock
        """
        self.file_path = Path(file_path)
        self.lock_key = str(self.file_path.resolve())

    async def __aenter__(self):
        """Acquire the lock."""
        # Initialize locks lock lazily on first use
        if self._locks_lock is None:
            AsyncFileLock._locks_lock = asyncio.Lock()

        # Get or create lock for this file
        async with self._locks_lock:
            if self.lock_key not in self._locks:
                self._locks[self.lock_key] = asyncio.Lock()
            lock = self._locks[self.lock_key]

        await lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the lock."""
        if self._locks_lock is not None:
            async with self._locks_lock:
                lock = self._locks.get(self.lock_key)

            if lock and lock.locked():
                lock.release()

        return False


async def locked_write(file_path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """
    Write to a file with async locking.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding (default: utf-8)
    """
    async with AsyncFileLock(file_path):
        # Run file I/O in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: Path(file_path).write_text(content, encoding=encoding)
        )


async def locked_append(file_path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """
    Append to a file with async locking.
    
    Args:
        file_path: Path to the file
        content: Content to append
        encoding: File encoding (default: utf-8)
    """
    async with AsyncFileLock(file_path):
        # Run file I/O in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _append_sync(file_path, content, encoding)
        )


def _append_sync(file_path: Path | str, content: str, encoding: str) -> None:
    """Synchronous append helper."""
    with open(file_path, "a", encoding=encoding) as f:
        f.write(content)
