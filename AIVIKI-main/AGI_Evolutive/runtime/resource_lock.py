"""Lightweight resource locking used by the orchestrator."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator


class ResourceLockRegistry:
    def __init__(self) -> None:
        self._locks: Dict[str, threading.Lock] = {}
        self._global = threading.Lock()

    def _get_lock(self, resource_id: str) -> threading.Lock:
        with self._global:
            lock = self._locks.get(resource_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[resource_id] = lock
            return lock

    @contextmanager
    def acquire(self, resource_id: str) -> Iterator[None]:
        lock = self._get_lock(resource_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


__all__ = ["ResourceLockRegistry"]
