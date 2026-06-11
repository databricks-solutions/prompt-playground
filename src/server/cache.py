"""Short-lived in-memory TTL caches for expensive workspace/MLflow calls."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_DEFAULT_TTL_SEC = 180


class TtlCache:
    """Simple thread-unsafe TTL cache keyed by string."""

    def __init__(self, ttl_sec: float = _DEFAULT_TTL_SEC):
        self._ttl_sec = ttl_sec
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts >= self._ttl_sec:
            del self._entries[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._entries[key] = (time.time(), value)

    def invalidate(self, key: Optional[str] = None) -> None:
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value
