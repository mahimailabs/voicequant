"""Thread-safe speaker embedding LRU cache."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class SpeakerCache:
    """LRU cache for voice embeddings keyed by voice_id."""

    def __init__(self, max_size: int = 50) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get(self, voice_id: str) -> Any | None:
        with self._lock:
            if voice_id not in self._cache:
                self._misses += 1
                return None
            self._hits += 1
            self._cache.move_to_end(voice_id)
            return self._cache[voice_id]

    def put(self, voice_id: str, embedding: Any) -> None:
        with self._lock:
            if voice_id in self._cache:
                self._cache.move_to_end(voice_id)
            self._cache[voice_id] = embedding
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            requests = self._hits + self._misses
            hit_rate = self._hits / requests if requests else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }
