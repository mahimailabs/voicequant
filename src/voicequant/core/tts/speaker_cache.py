"""Thread-safe LRU cache for speaker embeddings."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class SpeakerCache:
    """LRU cache keyed by voice_id for precomputed speaker embeddings."""

    def __init__(self, max_size: int = 50) -> None:
        self.max_size = max_size
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, voice_id: str) -> Any | None:
        with self._lock:
            if voice_id in self._data:
                self._data.move_to_end(voice_id)
                self._hits += 1
                return self._data[voice_id]
            self._misses += 1
            return None

    def put(self, voice_id: str, embedding: Any) -> None:
        with self._lock:
            if voice_id in self._data:
                self._data.move_to_end(voice_id)
                self._data[voice_id] = embedding
                return
            if len(self._data) >= self.max_size:
                self._data.popitem(last=False)
            self._data[voice_id] = embedding

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }
