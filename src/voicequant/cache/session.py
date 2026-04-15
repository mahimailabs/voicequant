"""Per-session compressed KV cache handle."""

from __future__ import annotations

import torch
from typing import Any

from ..core.engine import TurboQuantEngine


class CacheSession:
    """Holds compressed KV state for one conversation session."""

    def __init__(self, engine: TurboQuantEngine):
        self._engine = engine
        self._compressed: dict[str, Any] | None = None
        self._seq_len: int = 0

    @property
    def engine(self) -> TurboQuantEngine:
        return self._engine

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def compressed(self) -> dict[str, Any] | None:
        return self._compressed

    @torch.no_grad()
    def compress(self, past_key_values) -> None:
        """Compress full KV cache from a model forward pass."""
        self._compressed = self._engine.compress_kv_cache(past_key_values)
        if self._compressed and self._compressed.get("layers"):
            ck = self._compressed["layers"][0][0][0]
            self._seq_len = ck["indices"].shape[0]

    def truncate(self, seq_len: int) -> None:
        """Truncate compressed cache to first seq_len positions."""
        if self._compressed is not None:
            self._compressed = self._engine.truncate_cache(
                self._compressed, seq_len
            )
        self._seq_len = min(seq_len, self._seq_len)

    def build(self):
        """Build a HuggingFace DynamicCache from compressed state."""
        if self._compressed is None:
            raise ValueError(
                "No compressed cache to build from. Call compress() first."
            )
        return self._engine.build_cache(self._compressed)

    def stats(self, past_key_values=None) -> dict:
        """Return compression statistics."""
        if past_key_values is not None:
            return self._engine.compression_stats(past_key_values)
        return self._engine.compressed_size_bytes(self._seq_len)

    def clear(self) -> None:
        """Release compressed state."""
        self._compressed = None
        self._seq_len = 0
