"""Structured compression metrics collector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompressionMetrics:
    seq_len: int = 0
    n_layers: int = 0
    n_heads: int = 0
    fp16_bytes: int = 0
    compressed_bytes: float = 0.0
    ratio: float = 0.0
    compress_time_ms: float = 0.0
    build_time_ms: float = 0.0


class MetricsCollector:
    """Accumulates compression metrics across calls."""

    def __init__(self):
        self._history: list[CompressionMetrics] = []

    def record(self, metrics: CompressionMetrics) -> None:
        self._history.append(metrics)

    @property
    def history(self) -> list[CompressionMetrics]:
        return list(self._history)

    @property
    def last(self) -> CompressionMetrics | None:
        return self._history[-1] if self._history else None

    def summary(self) -> dict[str, Any]:
        if not self._history:
            return {}
        ratios = [m.ratio for m in self._history]
        times = [m.compress_time_ms for m in self._history]
        return {
            "count": len(self._history),
            "avg_ratio": sum(ratios) / len(ratios),
            "avg_compress_ms": sum(times) / len(times),
        }
