"""Engine contract shared by all VoiceQuant modality engines (LLM, STT, TTS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class HealthStatus:
    healthy: bool
    modality: str
    detail: str | None = None


@dataclass
class CapacityReport:
    active: int
    headroom: int
    saturated: bool
    latency_metric: float


class ModalityEngine(Protocol):
    def health(self) -> HealthStatus: ...
    def capacity(self) -> CapacityReport: ...
    def metrics(self) -> dict[str, float]: ...
    def shutdown(self) -> None: ...
