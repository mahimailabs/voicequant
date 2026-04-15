"""STTEngine — faster-whisper wrapper satisfying the ModalityEngine protocol.

Model loading is lazy: importing this module and instantiating STTEngine does
not download or load the Whisper model. The model loads on first transcribe()
call or an explicit load_model().
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from voicequant.core.protocol import CapacityReport, HealthStatus
from voicequant.core.stt.config import STTConfig


@dataclass
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: list[dict] = field(default_factory=list)


class STTEngine:
    """faster-whisper-backed STT engine with lazy model loading."""

    def __init__(self, config: STTConfig | None = None) -> None:
        self.config = config or STTConfig()
        self._model: Any = None
        self._model_loaded = False
        self._lock = threading.Lock()
        self._active = 0
        self._active_lock = threading.Lock()
        self._transcriptions_total = 0
        self._latency_sum_ms = 0.0
        self._start_time = time.time()

    def load_model(self) -> None:
        with self._lock:
            if self._model_loaded:
                return
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                model_size_or_path=self.config.model_name,
                device=self.config.device,
                device_index=self.config.device_index,
                compute_type=self.config.compute_type,
            )
            self._model_loaded = True

    def _incr_active(self) -> None:
        with self._active_lock:
            self._active += 1

    def _decr_active(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        response_format: str = "json",
    ) -> TranscriptionResult:
        if not self._model_loaded:
            self.load_model()

        self._incr_active()
        t0 = time.time()
        try:
            segments_iter, info = self._model.transcribe(
                audio_path,
                language=language or self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                condition_on_previous_text=self.config.condition_on_previous_text,
            )
            segments = []
            text_parts = []
            for i, seg in enumerate(segments_iter):
                segments.append(
                    {
                        "id": i,
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text,
                    }
                )
                text_parts.append(seg.text)

            result = TranscriptionResult(
                text="".join(text_parts).strip(),
                language=getattr(info, "language", "") or "",
                duration=float(getattr(info, "duration", 0.0)),
                segments=segments,
            )
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            self._latency_sum_ms += elapsed_ms
            self._transcriptions_total += 1
            self._decr_active()

        return result

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        response_format: str = "json",
        suffix: str = ".wav",
    ) -> TranscriptionResult:
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)
            return self.transcribe(
                path, language=language, response_format=response_format
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # --- ModalityEngine protocol ---

    def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._model_loaded,
            modality="stt",
            detail=self.config.model_name if self._model_loaded else "not loaded",
        )

    def capacity(self) -> CapacityReport:
        active = self._active
        headroom = max(0, self.config.max_concurrent - active)
        saturated = active >= self.config.max_concurrent
        avg_ms = (
            self._latency_sum_ms / self._transcriptions_total
            if self._transcriptions_total > 0
            else 0.0
        )
        return CapacityReport(
            active=active,
            headroom=headroom,
            saturated=saturated,
            latency_metric=avg_ms,
        )

    def metrics(self) -> dict[str, float]:
        avg_ms = (
            self._latency_sum_ms / self._transcriptions_total
            if self._transcriptions_total > 0
            else 0.0
        )
        return {
            "transcriptions_total": float(self._transcriptions_total),
            "avg_latency_ms": float(avg_ms),
            "active_sessions": float(self._active),
            "model_loaded": float(1 if self._model_loaded else 0),
            "uptime_seconds": float(time.time() - self._start_time),
        }

    def shutdown(self) -> None:
        with self._lock:
            self._model = None
            self._model_loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
