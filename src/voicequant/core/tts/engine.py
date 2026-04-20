"""Kokoro ONNX-backed TTS engine with lazy loading."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from voicequant.core.protocol import CapacityReport, HealthStatus
from voicequant.core.tts.audio import float32_to_pcm, float32_to_wav, get_audio_duration, wav_to_mp3
from voicequant.core.tts.config import TTSConfig
from voicequant.core.tts.speaker_cache import SpeakerCache

DEFAULT_KOKORO_VOICES = [
    "af_heart",
    "af_alloy",
    "af_nova",
    "am_fable",
    "am_echo",
    "bf_ember",
    "bm_glass",
]


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str
    voice: str


class TTSEngine:
    """Kokoro TTS engine implementing the ModalityEngine protocol."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self._model: Any = None
        self._model_loaded = False
        self._speaker_cache: SpeakerCache | None = None
        self._load_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active = 0
        self._syntheses_total = 0
        self._latency_sum_ms = 0.0
        self._start_time = time.time()

    def load_model(self) -> None:
        """Load Kokoro model once."""
        with self._load_lock:
            if self._model_loaded:
                return

            import kokoro_onnx

            model_cls = getattr(kokoro_onnx, "Kokoro", None) or getattr(
                kokoro_onnx, "KokoroOnnx", None
            )
            if model_cls is None:
                raise ImportError("kokoro_onnx does not expose a Kokoro model class")

            self._model = model_cls(self.config.model_path, self.config.voices_path)
            self._speaker_cache = SpeakerCache(self.config.speaker_cache_size)
            self._model_loaded = True

    def _incr_active(self) -> None:
        with self._active_lock:
            self._active += 1

    def _decr_active(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)

    def _record_synthesis(self, elapsed_ms: float) -> None:
        with self._active_lock:
            self._latency_sum_ms += elapsed_ms
            self._syntheses_total += 1

    def _voice_payload(self, voice: str):
        if self._speaker_cache is None:
            self._speaker_cache = SpeakerCache(self.config.speaker_cache_size)
        cached = self._speaker_cache.get(voice)
        if cached is not None:
            return cached
        self._speaker_cache.put(voice, voice)
        return voice

    def _synthesize_samples(self, text: str, voice_payload: Any):
        if hasattr(self._model, "create"):
            out = self._model.create(text, voice_payload)
        else:
            raise RuntimeError("Unsupported kokoro_onnx model interface: missing create()")

        sample_rate = self.config.sample_rate
        samples = None

        if isinstance(out, dict):
            if "audio" in out and out["audio"] is not None:
                samples = out["audio"]
            elif "samples" in out and out["samples"] is not None:
                samples = out["samples"]
            elif "waveform" in out and out["waveform"] is not None:
                samples = out["waveform"]
            sample_rate = int(out.get("sample_rate", self.config.sample_rate))
        elif isinstance(out, tuple) and len(out) >= 2:
            samples, sample_rate = out[0], int(out[1])
        else:
            samples = out

        if samples is None:
            raise RuntimeError("TTS backend returned no samples")

        return samples, sample_rate

    def _encode_audio(self, samples, sample_rate: int, output_format: str) -> bytes:
        fmt = output_format.lower()
        if fmt == "wav":
            return float32_to_wav(samples, sample_rate)
        if fmt == "pcm":
            return float32_to_pcm(samples, sample_rate)
        if fmt == "mp3":
            wav = float32_to_wav(samples, sample_rate)
            return wav_to_mp3(wav)
        raise ValueError(f"Unsupported response format: {output_format}")

    @staticmethod
    def _duration_from_samples(samples, sample_rate: int) -> float:
        try:
            return len(samples) / sample_rate if sample_rate else 0.0
        except TypeError:
            return 0.0

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        output_format: str | None = None,
    ) -> SynthesisResult:
        """Synthesize text into audio bytes."""
        if len(text) > self.config.max_text_length:
            raise ValueError(
                f"Input text too long ({len(text)} > {self.config.max_text_length})"
            )

        if not self._model_loaded:
            self.load_model()

        selected_voice = voice or self.config.default_voice
        selected_format = (output_format or self.config.output_format).lower()

        self._incr_active()
        t0 = time.time()
        try:
            voice_payload = self._voice_payload(selected_voice)
            samples, sample_rate = self._synthesize_samples(text=text, voice_payload=voice_payload)
            precomputed_duration = self._duration_from_samples(samples, sample_rate)
            audio_bytes = self._encode_audio(samples, sample_rate, selected_format)
            if selected_format in {"wav", "pcm"}:
                duration_seconds = get_audio_duration(audio_bytes, selected_format, sample_rate)
            else:
                duration_seconds = precomputed_duration

            return SynthesisResult(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                format=selected_format,
                voice=selected_voice,
            )
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            self._record_synthesis(elapsed_ms)
            self._decr_active()

    def list_voices(self) -> list[str]:
        if self._model_loaded and hasattr(self._model, "get_voices"):
            voices = self._model.get_voices()
            if isinstance(voices, list):
                return [str(v) for v in voices]
        return DEFAULT_KOKORO_VOICES

    def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._model_loaded,
            modality="tts",
            detail=self.config.model_name if self._model_loaded else "not loaded",
        )

    def capacity(self) -> CapacityReport:
        with self._active_lock:
            active = self._active
            syntheses_total = self._syntheses_total
            latency_sum_ms = self._latency_sum_ms
        headroom = max(0, self.config.max_concurrent - active)
        saturated = active >= self.config.max_concurrent
        avg_ms = latency_sum_ms / syntheses_total if syntheses_total > 0 else 0.0
        return CapacityReport(
            active=active,
            headroom=headroom,
            saturated=saturated,
            latency_metric=avg_ms,
        )

    def metrics(self) -> dict[str, float]:
        with self._active_lock:
            active = self._active
            syntheses_total = self._syntheses_total
            latency_sum_ms = self._latency_sum_ms
        avg_ms = latency_sum_ms / syntheses_total if syntheses_total > 0 else 0.0
        cache_stats = self._speaker_cache.stats() if self._speaker_cache else {}
        return {
            "syntheses_total": float(syntheses_total),
            "avg_latency_ms": float(avg_ms),
            "active_sessions": float(active),
            "model_loaded": float(1 if self._model_loaded else 0),
            "speaker_cache_hit_rate": float(cache_stats.get("hit_rate", 0.0)),
            "speaker_cache_size": float(cache_stats.get("size", 0)),
            "uptime_seconds": float(time.time() - self._start_time),
        }

    def shutdown(self) -> None:
        with self._load_lock:
            self._model = None
            self._model_loaded = False
            if self._speaker_cache is not None:
                self._speaker_cache.clear()
                self._speaker_cache = None
