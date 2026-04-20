"""Kokoro ONNX-backed TTS engine with lazy loading."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from voicequant.core.protocol import CapacityReport, HealthStatus
from voicequant.core.tts.audio import (
    float32_to_pcm,
    float32_to_wav,
    get_audio_duration,
    wav_to_mp3,
    wav_to_opus,
)
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
        self._active = 0
        self._active_lock = threading.Lock()
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

            kwargs = {"device": self.config.device}
            if self.config.model_path:
                kwargs["model_path"] = self.config.model_path
            self._model = model_cls(**kwargs)
            self._speaker_cache = SpeakerCache(self.config.speaker_cache_size)
            self._model_loaded = True

    def _incr_active(self) -> None:
        with self._active_lock:
            self._active += 1

    def _decr_active(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)

    def _get_or_create_speaker_embedding(self, voice: str) -> Any:
        if self._speaker_cache is None:
            self._speaker_cache = SpeakerCache(self.config.speaker_cache_size)

        cached = self._speaker_cache.get(voice)
        if cached is not None:
            return cached

        embedding: Any = voice
        if hasattr(self._model, "get_speaker_embedding"):
            embedding = self._model.get_speaker_embedding(voice)
        elif hasattr(self._model, "load_voice"):
            embedding = self._model.load_voice(voice)

        self._speaker_cache.put(voice, embedding)
        return embedding

    def _synthesize_samples(self, text: str, voice: str, speaker_embedding: Any):
        if hasattr(self._model, "synthesize"):
            out = self._model.synthesize(text=text, voice=voice, speaker=speaker_embedding)
        elif hasattr(self._model, "generate"):
            out = self._model.generate(text=text, voice=voice, speaker=speaker_embedding)
        elif callable(self._model):
            out = self._model(text=text, voice=voice, speaker=speaker_embedding)
        else:
            raise RuntimeError("Unsupported kokoro_onnx model interface")

        if isinstance(out, dict):
            samples = out.get("audio") or out.get("samples") or out.get("waveform")
            sample_rate = int(out.get("sample_rate", self.config.sample_rate))
        elif isinstance(out, tuple) and len(out) >= 2:
            samples, sample_rate = out[0], int(out[1])
        else:
            samples, sample_rate = out, self.config.sample_rate

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
        if fmt == "opus":
            wav = float32_to_wav(samples, sample_rate)
            return wav_to_opus(wav)
        raise ValueError(f"Unsupported response format: {output_format}")

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
            speaker_embedding = self._get_or_create_speaker_embedding(selected_voice)
            samples, sample_rate = self._synthesize_samples(
                text=text,
                voice=selected_voice,
                speaker_embedding=speaker_embedding,
            )
            audio_bytes = self._encode_audio(samples, sample_rate, selected_format)
            duration = get_audio_duration(audio_bytes, selected_format, sample_rate)
            return SynthesisResult(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                duration_seconds=duration,
                format=selected_format,
                voice=selected_voice,
            )
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            self._latency_sum_ms += elapsed_ms
            self._syntheses_total += 1
            self._decr_active()

    def list_voices(self) -> list[str]:
        if self._model_loaded and hasattr(self._model, "list_voices"):
            voices = self._model.list_voices()
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
        active = self._active
        headroom = max(0, self.config.max_concurrent - active)
        saturated = active >= self.config.max_concurrent
        avg_ms = (
            self._latency_sum_ms / self._syntheses_total
            if self._syntheses_total > 0
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
            self._latency_sum_ms / self._syntheses_total
            if self._syntheses_total > 0
            else 0.0
        )
        cache_stats = self._speaker_cache.stats() if self._speaker_cache else {}
        return {
            "syntheses_total": float(self._syntheses_total),
            "avg_latency_ms": float(avg_ms),
            "active_sessions": float(self._active),
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
