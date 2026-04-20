"""TTSEngine — kokoro-onnx wrapper satisfying the ModalityEngine protocol.

Model loading is lazy: importing this module and instantiating TTSEngine does
not download the Kokoro model. The model loads on first synthesize() call or
an explicit load_model().
"""

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


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    format: str
    voice: str


# Built-in Kokoro voices.
KOKORO_VOICES: list[dict[str, str]] = [
    {"voice_id": "af_heart", "description": "American English, female (heart)"},
    {"voice_id": "af_bella", "description": "American English, female (bella)"},
    {"voice_id": "af_nicole", "description": "American English, female (nicole)"},
    {"voice_id": "af_sarah", "description": "American English, female (sarah)"},
    {"voice_id": "af_sky", "description": "American English, female (sky)"},
    {"voice_id": "am_adam", "description": "American English, male (adam)"},
    {"voice_id": "am_michael", "description": "American English, male (michael)"},
    {"voice_id": "bf_emma", "description": "British English, female (emma)"},
    {"voice_id": "bf_isabella", "description": "British English, female (isabella)"},
    {"voice_id": "bm_george", "description": "British English, male (george)"},
    {"voice_id": "bm_lewis", "description": "British English, male (lewis)"},
]


def _detect_backend(model_name: str) -> str:
    """Return 'orpheus' or 'kokoro' based on the model name."""
    name = (model_name or "").lower()
    if "orpheus" in name or "canopylabs" in name:
        return "orpheus"
    return "kokoro"


class TTSEngine:
    """kokoro-onnx-backed TTS engine with lazy model loading."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self._backend = _detect_backend(self.config.model_name)
        self._model: Any = None
        self._model_loaded = False
        self._speaker_cache: SpeakerCache | None = None
        self._orpheus: Any = None
        self._lock = threading.Lock()
        self._active = 0
        self._active_lock = threading.Lock()
        self._syntheses_total = 0
        self._latency_sum_ms = 0.0
        self._start_time = time.time()

    @property
    def backend(self) -> str:
        return self._backend

    def load_model(self) -> None:
        with self._lock:
            if self._model_loaded:
                return
            if self._backend == "orpheus":
                self._load_orpheus()
            else:
                self._load_kokoro()
            self._speaker_cache = SpeakerCache(self.config.speaker_cache_size)
            self._model_loaded = True

    def _load_kokoro(self) -> None:
        from kokoro_onnx import Kokoro

        if self.config.model_path:
            self._model = Kokoro(self.config.model_path)
        else:
            self._model = Kokoro()

    def _load_orpheus(self) -> None:
        try:
            from voicequant.core.tts.orpheus_adapter import (
                OrpheusAdapter,
                OrpheusConfig,
            )
        except ImportError as e:
            raise ImportError(
                "Orpheus backend module failed to import. "
                "Install with: pip install voicequant[tts-orpheus]"
            ) from e

        self._orpheus = OrpheusAdapter(
            OrpheusConfig(
                model_name=self.config.model_name,
                device=self.config.device,
                tq_bits=self.config.tq_bits,
                tq_enabled=self.config.tq_enabled,
                sample_rate=self.config.sample_rate,
            )
        )
        self._orpheus.load_model()
        self._model = self._orpheus

    def list_voices(self) -> list[dict[str, str]]:
        """Return available voice identifiers for the active backend."""
        if self._backend == "orpheus":
            if self._orpheus is not None:
                return self._orpheus.list_voices()
            from voicequant.core.tts.orpheus_adapter import ORPHEUS_VOICES

            return [dict(v) for v in ORPHEUS_VOICES]
        return list(KOKORO_VOICES)

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    def get_compression_stats(self) -> dict[str, Any] | None:
        """Return KV compression metrics for Orpheus, None for Kokoro."""
        if self._backend != "orpheus":
            return None
        if self._orpheus is None:
            return None
        return self._orpheus.get_compression_stats()

    def stream_samples(self, text: str, voice: str | None = None):
        """Yield native float32 sample arrays when the backend supports it."""
        if not self._model_loaded:
            self.load_model()
        if self._backend == "orpheus" and self._orpheus is not None:
            yield from self._orpheus.stream_samples(text, voice=voice)
            return
        # Kokoro does not stream natively; fall back to one-shot synthesize.
        result = self.synthesize(text, voice=voice, output_format="pcm")
        import numpy as np

        pcm = np.frombuffer(result.audio_bytes, dtype=np.int16)
        yield pcm.astype(np.float32) / 32767.0

    def _incr_active(self) -> None:
        with self._active_lock:
            self._active += 1

    def _decr_active(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)

    def _get_voice_embedding(self, voice_id: str) -> Any:
        """Fetch the speaker embedding, using the LRU cache."""
        assert self._speaker_cache is not None
        cached = self._speaker_cache.get(voice_id)
        if cached is not None:
            return cached
        embedding = self._load_voice_embedding(voice_id)
        self._speaker_cache.put(voice_id, embedding)
        return embedding

    def _load_voice_embedding(self, voice_id: str) -> Any:
        """Ask the Kokoro model for the voice embedding by id."""
        get_voice = getattr(self._model, "get_voice", None)
        if callable(get_voice):
            return get_voice(voice_id)
        return voice_id

    def _encode_audio(
        self, samples: Any, sample_rate: int, output_format: str
    ) -> tuple[bytes, str]:
        fmt = output_format.lower()
        if fmt == "wav":
            return float32_to_wav(samples, sample_rate), "wav"
        if fmt == "pcm":
            return float32_to_pcm(samples, sample_rate), "pcm"
        if fmt == "mp3":
            wav_bytes = float32_to_wav(samples, sample_rate)
            return wav_to_mp3(wav_bytes), "mp3"
        if fmt == "opus":
            wav_bytes = float32_to_wav(samples, sample_rate)
            return wav_to_opus(wav_bytes), "opus"
        raise ValueError(f"Unsupported output format: {output_format}")

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        output_format: str | None = None,
    ) -> SynthesisResult:
        if not self._model_loaded:
            self.load_model()

        voice_id = voice or self.config.default_voice
        fmt = (output_format or self.config.output_format).lower()

        self._incr_active()
        t0 = time.time()
        try:
            if self._backend == "orpheus" and self._orpheus is not None:
                result = self._orpheus.synthesize(
                    text, voice=voice_id, output_format=fmt
                )
                if not result.voice:
                    result = SynthesisResult(
                        audio_bytes=result.audio_bytes,
                        sample_rate=result.sample_rate,
                        duration_seconds=result.duration_seconds,
                        format=result.format,
                        voice=voice_id,
                    )
            else:
                embedding = self._get_voice_embedding(voice_id)
                samples, sample_rate = self._model.create(
                    text,
                    voice=embedding,
                    speed=1.0,
                    lang="en-us",
                )

                audio_bytes, resolved_fmt = self._encode_audio(
                    samples, int(sample_rate), fmt
                )
                duration = get_audio_duration(
                    audio_bytes, resolved_fmt, int(sample_rate)
                )
                result = SynthesisResult(
                    audio_bytes=audio_bytes,
                    sample_rate=int(sample_rate),
                    duration_seconds=duration,
                    format=resolved_fmt,
                    voice=voice_id,
                )
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            self._latency_sum_ms += elapsed_ms
            self._syntheses_total += 1
            self._decr_active()

        return result

    # --- ModalityEngine protocol ---

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
        cache_stats = (
            self._speaker_cache.stats()
            if self._speaker_cache is not None
            else {"hit_rate": 0.0, "size": 0}
        )
        return {
            "syntheses_total": float(self._syntheses_total),
            "avg_latency_ms": float(avg_ms),
            "active_sessions": float(self._active),
            "model_loaded": float(1 if self._model_loaded else 0),
            "speaker_cache_hit_rate": float(cache_stats["hit_rate"]),
            "speaker_cache_size": float(cache_stats["size"]),
            "uptime_seconds": float(time.time() - self._start_time),
        }

    def shutdown(self) -> None:
        with self._lock:
            if self._orpheus is not None:
                cleanup = getattr(self._orpheus, "shutdown", None) or getattr(
                    self._orpheus, "close", None
                )
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception:
                        pass
                self._orpheus = None
            self._model = None
            self._model_loaded = False
            if self._speaker_cache is not None:
                self._speaker_cache.clear()
                self._speaker_cache = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
