"""TTSEngine tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from voicequant.core.protocol import CapacityReport, HealthStatus, ModalityEngine
from voicequant.core.tts.config import TTSConfig
from voicequant.core.tts.engine import SynthesisResult, TTSEngine
from voicequant.core.tts.speaker_cache import SpeakerCache


def test_tts_engine_does_not_load_model_on_construction():
    engine = TTSEngine(TTSConfig(device="cpu"))
    assert engine._model is None
    assert engine._model_loaded is False


def test_tts_engine_health_when_unloaded():
    engine = TTSEngine(TTSConfig(device="cpu"))
    h = engine.health()
    assert isinstance(h, HealthStatus)
    assert h.modality == "tts"
    assert h.healthy is False
    assert h.detail == "not loaded"


def test_tts_engine_satisfies_protocol():
    engine: ModalityEngine = TTSEngine(TTSConfig(device="cpu"))
    h = engine.health()
    assert isinstance(h, HealthStatus)
    c = engine.capacity()
    assert isinstance(c, CapacityReport)
    assert c.active == 0
    assert c.saturated is False
    m = engine.metrics()
    assert set(m.keys()) >= {
        "syntheses_total",
        "avg_latency_ms",
        "active_sessions",
        "model_loaded",
        "speaker_cache_hit_rate",
        "speaker_cache_size",
    }
    assert m["model_loaded"] == 0.0
    assert engine.shutdown() is None


def test_synthesis_result_dataclass_fields():
    r = SynthesisResult(
        audio_bytes=b"abc",
        sample_rate=24000,
        duration_seconds=0.5,
        format="wav",
        voice="af_heart",
    )
    assert r.audio_bytes == b"abc"
    assert r.sample_rate == 24000
    assert r.duration_seconds == 0.5
    assert r.format == "wav"
    assert r.voice == "af_heart"


def _install_fake_model(engine: TTSEngine) -> MagicMock:
    """Bypass load_model and inject a mock Kokoro-like model."""
    model = MagicMock()
    samples = np.zeros(2400, dtype=np.float32)  # 0.1s at 24kHz
    model.create.return_value = (samples, 24000)
    model.get_voice = MagicMock(side_effect=lambda vid: f"emb-{vid}")
    engine._model = model
    engine._model_loaded = True
    engine._speaker_cache = SpeakerCache(engine.config.speaker_cache_size)
    return model


def test_synthesize_returns_synthesis_result_with_wav():
    engine = TTSEngine(TTSConfig(device="cpu"))
    _install_fake_model(engine)

    result = engine.synthesize("hello", voice="af_heart", output_format="wav")

    assert isinstance(result, SynthesisResult)
    assert result.voice == "af_heart"
    assert result.format == "wav"
    assert result.sample_rate == 24000
    assert result.audio_bytes[:4] == b"RIFF"
    assert result.duration_seconds > 0


def test_synthesize_uses_speaker_cache():
    engine = TTSEngine(TTSConfig(device="cpu"))
    model = _install_fake_model(engine)

    engine.synthesize("hello", voice="af_heart")
    engine.synthesize("world", voice="af_heart")

    # get_voice should be called exactly once: second call hit the cache.
    assert model.get_voice.call_count == 1
    stats = engine._speaker_cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_synthesize_pcm_output():
    engine = TTSEngine(TTSConfig(device="cpu"))
    _install_fake_model(engine)

    result = engine.synthesize("hello", output_format="pcm")

    assert result.format == "pcm"
    # 2400 samples * 2 bytes per int16 sample.
    assert len(result.audio_bytes) == 2400 * 2


def test_metrics_update_after_synthesis():
    engine = TTSEngine(TTSConfig(device="cpu"))
    _install_fake_model(engine)
    engine.synthesize("hello")

    m = engine.metrics()
    assert m["syntheses_total"] == 1.0
    assert m["model_loaded"] == 1.0
    assert m["avg_latency_ms"] >= 0


def test_shutdown_unloads_model():
    engine = TTSEngine(TTSConfig(device="cpu"))
    _install_fake_model(engine)
    engine.shutdown()
    assert engine._model is None
    assert engine._model_loaded is False
    assert engine._speaker_cache is None
