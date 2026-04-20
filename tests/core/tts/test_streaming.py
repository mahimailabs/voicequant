"""Tests for the StreamingSynthesizer (M5 Phase 1)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from voicequant.core.tts.engine import SynthesisResult
from voicequant.core.tts.streaming import (
    StreamingChunk,
    StreamingSynthesizer,
    TTSStreamingConfig,
)


def _fake_engine(n_samples: int = 24000, sample_rate: int = 24000) -> MagicMock:
    """Build a mock TTSEngine whose synthesize() returns fixed PCM bytes."""
    engine = MagicMock()
    samples = np.zeros(n_samples, dtype=np.float32)
    pcm = (samples * 32767.0).astype(np.int16).tobytes()
    engine.synthesize.return_value = SynthesisResult(
        audio_bytes=pcm,
        sample_rate=sample_rate,
        duration_seconds=n_samples / sample_rate,
        format="pcm",
        voice="af_heart",
    )
    # No stream_samples attribute: force non-genuine-streaming path.
    if hasattr(engine, "stream_samples"):
        del engine.stream_samples
    return engine


def test_tts_streaming_config_defaults():
    cfg = TTSStreamingConfig()
    assert cfg.chunk_size_samples == 4800
    assert cfg.min_chunk_size_samples == 1200
    assert cfg.output_format == "pcm"
    assert cfg.buffer_ahead_chunks == 2


def test_streaming_synthesizer_instantiates():
    engine = _fake_engine()
    synth = StreamingSynthesizer(engine)
    assert synth.engine is engine
    assert isinstance(synth.config, TTSStreamingConfig)


def test_stream_yields_streaming_chunks():
    engine = _fake_engine(n_samples=24000)
    synth = StreamingSynthesizer(engine, TTSStreamingConfig(output_format="pcm"))
    chunks = list(synth.stream("hello"))
    assert len(chunks) > 1
    assert all(isinstance(c, StreamingChunk) for c in chunks)
    # Required fields
    for c in chunks:
        assert isinstance(c.audio_bytes, bytes)
        assert isinstance(c.chunk_index, int)
        assert isinstance(c.is_final, bool)
        assert isinstance(c.timestamp_ms, float)
        assert isinstance(c.samples_count, int)
        assert isinstance(c.duration_ms, float)


def test_first_chunk_respects_min_chunk_size():
    engine = _fake_engine(n_samples=24000)
    cfg = TTSStreamingConfig(
        chunk_size_samples=4800, min_chunk_size_samples=1200, output_format="pcm"
    )
    synth = StreamingSynthesizer(engine, cfg)
    chunks = list(synth.stream("hi"))
    first = chunks[0]
    # PCM int16 -> 2 bytes per sample
    assert first.samples_count == 1200
    assert first.chunk_index == 0


def test_subsequent_chunks_use_chunk_size_samples():
    engine = _fake_engine(n_samples=24000)
    cfg = TTSStreamingConfig(
        chunk_size_samples=4800, min_chunk_size_samples=1200, output_format="pcm"
    )
    synth = StreamingSynthesizer(engine, cfg)
    chunks = list(synth.stream("hi"))
    # Second chunk should hit chunk_size (unless final)
    assert len(chunks) >= 2
    second = chunks[1]
    assert second.samples_count == 4800 or second.is_final


def test_final_chunk_has_is_final_true():
    engine = _fake_engine(n_samples=24000)
    synth = StreamingSynthesizer(engine, TTSStreamingConfig(output_format="pcm"))
    chunks = list(synth.stream("hi"))
    assert chunks[-1].is_final is True
    for c in chunks[:-1]:
        assert c.is_final is False


def test_ttfa_tracked_after_stream():
    engine = _fake_engine(n_samples=24000)
    synth = StreamingSynthesizer(engine, TTSStreamingConfig(output_format="pcm"))
    for _ in synth.stream("hi"):
        pass
    assert synth.last_ttfa_ms > 0.0
    assert synth.last_total_chunks > 0


def test_stream_uses_backend_stream_samples_when_available():
    """If the backend exposes stream_samples, StreamingSynthesizer uses it."""

    class TokenEngine:
        sample_rate = 24000

        def stream_samples(self, text, voice=None):
            # Yield 5 separate 1000-sample chunks.
            for _ in range(5):
                yield np.ones(1000, dtype=np.float32) * 0.1

        def synthesize(self, *a, **k):  # should not be called
            raise AssertionError("synthesize should not be called")

    synth = StreamingSynthesizer(
        TokenEngine(),
        TTSStreamingConfig(
            chunk_size_samples=2400, min_chunk_size_samples=1000, output_format="pcm"
        ),
    )
    chunks = list(synth.stream("hi"))
    assert len(chunks) >= 2
    assert chunks[-1].is_final is True
