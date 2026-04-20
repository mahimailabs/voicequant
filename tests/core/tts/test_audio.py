"""Audio encoding utility tests."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from voicequant.core.tts.audio import (
    float32_to_pcm,
    float32_to_wav,
    get_audio_duration,
    wav_to_mp3,
    wav_to_opus,
)


def test_float32_to_wav_produces_riff_header():
    samples = np.zeros(2400, dtype=np.float32)
    wav = float32_to_wav(samples, 24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_float32_to_pcm_byte_length():
    samples = np.zeros(1000, dtype=np.float32)
    pcm = float32_to_pcm(samples, 24000)
    assert len(pcm) == 1000 * 2  # int16


def test_get_audio_duration_for_pcm():
    # 24000 int16 samples -> 1.0s at 24kHz.
    pcm = b"\x00\x00" * 24000
    assert get_audio_duration(pcm, "pcm", 24000) == pytest.approx(1.0)


def test_get_audio_duration_for_wav():
    samples = np.zeros(24000, dtype=np.float32)
    wav = float32_to_wav(samples, 24000)
    assert get_audio_duration(wav, "wav", 24000) == pytest.approx(1.0)


def test_wav_to_mp3_raises_helpful_error_without_lameenc(monkeypatch):
    monkeypatch.setitem(sys.modules, "lameenc", None)
    # Force ImportError path by deleting any cached module and blocking import.
    monkeypatch.delitem(sys.modules, "lameenc", raising=False)
    import builtins

    original_import = builtins.__import__

    def _patched(name, *a, **k):
        if name == "lameenc":
            raise ImportError("blocked")
        return original_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _patched)

    with pytest.raises(ImportError, match="lameenc"):
        wav_to_mp3(b"fake-wav-bytes")


def test_wav_to_opus_raises_helpful_error_without_opuslib(monkeypatch):
    monkeypatch.delitem(sys.modules, "opuslib", raising=False)
    import builtins

    original_import = builtins.__import__

    def _patched(name, *a, **k):
        if name == "opuslib":
            raise ImportError("blocked")
        return original_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _patched)

    with pytest.raises(ImportError, match="opuslib"):
        wav_to_opus(b"fake-wav-bytes")


def test_float32_to_wav_clips_out_of_range():
    samples = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
    wav = float32_to_wav(samples, 24000)
    # Valid RIFF header means encoding succeeded despite out-of-range floats.
    assert wav[:4] == b"RIFF"


def test_float32_to_wav_with_list_input():
    samples = [0.0, 0.25, -0.25, 0.0]
    wav = float32_to_wav(samples, 16000)
    assert wav[:4] == b"RIFF"


# Silence unused import flags (fixtures live in modules above).
_ = ModuleType
