from __future__ import annotations

import pytest

from voicequant.core.tts.audio import (
    float32_to_pcm,
    float32_to_wav,
    get_audio_duration,
    wav_to_mp3,
    wav_to_opus,
)


def test_float32_to_wav_has_riff_header():
    samples = [0.0, 0.5, -0.5, 0.1]
    wav_bytes = float32_to_wav(samples, 24000)
    assert wav_bytes[:4] == b"RIFF"


def test_float32_to_pcm_byte_length():
    samples = [0.1] * 100
    pcm = float32_to_pcm(samples, 24000)
    assert len(pcm) == 200


def test_get_audio_duration_pcm():
    samples = [0.1] * 24000
    pcm = float32_to_pcm(samples, 24000)
    dur = get_audio_duration(pcm, "pcm", 24000)
    assert dur == pytest.approx(1.0, rel=1e-3)


def test_wav_to_mp3_importerror_message():
    with pytest.raises(ImportError, match="mp3 encoding requires lameenc"):
        wav_to_mp3(b"not-a-real-wav")


def test_wav_to_opus_importerror_message():
    with pytest.raises(ImportError, match="opus encoding requires opuslib"):
        wav_to_opus(b"not-a-real-wav")
