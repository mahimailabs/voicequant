"""Audio conversion helpers for TTS outputs."""

from __future__ import annotations

import io
import wave
from array import array
from typing import Iterable


def _to_int16_bytes(samples: Iterable[float]) -> bytes:
    pcm = array("h")
    for s in samples:
        v = max(-1.0, min(1.0, float(s)))
        pcm.append(int(v * 32767.0))
    return pcm.tobytes()


def float32_to_wav(samples, sample_rate: int) -> bytes:
    """Convert float32 samples in [-1,1] to WAV bytes."""
    pcm_bytes = _to_int16_bytes(samples)
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buff.getvalue()


def float32_to_pcm(samples, sample_rate: int) -> bytes:
    """Convert float32 samples in [-1,1] to raw int16 PCM bytes."""
    _ = sample_rate
    return _to_int16_bytes(samples)


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to MP3 bytes (optional dependency)."""
    try:
        import lameenc
    except ImportError as exc:
        raise ImportError(
            "mp3 encoding requires lameenc. pip install lameenc"
        ) from exc

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        nchannels = wf.getnchannels()
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(nchannels)
    encoder.set_quality(2)
    return encoder.encode(pcm) + encoder.flush()


def wav_to_opus(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to Opus bytes (optional dependency)."""
    try:
        import opuslib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "opus encoding requires opuslib. pip install opuslib"
        ) from exc

    raise NotImplementedError("Opus conversion path is not implemented yet")


def get_audio_duration(audio_bytes: bytes, format: str, sample_rate: int) -> float:
    """Estimate duration in seconds from audio byte payload."""
    fmt = format.lower()
    if fmt == "wav":
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate else 0.0

    if fmt == "pcm":
        bytes_per_sample = 2
        samples = len(audio_bytes) / bytes_per_sample
        return samples / sample_rate if sample_rate else 0.0

    return 0.0
