"""Audio format conversion utilities for TTS output.

wav and pcm use stdlib only. mp3 and opus are stretch goals that require
optional packages and raise a helpful ImportError when they are missing.
"""

from __future__ import annotations

import io
import wave
from typing import Any


def _to_int16(samples: Any):
    """Convert float32 samples (numpy array or list) to int16 numpy array."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16)


def float32_to_wav(samples: Any, sample_rate: int) -> bytes:
    """Encode float32 samples to WAV file bytes (16-bit PCM)."""
    int16 = _to_int16(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def float32_to_pcm(samples: Any, sample_rate: int) -> bytes:
    """Encode float32 samples to raw int16 PCM bytes (no header)."""
    del sample_rate  # rate is not encoded in raw PCM
    int16 = _to_int16(samples)
    return int16.tobytes()


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Encode WAV bytes as MP3. Requires lameenc."""
    try:
        import lameenc
    except ImportError as e:
        raise ImportError(
            "mp3 encoding requires lameenc. pip install lameenc"
        ) from e

    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(n_channels)
    encoder.set_quality(2)
    pcm = np.frombuffer(frames, dtype=np.int16)
    mp3 = encoder.encode(pcm.tobytes())
    mp3 += encoder.flush()
    return bytes(mp3)


def wav_to_opus(wav_bytes: bytes) -> bytes:
    """Encode WAV bytes as Opus. Requires opuslib."""
    try:
        import opuslib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "opus encoding requires opuslib. pip install opuslib"
        ) from e

    # opuslib gives you a codec; full container framing is out of scope here.
    # Decode WAV then pass raw PCM to opuslib. Callers needing a file container
    # should install a full encoder.
    import numpy as np
    from opuslib import Encoder

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    pcm = np.frombuffer(frames, dtype=np.int16)
    encoder = Encoder(sample_rate, n_channels, "audio")
    frame_size = sample_rate // 50  # 20ms frames
    out = bytearray()
    for i in range(0, len(pcm) - frame_size + 1, frame_size):
        chunk = pcm[i : i + frame_size].tobytes()
        out.extend(encoder.encode(chunk, frame_size))
    return bytes(out)


def get_audio_duration(audio_bytes: bytes, format: str, sample_rate: int) -> float:
    """Compute duration in seconds from audio bytes."""
    fmt = format.lower()
    if fmt == "wav":
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_frames = wf.getnframes()
            rate = wf.getframerate()
        return n_frames / float(rate) if rate > 0 else 0.0
    if fmt == "pcm":
        n_samples = len(audio_bytes) // 2  # int16
        return n_samples / float(sample_rate) if sample_rate > 0 else 0.0
    # Best-effort fallback: not meaningful for compressed formats.
    return 0.0
