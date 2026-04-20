"""Streaming TTS synthesizer — chunked audio streaming with low TTFA.

Wraps a TTSEngine (or an Orpheus-like backend) and emits audio in chunks
so clients can begin playback before the full utterance is ready.

For Kokoro the underlying model generates the entire waveform in one
call; we split it into chunk_size_samples pieces to simulate streaming
(still valuable for client-side playback overlap). For Orpheus the
adapter streams tokens incrementally and we forward chunks as they
arrive.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from voicequant.core.tts.audio import float32_to_pcm, float32_to_wav


class TTSStreamingConfig(BaseModel):
    """Streaming knobs for chunked TTS output.

    chunk_size_samples at 24kHz: 4800 -> 200ms, 1200 -> 50ms.
    """

    chunk_size_samples: int = Field(default=4800)
    min_chunk_size_samples: int = Field(default=1200)
    output_format: str = Field(default="pcm")
    buffer_ahead_chunks: int = Field(default=2)


@dataclass
class StreamingChunk:
    audio_bytes: bytes
    chunk_index: int
    is_final: bool
    timestamp_ms: float
    samples_count: int
    duration_ms: float


def _encode_chunk(samples: Any, sample_rate: int, output_format: str) -> bytes:
    fmt = output_format.lower()
    if fmt == "pcm":
        return float32_to_pcm(samples, sample_rate)
    if fmt == "wav":
        return float32_to_wav(samples, sample_rate)
    raise ValueError(f"Unsupported streaming format: {output_format}")


class StreamingSynthesizer:
    """Compose a TTSEngine with chunked audio streaming.

    Does not replace TTSEngine. The engine does the heavy lifting; this
    class breaks its output into StreamingChunk objects and tracks TTFA.
    """

    def __init__(
        self,
        engine: Any,
        config: TTSStreamingConfig | None = None,
    ) -> None:
        self.engine = engine
        self.config = config or TTSStreamingConfig()
        self._last_ttfa_ms: float = 0.0
        self._last_total_chunks: int = 0
        self._last_total_ms: float = 0.0

    @property
    def last_ttfa_ms(self) -> float:
        return self._last_ttfa_ms

    @property
    def last_total_chunks(self) -> int:
        return self._last_total_chunks

    def stream(
        self,
        text: str,
        voice: str | None = None,
    ) -> Generator[StreamingChunk, None, None]:
        """Yield audio chunks for ``text``.

        Genuine streaming when the backend exposes a ``stream_samples``
        method (Orpheus). Otherwise falls back to synthesize() and chunks
        the result (Kokoro).
        """
        t0 = time.perf_counter()
        self._last_ttfa_ms = 0.0
        self._last_total_chunks = 0
        self._last_total_ms = 0.0

        backend_stream = getattr(self.engine, "stream_samples", None)
        if callable(backend_stream):
            sample_iter: Iterable[Any] = backend_stream(text, voice=voice)
            sample_rate = getattr(self.engine, "sample_rate", 24000)
            yield from self._emit_from_iterable(sample_iter, sample_rate, t0)
            return

        result = self.engine.synthesize(text, voice=voice, output_format="pcm")
        sample_rate = int(result.sample_rate)
        import numpy as np

        pcm = np.frombuffer(result.audio_bytes, dtype=np.int16)
        samples = pcm.astype(np.float32) / 32767.0
        yield from self._emit_from_array(samples, sample_rate, t0)

    def _emit_from_array(
        self,
        samples: Any,
        sample_rate: int,
        t0: float,
    ) -> Generator[StreamingChunk, None, None]:
        """Chunk a pre-generated float32 array into StreamingChunks."""
        import numpy as np

        arr = np.asarray(samples, dtype=np.float32)
        total = int(arr.shape[0])
        if total == 0:
            yield self._make_chunk(
                arr, sample_rate, 0, True, t0
            )
            return

        min_chunk = max(1, int(self.config.min_chunk_size_samples))
        step = max(1, int(self.config.chunk_size_samples))
        idx = 0
        start = 0
        first_emitted = False
        while start < total:
            end = start + (min_chunk if not first_emitted else step)
            end = min(end, total)
            is_final = end >= total
            chunk_samples = arr[start:end]
            chunk = self._make_chunk(
                chunk_samples, sample_rate, idx, is_final, t0
            )
            if not first_emitted:
                self._last_ttfa_ms = chunk.timestamp_ms
                first_emitted = True
            yield chunk
            idx += 1
            start = end

        self._last_total_chunks = idx
        self._last_total_ms = (time.perf_counter() - t0) * 1000.0

    def _emit_from_iterable(
        self,
        sample_iter: Iterable[Any],
        sample_rate: int,
        t0: float,
    ) -> Generator[StreamingChunk, None, None]:
        """Forward samples from a backend generator into StreamingChunks."""
        import numpy as np

        buffer: list[Any] = []
        buffered = 0
        idx = 0
        first_emitted = False
        min_chunk = max(1, int(self.config.min_chunk_size_samples))
        step = max(1, int(self.config.chunk_size_samples))

        def _flush(is_final: bool) -> StreamingChunk | None:
            nonlocal idx, first_emitted, buffered
            if buffered == 0 and not is_final:
                return None
            if buffer:
                arr = np.concatenate([np.asarray(x, dtype=np.float32) for x in buffer])
            else:
                arr = np.zeros(0, dtype=np.float32)
            buffer.clear()
            buffered = 0
            chunk = self._make_chunk(arr, sample_rate, idx, is_final, t0)
            if not first_emitted:
                self._last_ttfa_ms = chunk.timestamp_ms
                first_emitted = True
            idx += 1
            return chunk

        for piece in sample_iter:
            arr = np.asarray(piece, dtype=np.float32)
            if arr.size == 0:
                continue
            buffer.append(arr)
            buffered += int(arr.size)
            target = min_chunk if not first_emitted else step
            while buffered >= target:
                out = np.concatenate(
                    [np.asarray(x, dtype=np.float32) for x in buffer]
                )
                head = out[:target]
                tail = out[target:]
                buffer.clear()
                buffered = int(tail.size)
                if tail.size > 0:
                    buffer.append(tail)
                chunk = self._make_chunk(
                    head, sample_rate, idx, False, t0
                )
                if not first_emitted:
                    self._last_ttfa_ms = chunk.timestamp_ms
                    first_emitted = True
                idx += 1
                yield chunk
                target = step

        final = _flush(True)
        if final is not None:
            yield final

        self._last_total_chunks = idx
        self._last_total_ms = (time.perf_counter() - t0) * 1000.0

    def _make_chunk(
        self,
        samples: Any,
        sample_rate: int,
        chunk_index: int,
        is_final: bool,
        t0: float,
    ) -> StreamingChunk:
        audio_bytes = _encode_chunk(samples, sample_rate, self.config.output_format)
        samples_count = int(getattr(samples, "shape", [len(samples)])[0])
        duration_ms = (
            (samples_count / float(sample_rate)) * 1000.0 if sample_rate > 0 else 0.0
        )
        timestamp_ms = (time.perf_counter() - t0) * 1000.0
        return StreamingChunk(
            audio_bytes=audio_bytes,
            chunk_index=chunk_index,
            is_final=is_final,
            timestamp_ms=timestamp_ms,
            samples_count=samples_count,
            duration_ms=duration_ms,
        )
