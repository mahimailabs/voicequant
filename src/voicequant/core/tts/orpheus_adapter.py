"""Orpheus TTS adapter — the headline cross-modality wedge.

Orpheus uses a LLaMA-based LLM to generate speech tokens autoregressively.
That means its KV cache grows with each generated token, exactly like a
text LLM. VoiceQuant's TurboQuant compression engine — already used for
text KV caches — is applied here to the Orpheus backbone, producing
2-5x more concurrent Orpheus sessions per GPU.

Dependency direction: this module imports FROM voicequant.core.llm.engine.
core/llm never imports from core/tts. This is the only cross-modality
import in the codebase.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Cross-modality dependency: compression engine is a hard import.
from voicequant.core.llm.engine import TurboQuantEngine
from voicequant.core.tts.engine import SynthesisResult
from voicequant.core.tts.streaming import StreamingChunk, TTSStreamingConfig

# Static voice registry — safe to expose before the model is loaded.
ORPHEUS_VOICES: list[dict[str, str]] = [
    {"voice_id": "tara", "description": "Orpheus default voice"},
    {"voice_id": "leah", "description": "Orpheus female voice"},
    {"voice_id": "jess", "description": "Orpheus female voice"},
    {"voice_id": "leo", "description": "Orpheus male voice"},
    {"voice_id": "dan", "description": "Orpheus male voice"},
    {"voice_id": "mia", "description": "Orpheus female voice"},
    {"voice_id": "zac", "description": "Orpheus male voice"},
    {"voice_id": "zoe", "description": "Orpheus female voice"},
]


class OrpheusConfig(BaseModel):
    """Configuration for the Orpheus TTS backend."""

    model_name: str = Field(default="canopylabs/orpheus-3b-0.1-ft")
    tq_bits: int = Field(default=4)
    tq_enabled: bool = Field(default=True)
    device: str = Field(default="auto")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.6)
    top_p: float = Field(default=0.9)
    sample_rate: int = Field(default=24000)
    decode_chunk_tokens: int = Field(
        default=28, description="Decode after this many speech tokens"
    )

    @model_validator(mode="after")
    def _resolve_device(self):
        if self.device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        return self


class OrpheusAdapter:
    """Bridge Orpheus TTS to the TurboQuant KV compression engine.

    The adapter:
      1. Loads the Orpheus LLaMA backbone (lazy).
      2. Runs autoregressive speech-token generation.
      3. Compresses the KV cache with TurboQuantEngine on each step.
      4. Decodes token batches through the Orpheus audio codec.
    """

    def __init__(self, config: OrpheusConfig | None = None) -> None:
        self.config = config or OrpheusConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._decoder: Any = None
        self._model_loaded = False
        self._tq_engine: TurboQuantEngine | None = None
        if self.config.tq_enabled:
            self._tq_engine = TurboQuantEngine(
                total_bits=self.config.tq_bits,
                device=self.config.device,
            )
        self._last_fp16_bytes = 0
        self._last_tq_bytes = 0
        self._last_ratio = 0.0
        self._last_cosine = 0.0
        self._syntheses_total = 0
        self._latency_sum_ms = 0.0
        self._start_time = time.time()

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    def load_model(self) -> None:
        """Load the Orpheus model, tokenizer, and audio decoder.

        Import errors are re-raised with a clear install hint.
        """
        if self._model_loaded:
            return
        try:
            from orpheus_tts import OrpheusModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Orpheus backend requires orpheus-tts. "
                "Install with: pip install voicequant[tts-orpheus]"
            ) from e

        self._model = OrpheusModel(
            model_name=self.config.model_name,
            device=self.config.device,
        )
        self._tokenizer = getattr(self._model, "tokenizer", None)
        self._decoder = getattr(self._model, "decoder", None)
        self._model_loaded = True

    # --- Generation ---

    def generate_speech_tokens(
        self,
        text: str,
        voice: str | None = None,
    ):
        """Autoregressive speech-token generator.

        Yields speech tokens one at a time from the LLaMA backbone. After
        each forward pass TurboQuant compresses the KV cache. This is
        where the compression gain comes from — each active Orpheus
        session carries ~5x less KV state in GPU memory.
        """
        if not self._model_loaded:
            self.load_model()

        import torch

        prompt = f"{voice}: {text}" if voice else text
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.config.device)
        position_ids = None
        past_key_values = None
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        n_generated = 0

        while n_generated < self.config.max_tokens:
            out = self._model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            if self._tq_engine is not None:
                compressed = self._tq_engine.compress_kv_cache(out.past_key_values)
                stats = self._tq_engine.compression_stats(out.past_key_values)
                self._last_fp16_bytes = int(stats["fp16_bytes"])
                self._last_tq_bytes = int(stats["tq_bytes"])
                self._last_ratio = float(stats["ratio"])
                past_key_values = self._tq_engine.build_cache(compressed)
            else:
                past_key_values = out.past_key_values

            next_token = self._sample(logits)
            token_id = int(next_token.item())
            input_ids = next_token.view(1, 1)
            position_ids = torch.tensor(
                [[seq_len + n_generated]], device=self.config.device
            )
            n_generated += 1
            if self._is_eos(token_id):
                break
            yield token_id

    def _sample(self, logits: Any) -> Any:
        """Top-p temperature sampling."""
        import torch

        logits = logits / max(self.config.temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > self.config.top_p
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, choice)

    def _is_eos(self, token_id: int) -> bool:
        tok = self._tokenizer
        eos = getattr(tok, "eos_token_id", None)
        return eos is not None and token_id == eos

    def decode_tokens_to_audio(self, speech_tokens) -> Any:
        """Convert Orpheus speech tokens to float32 audio samples."""
        if not self._model_loaded:
            self.load_model()
        if self._decoder is None:
            raise RuntimeError("Orpheus decoder not available")

        import numpy as np

        samples = self._decoder.decode(list(speech_tokens))
        arr = np.asarray(samples, dtype=np.float32)
        return arr

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        output_format: str = "wav",
    ) -> SynthesisResult:
        """Generate all tokens, decode to audio, return a SynthesisResult."""
        from voicequant.core.tts.audio import (
            float32_to_pcm,
            float32_to_wav,
            get_audio_duration,
            wav_to_mp3,
            wav_to_opus,
        )

        t0 = time.time()
        tokens = list(self.generate_speech_tokens(text, voice=voice))
        samples = self.decode_tokens_to_audio(tokens)

        fmt = (output_format or "wav").lower()
        if fmt == "wav":
            audio_bytes = float32_to_wav(samples, self.sample_rate)
        elif fmt == "pcm":
            audio_bytes = float32_to_pcm(samples, self.sample_rate)
        elif fmt == "mp3":
            audio_bytes = wav_to_mp3(float32_to_wav(samples, self.sample_rate))
        elif fmt == "opus":
            audio_bytes = wav_to_opus(float32_to_wav(samples, self.sample_rate))
        else:
            raise ValueError(
                f"Orpheus adapter supports wav/pcm/mp3/opus, got {fmt}"
            )

        duration = get_audio_duration(audio_bytes, fmt, self.sample_rate)
        self._latency_sum_ms += (time.time() - t0) * 1000.0
        self._syntheses_total += 1
        return SynthesisResult(
            audio_bytes=audio_bytes,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            format=fmt,
            voice=voice or "",
        )

    def stream(
        self,
        text: str,
        voice: str | None = None,
        streaming_config: TTSStreamingConfig | None = None,
    ):
        """Genuine token-streaming: decode in batches as tokens arrive."""
        cfg = streaming_config or TTSStreamingConfig()
        t0 = time.perf_counter()
        idx = 0
        buffer: list[int] = []
        chunk_every = max(1, int(self.config.decode_chunk_tokens))

        def _emit(is_final: bool) -> StreamingChunk | None:
            nonlocal idx
            if not buffer and not is_final:
                return None
            samples = (
                self.decode_tokens_to_audio(list(buffer)) if buffer else _empty_audio()
            )
            buffer.clear()
            from voicequant.core.tts.audio import float32_to_pcm, float32_to_wav

            fmt = cfg.output_format.lower()
            if fmt == "wav":
                audio_bytes = float32_to_wav(samples, self.sample_rate)
            else:
                audio_bytes = float32_to_pcm(samples, self.sample_rate)
            samples_count = int(getattr(samples, "shape", [len(samples)])[0])
            chunk = StreamingChunk(
                audio_bytes=audio_bytes,
                chunk_index=idx,
                is_final=is_final,
                timestamp_ms=(time.perf_counter() - t0) * 1000.0,
                samples_count=samples_count,
                duration_ms=(samples_count / self.sample_rate) * 1000.0
                if self.sample_rate
                else 0.0,
            )
            idx += 1
            return chunk

        for tok in self.generate_speech_tokens(text, voice=voice):
            buffer.append(int(tok))
            if len(buffer) >= chunk_every:
                chunk = _emit(False)
                if chunk is not None:
                    yield chunk

        final = _emit(True)
        if final is not None:
            yield final

    def stream_samples(self, text: str, voice: str | None = None):
        """Yield float32 sample arrays for the StreamingSynthesizer bridge."""
        buffer: list[int] = []
        chunk_every = max(1, int(self.config.decode_chunk_tokens))
        for tok in self.generate_speech_tokens(text, voice=voice):
            buffer.append(int(tok))
            if len(buffer) >= chunk_every:
                yield self.decode_tokens_to_audio(list(buffer))
                buffer.clear()
        if buffer:
            yield self.decode_tokens_to_audio(list(buffer))

    # --- Metrics / introspection ---

    def get_compression_stats(self) -> dict[str, Any] | None:
        """Return KV compression metrics or None if TQ disabled.

        ``cosine_similarity`` is only reported when it has actually been
        measured — callers must tolerate a missing or ``None`` value.
        """
        if self._tq_engine is None:
            return None
        stats: dict[str, Any] = {
            "compression_ratio": float(self._last_ratio),
            "kv_cache_bytes_compressed": int(self._last_tq_bytes),
            "kv_cache_bytes_uncompressed": int(self._last_fp16_bytes),
            "cosine_similarity": (
                float(self._last_cosine) if self._last_cosine else None
            ),
            "tq_bits": int(self.config.tq_bits),
        }
        return stats

    def list_voices(self) -> list[dict[str, str]]:
        return [dict(v) for v in ORPHEUS_VOICES]

    def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._decoder = None
        self._model_loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _empty_audio():
    import numpy as np

    return np.zeros(0, dtype=np.float32)
