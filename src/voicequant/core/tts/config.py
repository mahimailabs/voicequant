"""Configuration for Kokoro-based TTS engine."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TTSConfig(BaseModel):
    """TTS configuration for Kokoro ONNX backend."""

    model_name: str = Field(default="kokoro")
    model_path: str | None = Field(default=None)
    voices_path: str | None = Field(default=None)
    device: str = Field(default="auto")
    default_voice: str = Field(default="af_heart")
    sample_rate: int = Field(default=24000, ge=8000, le=96000)
    max_concurrent: int = Field(default=20, ge=1)
    speaker_cache_size: int = Field(default=50, ge=1)
    output_format: str = Field(default="wav")
    max_text_length: int = Field(default=4096, ge=1)

    @field_validator("device")
    @classmethod
    def _resolve_device(cls, value: str) -> str:
        if value not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        if value != "auto":
            return value
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @field_validator("output_format")
    @classmethod
    def _validate_output_format(cls, value: str) -> str:
        fmt = value.lower()
        if fmt not in {"wav", "pcm", "mp3"}:
            raise ValueError("output_format must be one of: wav, pcm, mp3")
        return fmt
