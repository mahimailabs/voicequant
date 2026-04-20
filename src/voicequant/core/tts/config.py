"""TTS configuration (Kokoro ONNX backend)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TTSConfig(BaseModel):
    model_name: str = Field(default="kokoro")
    model_path: str | None = Field(default=None)
    device: str = Field(default="auto")
    default_voice: str = Field(default="af_heart")
    sample_rate: int = Field(default=24000)
    max_concurrent: int = Field(default=20)
    speaker_cache_size: int = Field(default=50)
    output_format: str = Field(default="wav")
    max_text_length: int = Field(default=4096)
    tq_bits: int = Field(default=4, description="Orpheus-only: TurboQuant bits")
    tq_enabled: bool = Field(default=True, description="Orpheus-only: enable KV compression")

    @model_validator(mode="after")
    def _resolve_device(self):
        if self.device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        return self
