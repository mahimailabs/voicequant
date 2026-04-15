"""STT configuration (faster-whisper / CTranslate2 backend)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class STTConfig(BaseModel):
    model_name: str = Field(default="Systran/faster-whisper-large-v3")
    model_size: str = Field(default="large-v3")
    compute_type: str = Field(default="float16")
    device: str = Field(default="auto")
    device_index: int = Field(default=0)
    language: str | None = Field(default=None)
    beam_size: int = Field(default=1)
    batch_size: int = Field(default=16)
    max_concurrent: int = Field(default=10)
    vad_filter: bool = Field(default=True)
    condition_on_previous_text: bool = Field(default=False)

    @model_validator(mode="after")
    def _resolve_device(self):
        if self.device == "auto":
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        return self
