"""Voice-optimized server configuration defaults.

Voice AI has different optimal settings than general LLM serving:
1. Low max_tokens: voice responses should be 1-3 sentences (50-150 tokens).
2. High concurrency: many short sessions, not few long ones.
3. Streaming is mandatory: TTFB matters more than total throughput.
4. Context grows gradually: starts at ~1K, grows to ~8K over 10+ turns.
5. System prompts are large: 500-2000 tokens of persona + tools.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

VOICE_DEFAULTS = {
    "kv_cache_dtype": "tq4",
    "residual_window": 256,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.90,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 8192,
    "default_max_tokens": 150,
    "temperature": 0.7,
    "stream": True,
    "host": "0.0.0.0",
    "port": 8000,
}

# Estimated concurrent sessions by GPU at 4K context with 7B AWQ model
GPU_CAPACITY_ESTIMATES = {
    "T4": {"memory_gb": 16, "model_gb": 4, "fp16_sessions": 8, "tq4_sessions": 40},
    "A10G": {"memory_gb": 24, "model_gb": 4, "fp16_sessions": 13, "tq4_sessions": 65},
    "L4": {"memory_gb": 24, "model_gb": 4, "fp16_sessions": 13, "tq4_sessions": 65},
    "A100": {"memory_gb": 80, "model_gb": 4, "fp16_sessions": 50, "tq4_sessions": 250},
    "H100": {"memory_gb": 80, "model_gb": 4, "fp16_sessions": 50, "tq4_sessions": 250},
}


class ServerConfig(BaseModel):
    """VoiceQuant server configuration."""

    model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct-AWQ", description="HuggingFace model ID"
    )
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")

    # TurboQuant settings
    kv_cache_dtype: str = Field(
        default="tq4", description="KV cache dtype: tq4, tq3, fp16"
    )
    residual_window: int = Field(default=256, description="FP16 residual window size")

    # vLLM engine settings
    max_model_len: int = Field(default=32768, description="Max context length")
    gpu_memory_utilization: float = Field(
        default=0.90, description="GPU memory utilization"
    )
    max_num_seqs: int = Field(default=64, description="Max concurrent sequences")
    max_num_batched_tokens: int = Field(default=8192, description="Max batch tokens")

    # Voice defaults
    default_max_tokens: int = Field(default=150, description="Default max tokens")
    temperature: float = Field(default=0.7, description="Default temperature")

    # Multi-modality (M1 scaffolding)
    stt_config: dict | None = Field(default=None)  # Replaced with STTConfig in M2
    tts_config: dict | None = Field(default=None)  # Replaced with TTSConfig in M4
    enabled_modalities: list[str] = Field(default_factory=lambda: ["llm"])

    @model_validator(mode="after")
    def _sync_modalities(self):
        if self.stt_config is not None and "stt" not in self.enabled_modalities:
            self.enabled_modalities.append("stt")
        if self.tts_config is not None and "tts" not in self.enabled_modalities:
            self.enabled_modalities.append("tts")
        return self
