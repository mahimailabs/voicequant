"""TurboQuant configuration with voice-optimized defaults."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TurboQuantConfig(BaseModel):
    """Configuration for TurboQuant KV cache compression.

    Voice AI has specific requirements that differ from general LLM serving:

    1. Low max_tokens: voice responses should be 1-3 sentences (50-150 tokens).
       A 500-token response takes 10+ seconds to speak = terrible UX.
    2. High concurrency: many short sessions, not few long ones.
    3. Streaming is mandatory: TTFB matters more than total throughput.
    4. Context grows gradually: starts at ~1K, grows to ~8K over 10+ turns.
    5. System prompts are large: 500-2000 tokens of persona + tools.
    """

    # TurboQuant compression settings
    kv_cache_dtype: str = Field(
        default="tq4",
        description="KV cache quantization: 'tq4' (4-bit, 0.99+ cosine), 'tq3' (3-bit), or 'fp16'",
    )
    residual_window: int = Field(
        default=256,
        description="Number of recent tokens kept in FP16 for quality preservation",
    )
    head_dim: int = Field(default=128, description="Attention head dimension")
    seed: int = Field(default=42, description="Random seed for rotation matrices")

    # vLLM engine settings
    max_model_len: int = Field(default=32768, description="Maximum context length")
    gpu_memory_utilization: float = Field(
        default=0.90,
        description="GPU memory utilization (0.0-1.0), leave headroom for bursts",
    )
    max_num_seqs: int = Field(default=64, description="Maximum concurrent sequences")
    max_num_batched_tokens: int = Field(default=8192, description="Max tokens per batch")

    # Response defaults (overridable per-request)
    default_max_tokens: int = Field(
        default=150,
        description="Default max tokens for voice responses (short 1-3 sentences)",
    )
    temperature: float = Field(default=0.7, description="Sampling temperature")
    stream: bool = Field(default=True, description="Enable streaming by default")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")

    @property
    def tq_bits(self) -> int:
        """Extract numeric bits from kv_cache_dtype."""
        if self.kv_cache_dtype == "tq4":
            return 4
        elif self.kv_cache_dtype == "tq3":
            return 3
        return 0

    @property
    def is_turboquant_enabled(self) -> bool:
        """Whether TurboQuant compression is active."""
        return self.kv_cache_dtype in ("tq3", "tq4")
