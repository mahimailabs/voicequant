"""High-level TurboQuant wrapper with voice-optimized defaults.

Wraps the core TurboQuantEngine from the compression library with a clean
Python API designed for voice AI inference pipelines.
"""

from __future__ import annotations

from typing import Any

import torch
from rich.console import Console

from voicequant.core.llm.engine import TurboQuantEngine
from voicequant.core.llm.config import TurboQuantConfig

console = Console()


class TurboQuantWrapper:
    """Voice-optimized wrapper around TurboQuantEngine.

    Provides a simplified API for compressing and decompressing KV caches
    in multi-session voice inference pipelines.

    Args:
        config: TurboQuant configuration. Uses voice-optimized defaults if None.
        device: Device for compression operations ('cuda' or 'cpu').
    """

    def __init__(
        self,
        config: TurboQuantConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or TurboQuantConfig()
        self.device = device
        self._engine: TurboQuantEngine | None = None

        if self.config.is_turboquant_enabled:
            self._engine = TurboQuantEngine(
                head_dim=self.config.head_dim,
                total_bits=self.config.tq_bits,
                seed=self.config.seed,
                device=device,
            )

    @property
    def engine(self) -> TurboQuantEngine:
        """Access the underlying TurboQuantEngine."""
        if self._engine is None:
            raise RuntimeError("TurboQuant is not enabled (kv_cache_dtype='fp16')")
        return self._engine

    def compress_kv_cache(self, past_key_values: Any) -> dict[str, Any]:
        """Compress KV cache from model output.

        Args:
            past_key_values: KV cache from model forward pass (DynamicCache or tuple).

        Returns:
            Compressed KV cache dictionary.
        """
        return self.engine.compress_kv_cache(past_key_values)

    def build_cache(self, compressed: dict[str, Any]) -> Any:
        """Build a DynamicCache from compressed representation for next forward pass.

        Args:
            compressed: Compressed KV cache from compress_kv_cache().

        Returns:
            DynamicCache compatible with model.forward().
        """
        return self.engine.build_cache(compressed)

    def truncate_cache(self, compressed: dict[str, Any], seq_len: int) -> dict[str, Any]:
        """Truncate compressed cache to first seq_len tokens.

        Used when a voice agent user interrupts mid-generation to roll back.

        Args:
            compressed: Compressed KV cache.
            seq_len: Number of tokens to keep.

        Returns:
            Truncated compressed KV cache.
        """
        return self.engine.truncate_cache(compressed, seq_len)

    def compression_stats(self, past_key_values: Any) -> dict[str, Any]:
        """Compute compression statistics for KV cache.

        Args:
            past_key_values: KV cache from model forward pass.

        Returns:
            Dict with seq_len, n_layers, n_heads, fp16_bytes, tq_bytes, ratio.
        """
        return self.engine.compression_stats(past_key_values)

    def estimate_capacity(
        self,
        model_memory_gb: float,
        gpu_memory_gb: float,
        avg_context_len: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
    ) -> dict[str, Any]:
        """Estimate how many concurrent sessions fit in GPU memory.

        Args:
            model_memory_gb: Memory used by model weights.
            gpu_memory_gb: Total GPU memory.
            avg_context_len: Average context length per session.
            n_layers: Number of transformer layers.
            n_heads: Number of attention heads.

        Returns:
            Dict with fp16_sessions, tq_sessions, multiplier.
        """
        available_gb = (gpu_memory_gb * self.config.gpu_memory_utilization) - model_memory_gb
        available_bytes = available_gb * (1024 ** 3)

        sizes = self.engine.compressed_size_bytes(avg_context_len)
        fp16_per_session = sizes["fp16_bytes"] * n_layers * n_heads
        tq_per_session = sizes["compressed_bytes"] * n_layers * n_heads

        fp16_sessions = max(1, int(available_bytes / fp16_per_session))
        tq_sessions = max(1, int(available_bytes / tq_per_session))

        return {
            "fp16_sessions": fp16_sessions,
            "tq_sessions": tq_sessions,
            "multiplier": tq_sessions / max(fp16_sessions, 1),
            "available_memory_gb": available_gb,
            "fp16_per_session_mb": fp16_per_session / (1024 ** 2),
            "tq_per_session_mb": tq_per_session / (1024 ** 2),
        }

    @torch.no_grad()
    def validate_quality(
        self,
        seq_len: int = 512,
        n_trials: int = 10,
    ) -> dict[str, float]:
        """Validate compression quality using random data.

        Args:
            seq_len: Sequence length for test data.
            n_trials: Number of trials to average.

        Returns:
            Dict with avg_key_cosine, avg_val_cosine, min_key_cosine, min_val_cosine.
        """
        import torch.nn.functional as F

        key_sims: list[float] = []
        val_sims: list[float] = []

        for _ in range(n_trials):
            K = torch.randn(seq_len, self.config.head_dim, device=self.device, dtype=torch.float16)
            V = torch.randn(seq_len, self.config.head_dim, device=self.device, dtype=torch.float16)

            ck = self.engine.compress_keys_pytorch(K)
            cv = self.engine.compress_values_pytorch(V)
            V_recon = self.engine.decompress_values_pytorch(cv)

            k_cos = F.cosine_similarity(
                K.float().flatten().unsqueeze(0),
                ck["k_mse"].float().flatten().unsqueeze(0),
            ).item()

            v_cos = F.cosine_similarity(
                V.float().flatten().unsqueeze(0),
                V_recon.float().flatten().unsqueeze(0),
            ).item()

            key_sims.append(k_cos)
            val_sims.append(v_cos)

        return {
            "avg_key_cosine": sum(key_sims) / len(key_sims),
            "avg_val_cosine": sum(val_sims) / len(val_sims),
            "min_key_cosine": min(key_sims),
            "min_val_cosine": min(val_sims),
        }
