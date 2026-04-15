"""Tests for TurboQuantWrapper — voice-optimized compression wrapper."""

import torch
import pytest

from voicequant.core.config import TurboQuantConfig
from voicequant.core.wrapper import TurboQuantWrapper


class TestTurboQuantConfig:
    def test_default_config(self):
        config = TurboQuantConfig()
        assert config.kv_cache_dtype == "tq4"
        assert config.residual_window == 256
        assert config.head_dim == 128
        assert config.default_max_tokens == 150
        assert config.stream is True
        assert config.temperature == 0.7

    def test_tq_bits_property(self):
        assert TurboQuantConfig(kv_cache_dtype="tq4").tq_bits == 4
        assert TurboQuantConfig(kv_cache_dtype="tq3").tq_bits == 3
        assert TurboQuantConfig(kv_cache_dtype="fp16").tq_bits == 0

    def test_is_turboquant_enabled(self):
        assert TurboQuantConfig(kv_cache_dtype="tq4").is_turboquant_enabled is True
        assert TurboQuantConfig(kv_cache_dtype="tq3").is_turboquant_enabled is True
        assert TurboQuantConfig(kv_cache_dtype="fp16").is_turboquant_enabled is False

    def test_voice_defaults_documented(self):
        """Voice-optimized defaults should be set for real-time use cases."""
        config = TurboQuantConfig()
        # Low max tokens for short voice responses
        assert config.default_max_tokens <= 200
        # High concurrency
        assert config.max_num_seqs >= 32
        # Streaming enabled
        assert config.stream is True


class TestTurboQuantWrapper:
    def test_wrapper_creates_engine(self):
        wrapper = TurboQuantWrapper(device="cpu")
        assert wrapper.engine is not None
        assert wrapper.config.kv_cache_dtype == "tq4"

    def test_wrapper_fp16_raises_on_engine_access(self):
        config = TurboQuantConfig(kv_cache_dtype="fp16")
        wrapper = TurboQuantWrapper(config=config, device="cpu")
        with pytest.raises(RuntimeError, match="TurboQuant is not enabled"):
            _ = wrapper.engine

    def test_validate_quality_tq4(self):
        wrapper = TurboQuantWrapper(device="cpu")
        quality = wrapper.validate_quality(seq_len=64, n_trials=3)
        assert "avg_key_cosine" in quality
        assert "avg_val_cosine" in quality
        assert "min_key_cosine" in quality
        assert "min_val_cosine" in quality
        # 4-bit should have decent quality
        assert quality["avg_val_cosine"] > 0.90

    def test_validate_quality_tq3(self):
        config = TurboQuantConfig(kv_cache_dtype="tq3")
        wrapper = TurboQuantWrapper(config=config, device="cpu")
        quality = wrapper.validate_quality(seq_len=64, n_trials=3)
        assert quality["avg_val_cosine"] > 0.85

    def test_estimate_capacity(self):
        wrapper = TurboQuantWrapper(device="cpu")
        capacity = wrapper.estimate_capacity(
            model_memory_gb=4.0,
            gpu_memory_gb=16.0,
            avg_context_len=4096,
        )
        assert capacity["tq_sessions"] > capacity["fp16_sessions"]
        assert capacity["multiplier"] > 1.0
        assert capacity["available_memory_gb"] > 0

    def test_compress_and_decompress_roundtrip(self):
        wrapper = TurboQuantWrapper(device="cpu")
        K = torch.randn(32, 128, dtype=torch.float16)
        V = torch.randn(32, 128, dtype=torch.float16)

        ck = wrapper.engine.compress_keys_pytorch(K)
        cv = wrapper.engine.compress_values_pytorch(V)
        V_recon = wrapper.engine.decompress_values_pytorch(cv)

        assert V_recon.shape == V.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            V.float().flatten().unsqueeze(0),
            V_recon.float().flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.85

    def test_truncate_cache(self):
        wrapper = TurboQuantWrapper(device="cpu")
        compressed = {
            "layers": [
                (
                    [{"indices": torch.zeros(100, 128, dtype=torch.uint8),
                      "k_mse": torch.zeros(100, 128),
                      "qjl_signs": torch.zeros(100, 128, dtype=torch.int8),
                      "vec_norms": torch.zeros(100),
                      "residual_norms": torch.zeros(100)}],
                    [{"indices": torch.zeros(100, 128, dtype=torch.uint8),
                      "vec_norms": torch.zeros(100)}],
                )
            ]
        }
        truncated = wrapper.truncate_cache(compressed, seq_len=50)
        ck = truncated["layers"][0][0][0]
        assert ck["indices"].shape[0] == 50
        assert ck["k_mse"].shape[0] == 50
