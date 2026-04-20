"""Tests for the Orpheus adapter (M5 Phase 3)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from voicequant.core.tts.orpheus_adapter import OrpheusAdapter, OrpheusConfig


def test_orpheus_config_defaults():
    cfg = OrpheusConfig()
    assert cfg.model_name == "canopylabs/orpheus-3b-0.1-ft"
    assert cfg.tq_bits == 4
    assert cfg.tq_enabled is True
    assert cfg.max_tokens == 2048
    assert 0.0 <= cfg.top_p <= 1.0


def test_orpheus_config_custom():
    cfg = OrpheusConfig(tq_bits=3, tq_enabled=False, temperature=0.3, max_tokens=512)
    assert cfg.tq_bits == 3
    assert cfg.tq_enabled is False
    assert cfg.temperature == 0.3
    assert cfg.max_tokens == 512


def test_orpheus_adapter_instantiates_without_model_load():
    adapter = OrpheusAdapter(OrpheusConfig(device="cpu"))
    assert adapter._model_loaded is False
    assert adapter._model is None


def test_orpheus_adapter_has_turboquant_engine_when_enabled():
    adapter = OrpheusAdapter(OrpheusConfig(tq_enabled=True, device="cpu"))
    assert adapter._tq_engine is not None


def test_orpheus_adapter_has_no_turboquant_engine_when_disabled():
    adapter = OrpheusAdapter(OrpheusConfig(tq_enabled=False, device="cpu"))
    assert adapter._tq_engine is None


def test_orpheus_adapter_imports_turboquant_from_core_llm():
    """Verify the cross-modality dependency direction.

    core/tts/orpheus_adapter must import from core/llm/engine; core/llm
    must not import anything from core/tts.
    """
    from voicequant.core.llm.engine import TurboQuantEngine
    from voicequant.core.tts import orpheus_adapter

    src = open(orpheus_adapter.__file__).read()
    assert "from voicequant.core.llm.engine import TurboQuantEngine" in src
    assert orpheus_adapter.TurboQuantEngine is TurboQuantEngine


def test_core_llm_does_not_import_core_tts():
    """Ensure no file in core/llm references core.tts — enforces direction."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3] / "src" / "voicequant" / "core" / "llm"
    for path in root.rglob("*.py"):
        text = path.read_text()
        assert "voicequant.core.tts" not in text, (
            f"{path} imports from core/tts (forbidden direction)"
        )


def test_get_compression_stats_none_when_tq_disabled():
    adapter = OrpheusAdapter(OrpheusConfig(tq_enabled=False, device="cpu"))
    assert adapter.get_compression_stats() is None


def test_get_compression_stats_shape_when_tq_enabled():
    adapter = OrpheusAdapter(OrpheusConfig(tq_enabled=True, device="cpu"))
    stats = adapter.get_compression_stats()
    assert stats is not None
    assert set(stats.keys()) >= {
        "compression_ratio",
        "kv_cache_bytes_compressed",
        "kv_cache_bytes_uncompressed",
        "cosine_similarity",
    }


def test_load_model_raises_helpful_error_when_package_missing(monkeypatch):
    import builtins

    original = builtins.__import__

    def blocked(name, *a, **k):
        if name == "orpheus_tts":
            raise ImportError("no orpheus-tts installed")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "orpheus_tts", raising=False)

    adapter = OrpheusAdapter(OrpheusConfig(device="cpu"))
    with pytest.raises(ImportError, match="orpheus-tts"):
        adapter.load_model()


def _install_mock_orpheus(monkeypatch, adapter: OrpheusAdapter) -> MagicMock:
    """Wire a fake Orpheus model+tokenizer+decoder into the adapter."""

    import torch

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    fake_decoder = MagicMock()

    # Tokenizer returns a tensor-friendly dict.
    def tokenize(text, return_tensors="pt"):
        return MagicMock(
            input_ids=torch.tensor([[1, 2, 3]]),
            __getitem__=lambda self, k: torch.tensor([[1, 2, 3]]),
            to=lambda *a, **k: {"input_ids": torch.tensor([[1, 2, 3]])},
        )

    def simple_to(*a, **k):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    tokenizer_output = MagicMock()
    tokenizer_output.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    fake_tokenizer.return_value = tokenizer_output
    fake_tokenizer.eos_token_id = 999

    # Model call returns logits + past_key_values
    class FakeOut:
        logits = torch.zeros(1, 3, 10)
        past_key_values = (
            (torch.zeros(1, 2, 3, 128), torch.zeros(1, 2, 3, 128)),
        )

    fake_model.return_value = FakeOut()
    fake_decoder.decode.return_value = np.zeros(480, dtype=np.float32)

    adapter._model = fake_model
    adapter._tokenizer = fake_tokenizer
    adapter._decoder = fake_decoder
    adapter._model_loaded = True
    return fake_model


def test_generate_speech_tokens_invokes_compress(monkeypatch):
    """When TQ is enabled the adapter must call compress_kv_cache per step."""
    adapter = OrpheusAdapter(
        OrpheusConfig(tq_enabled=True, device="cpu", max_tokens=2)
    )
    _install_mock_orpheus(monkeypatch, adapter)

    tq = adapter._tq_engine
    assert tq is not None
    compress_calls = []
    original_compress = tq.compress_kv_cache
    original_build = tq.build_cache
    original_stats = tq.compression_stats

    def fake_compress(past):
        compress_calls.append(past)
        return {"layers": []}

    def fake_build(comp):
        return ()

    def fake_stats(past):
        return {"fp16_bytes": 100, "tq_bytes": 20, "ratio": 5.0,
                "seq_len": 3, "n_layers": 1, "n_heads": 2}

    tq.compress_kv_cache = fake_compress
    tq.build_cache = fake_build
    tq.compression_stats = fake_stats

    try:
        tokens = list(adapter.generate_speech_tokens("hello"))
    finally:
        tq.compress_kv_cache = original_compress
        tq.build_cache = original_build
        tq.compression_stats = original_stats

    assert len(tokens) >= 1
    assert len(compress_calls) >= 1


def test_decode_tokens_to_audio_returns_float32(monkeypatch):
    adapter = OrpheusAdapter(OrpheusConfig(tq_enabled=False, device="cpu"))
    _install_mock_orpheus(monkeypatch, adapter)

    samples = adapter.decode_tokens_to_audio([1, 2, 3, 4, 5])
    assert samples.dtype == np.float32
    assert samples.ndim == 1


def test_orpheus_list_voices_non_empty():
    adapter = OrpheusAdapter(OrpheusConfig(device="cpu"))
    voices = adapter.list_voices()
    assert len(voices) > 0
    assert "voice_id" in voices[0]
