"""Tests for the multi-backend TTSEngine (M5 Phase 4)."""

from __future__ import annotations

from unittest.mock import MagicMock

from voicequant.core.tts.config import TTSConfig
from voicequant.core.tts.engine import TTSEngine, _detect_backend


def test_detect_backend_kokoro():
    assert _detect_backend("kokoro") == "kokoro"
    assert _detect_backend("kokoro-v1") == "kokoro"
    assert _detect_backend("") == "kokoro"
    assert _detect_backend("some-random-model") == "kokoro"


def test_detect_backend_orpheus():
    assert _detect_backend("orpheus") == "orpheus"
    assert _detect_backend("orpheus-3b") == "orpheus"
    assert _detect_backend("canopylabs/orpheus-3b-0.1-ft") == "orpheus"


def test_tts_engine_backend_is_kokoro_by_default():
    e = TTSEngine(TTSConfig(model_name="kokoro", device="cpu"))
    assert e.backend == "kokoro"


def test_tts_engine_backend_is_orpheus_for_orpheus_models():
    e = TTSEngine(TTSConfig(model_name="orpheus", device="cpu"))
    assert e.backend == "orpheus"

    e2 = TTSEngine(TTSConfig(model_name="canopylabs/orpheus-3b-0.1-ft", device="cpu"))
    assert e2.backend == "orpheus"


def test_tts_config_has_tq_fields():
    cfg = TTSConfig()
    assert cfg.tq_bits == 4
    assert cfg.tq_enabled is True

    cfg2 = TTSConfig(tq_bits=3, tq_enabled=False)
    assert cfg2.tq_bits == 3
    assert cfg2.tq_enabled is False


def test_get_compression_stats_returns_none_for_kokoro():
    e = TTSEngine(TTSConfig(model_name="kokoro", device="cpu"))
    assert e.get_compression_stats() is None


def test_get_compression_stats_dict_for_orpheus_when_loaded(monkeypatch):
    """With an injected Orpheus adapter, get_compression_stats returns a dict."""
    e = TTSEngine(TTSConfig(model_name="orpheus", device="cpu", tq_enabled=True))

    fake_orpheus = MagicMock()
    fake_orpheus.get_compression_stats.return_value = {
        "compression_ratio": 5.2,
        "kv_cache_bytes_compressed": 200,
        "kv_cache_bytes_uncompressed": 1040,
        "cosine_similarity": 0.99,
    }
    e._orpheus = fake_orpheus
    e._model_loaded = True

    stats = e.get_compression_stats()
    assert stats is not None
    assert stats["compression_ratio"] == 5.2
    assert stats["kv_cache_bytes_compressed"] == 200


def test_get_compression_stats_none_for_orpheus_when_not_loaded():
    e = TTSEngine(TTSConfig(model_name="orpheus", device="cpu"))
    # Not loaded -> _orpheus is None -> returns None
    assert e.get_compression_stats() is None


def test_load_model_orpheus_raises_helpful_error_when_not_installed(monkeypatch):
    """Loading orpheus without orpheus-tts installed raises an informative error."""
    import builtins
    import sys

    original = builtins.__import__

    def blocked(name, *a, **k):
        if name == "orpheus_tts":
            raise ImportError("no orpheus-tts")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "orpheus_tts", raising=False)

    e = TTSEngine(TTSConfig(model_name="orpheus", device="cpu"))
    import pytest

    with pytest.raises(ImportError):
        e.load_model()


def test_tq_fields_propagated_to_orpheus(monkeypatch):
    """When loading orpheus, TTSConfig.tq_bits/tq_enabled flow into OrpheusConfig."""
    captured = {}

    class DummyOrpheus:
        def __init__(self, cfg):
            captured["cfg"] = cfg

        def load_model(self):
            pass

        def get_compression_stats(self):
            return None

    import voicequant.core.tts.orpheus_adapter as oa

    monkeypatch.setattr(oa, "OrpheusAdapter", DummyOrpheus)

    e = TTSEngine(
        TTSConfig(model_name="orpheus", device="cpu", tq_bits=3, tq_enabled=True)
    )
    e.load_model()
    cfg = captured.get("cfg")
    assert cfg is not None
    assert cfg.tq_bits == 3
    assert cfg.tq_enabled is True
