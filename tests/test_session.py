"""Test CacheSession lifecycle."""

import torch
import pytest

from voicequant import TurboQuantEngine
from voicequant.cache import CacheSession


def _make_past_kv(n_layers=2, n_heads=2, seq_len=64, d=128):
    """Build fake past_key_values as list of (K, V) tuples."""
    past_kv = []
    for _ in range(n_layers):
        k = torch.randn(1, n_heads, seq_len, d, dtype=torch.float16)
        v = torch.randn(1, n_heads, seq_len, d, dtype=torch.float16)
        past_kv.append((k, v))
    return past_kv


def test_session_initial_state():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)

    assert session.seq_len == 0
    assert session.compressed is None
    assert session.engine is engine


def test_session_compress():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)

    past_kv = _make_past_kv(seq_len=64)
    session.compress(past_kv)

    assert session.seq_len == 64
    assert session.compressed is not None
    assert len(session.compressed["layers"]) == 2


def test_session_truncate():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)

    past_kv = _make_past_kv(seq_len=128)
    session.compress(past_kv)
    assert session.seq_len == 128

    session.truncate(64)
    assert session.seq_len == 64

    # Check underlying tensors are truncated
    ck = session.compressed["layers"][0][0][0]
    assert ck["indices"].shape[0] == 64


def test_session_truncate_without_compress():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)
    session.truncate(64)
    assert session.seq_len == 0


def test_session_clear():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)

    past_kv = _make_past_kv(seq_len=64)
    session.compress(past_kv)
    assert session.seq_len == 64

    session.clear()
    assert session.seq_len == 0
    assert session.compressed is None


def test_build_without_compress_raises():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)
    with pytest.raises(ValueError, match="No compressed cache"):
        session.build()


def test_session_stats():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    session = CacheSession(engine)

    past_kv = _make_past_kv(seq_len=256)
    session.compress(past_kv)

    stats = session.stats()
    assert "compression_ratio" in stats
    assert stats["compression_ratio"] > 1.0
