"""Test truncate_cache() correctness."""

import torch
import pytest

from voicequant import TurboQuantEngine


def _make_compressed(engine, seq_len, n_heads=2, n_layers=1):
    """Build a compressed cache dict by calling the private PT methods directly."""
    layers = []
    for _ in range(n_layers):
        ck_list, cv_list = [], []
        for _ in range(n_heads):
            K = torch.randn(seq_len, engine.head_dim).half()
            V = torch.randn(seq_len, engine.head_dim).half()
            ck_list.append(engine._compress_keys_pt(K))
            cv_list.append(engine._compress_values_pt(V))
        layers.append((ck_list, cv_list))
    return {"layers": layers}


def test_round_trip_shape():
    """Compress M tokens, truncate to N, build_cache -> seq_len is N."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    compressed = _make_compressed(engine, seq_len=256, n_heads=2, n_layers=1)

    truncated = engine.truncate_cache(compressed, seq_len=64)

    for ck_list, cv_list in truncated["layers"]:
        for ck in ck_list:
            assert ck["indices"].shape[0] == 64
            assert ck["k_mse"].shape[0] == 64
            assert ck["qjl_signs"].shape[0] == 64
            assert ck["vec_norms"].shape[0] == 64
            assert ck["residual_norms"].shape[0] == 64
        for cv in cv_list:
            assert cv["indices"].shape[0] == 64
            assert cv["vec_norms"].shape[0] == 64


def test_bitwise_equivalence():
    """Truncating compress(tokens[0:M]) to N == compressing only tokens[0:N]."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, seed=42, device="cpu")
    M, N = 256, 64

    K = torch.randn(M, 128).half()
    V = torch.randn(M, 128).half()

    # Compress full then truncate
    ck_full = engine._compress_keys_pt(K)
    cv_full = engine._compress_values_pt(V)
    compressed_full = {"layers": [([ck_full], [cv_full])]}
    truncated = engine.truncate_cache(compressed_full, seq_len=N)

    # Compress only first N tokens
    ck_short = engine._compress_keys_pt(K[:N])
    cv_short = engine._compress_values_pt(V[:N])

    ck_t = truncated["layers"][0][0][0]
    cv_t = truncated["layers"][0][1][0]

    assert torch.equal(ck_t["indices"], ck_short["indices"])
    assert torch.equal(ck_t["qjl_signs"], ck_short["qjl_signs"])
    assert torch.allclose(ck_t["k_mse"], ck_short["k_mse"])
    assert torch.allclose(ck_t["vec_norms"], ck_short["vec_norms"])
    assert torch.allclose(ck_t["residual_norms"], ck_short["residual_norms"])
    assert torch.equal(cv_t["indices"], cv_short["indices"])
    assert torch.allclose(cv_t["vec_norms"], cv_short["vec_norms"])


def test_noop_for_large_seq_len():
    """Truncating to seq_len > actual length returns original length."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    compressed = _make_compressed(engine, seq_len=64)

    truncated = engine.truncate_cache(compressed, seq_len=99999)

    ck = truncated["layers"][0][0][0]
    assert ck["indices"].shape[0] == 64


def test_none_input():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    assert engine.truncate_cache(None, 10) is None


def test_empty_layers():
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    result = engine.truncate_cache({"layers": []}, 10)
    assert result == {"layers": []}


def test_zero_truncation():
    """Truncating to 0 returns valid empty tensors."""
    engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cpu")
    compressed = _make_compressed(engine, seq_len=64)

    truncated = engine.truncate_cache(compressed, seq_len=0)

    for ck_list, cv_list in truncated["layers"]:
        for ck in ck_list:
            assert ck["indices"].shape[0] == 0
            assert ck["k_mse"].shape[0] == 0
        for cv in cv_list:
            assert cv["indices"].shape[0] == 0
            assert cv["vec_norms"].shape[0] == 0
