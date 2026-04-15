"""Tests for TurboQuant per-layer quality validator."""

from voicequant.core.validator import validate_model


class TestValidator:
    def test_validate_tq4_passes(self):
        result = validate_model(
            model="test-model",
            bits=4,
            threshold=0.85,
            seq_len=64,
            n_trials=3,
        )
        assert result["overall_pass"] is True
        assert "quality" in result
        assert "capacity" in result

    def test_validate_tq3_passes_low_threshold(self):
        result = validate_model(
            model="test-model",
            bits=3,
            threshold=0.80,
            seq_len=64,
            n_trials=3,
        )
        assert result["overall_pass"] is True

    def test_validate_returns_capacity(self):
        result = validate_model(
            model="test-model",
            bits=4,
            threshold=0.85,
            seq_len=64,
            n_trials=2,
        )
        capacity = result["capacity"]
        assert capacity["tq_sessions"] > 0
        assert capacity["fp16_sessions"] > 0
        assert capacity["multiplier"] > 1.0
