from __future__ import annotations

import pytest

from voicequant.core.tts.config import TTSConfig


def test_tts_config_defaults():
    cfg = TTSConfig()
    assert cfg.model_name == "kokoro"
    assert cfg.default_voice == "af_heart"
    assert cfg.sample_rate == 24000
    assert cfg.max_concurrent == 20
    assert cfg.speaker_cache_size == 50
    assert cfg.output_format == "wav"
    assert cfg.max_text_length == 4096
    assert cfg.device in {"cpu", "cuda"}


def test_tts_config_device_auto_detection_cpu(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    cfg = TTSConfig(device="auto")
    assert cfg.device == "cpu"


def test_tts_config_custom_values_override_defaults():
    cfg = TTSConfig(
        model_name="kokoro-custom",
        model_path="/tmp/kokoro.onnx",
        device="cpu",
        default_voice="bf_ember",
        sample_rate=22050,
        max_concurrent=4,
        speaker_cache_size=7,
        output_format="pcm",
        max_text_length=500,
    )
    assert cfg.model_name == "kokoro-custom"
    assert cfg.model_path == "/tmp/kokoro.onnx"
    assert cfg.device == "cpu"
    assert cfg.default_voice == "bf_ember"
    assert cfg.sample_rate == 22050
    assert cfg.max_concurrent == 4
    assert cfg.speaker_cache_size == 7
    assert cfg.output_format == "pcm"
    assert cfg.max_text_length == 500


def test_max_text_length_validation():
    with pytest.raises(ValueError):
        TTSConfig(max_text_length=0)
