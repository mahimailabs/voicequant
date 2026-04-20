"""TTSConfig tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voicequant.core.tts.config import TTSConfig


def test_tts_config_defaults():
    cfg = TTSConfig()
    assert cfg.model_name == "kokoro"
    assert cfg.model_path is None
    assert cfg.default_voice == "af_heart"
    assert cfg.sample_rate == 24000
    assert cfg.max_concurrent == 20
    assert cfg.speaker_cache_size == 50
    assert cfg.output_format == "wav"
    assert cfg.max_text_length == 4096
    assert cfg.device in {"cpu", "cuda"}


def test_tts_config_device_auto_resolves():
    cfg = TTSConfig(device="auto")
    assert cfg.device in {"cpu", "cuda"}


def test_tts_config_custom_overrides():
    cfg = TTSConfig(
        device="cpu",
        default_voice="am_adam",
        sample_rate=16000,
        speaker_cache_size=10,
        output_format="pcm",
        max_text_length=2048,
    )
    assert cfg.device == "cpu"
    assert cfg.default_voice == "am_adam"
    assert cfg.sample_rate == 16000
    assert cfg.speaker_cache_size == 10
    assert cfg.output_format == "pcm"
    assert cfg.max_text_length == 2048


def test_tts_config_max_text_length_accepts_int():
    cfg = TTSConfig(max_text_length=100)
    assert cfg.max_text_length == 100


def test_tts_config_max_text_length_rejects_non_int():
    with pytest.raises(ValidationError):
        TTSConfig(max_text_length="not-an-int")  # type: ignore[arg-type]
