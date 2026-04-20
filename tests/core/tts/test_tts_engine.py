from __future__ import annotations

import sys
from types import ModuleType

from voicequant.core.protocol import ModalityEngine
from voicequant.core.tts.config import TTSConfig
from voicequant.core.tts.engine import SynthesisResult, TTSEngine


class _MockKokoro:
    def __init__(self, model_path=None, voices_path=None):
        self.model_path = model_path
        self.voices_path = voices_path
        self.calls = []

    def create(self, text, voice):
        self.calls.append((text, voice))
        return {"audio": [0.0, 0.1, -0.1, 0.0], "sample_rate": 24000}

    def get_voices(self):
        return ["af_heart", "bf_ember"]



def test_engine_lazy_loading_initial_state():
    e = TTSEngine(TTSConfig(device="cpu"))
    assert e.health().healthy is False


def test_engine_satisfies_protocol_typing():
    e: ModalityEngine = TTSEngine(TTSConfig(device="cpu"))
    assert e.health().modality == "tts"


def test_synthesis_result_dataclass_fields():
    r = SynthesisResult(b"abc", 24000, 0.25, "wav", "af_heart")
    assert r.audio_bytes == b"abc"
    assert r.sample_rate == 24000
    assert r.duration_seconds == 0.25
    assert r.format == "wav"
    assert r.voice == "af_heart"


def test_synthesize_uses_mocked_kokoro(monkeypatch):
    mod = ModuleType("kokoro_onnx")
    mod.Kokoro = _MockKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)

    e = TTSEngine(TTSConfig(device="cpu", output_format="pcm"))
    result = e.synthesize("hello", voice="af_heart", output_format="pcm")

    assert isinstance(result, SynthesisResult)
    assert result.format == "pcm"
    assert result.voice == "af_heart"
    assert len(result.audio_bytes) > 0
    assert e.health().healthy is True


def test_speaker_cache_used_across_syntheses(monkeypatch):
    mod = ModuleType("kokoro_onnx")
    mod.Kokoro = _MockKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)

    e = TTSEngine(TTSConfig(device="cpu"))
    e.synthesize("hello", voice="af_heart")
    e.synthesize("again", voice="af_heart")
    second = e.metrics()
    assert second["speaker_cache_hit_rate"] == 0.5
