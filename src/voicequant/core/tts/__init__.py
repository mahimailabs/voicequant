"""VoiceQuant TTS — Kokoro-backed text-to-speech."""

from voicequant.core.tts.config import TTSConfig
from voicequant.core.tts.engine import KOKORO_VOICES, SynthesisResult, TTSEngine
from voicequant.core.tts.speaker_cache import SpeakerCache

__all__ = [
    "KOKORO_VOICES",
    "SpeakerCache",
    "SynthesisResult",
    "TTSConfig",
    "TTSEngine",
]
