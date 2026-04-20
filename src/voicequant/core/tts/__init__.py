"""VoiceQuant TTS core package."""

from voicequant.core.tts.audio import (
    float32_to_pcm,
    float32_to_wav,
    get_audio_duration,
    wav_to_mp3,
    wav_to_opus,
)
from voicequant.core.tts.config import TTSConfig
from voicequant.core.tts.engine import SynthesisResult, TTSEngine
from voicequant.core.tts.speaker_cache import SpeakerCache

__all__ = [
    "TTSConfig",
    "SpeakerCache",
    "TTSEngine",
    "SynthesisResult",
    "float32_to_wav",
    "float32_to_pcm",
    "wav_to_mp3",
    "wav_to_opus",
    "get_audio_duration",
]
