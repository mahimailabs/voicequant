"""VoiceQuant TTS core package."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "TTSConfig":
        from voicequant.core.tts.config import TTSConfig

        return TTSConfig
    if name == "SpeakerCache":
        from voicequant.core.tts.speaker_cache import SpeakerCache

        return SpeakerCache
    if name == "TTSEngine":
        from voicequant.core.tts.engine import TTSEngine

        return TTSEngine
    if name == "SynthesisResult":
        from voicequant.core.tts.engine import SynthesisResult

        return SynthesisResult
    if name in {
        "float32_to_wav",
        "float32_to_pcm",
        "wav_to_mp3",
        "wav_to_opus",
        "get_audio_duration",
    }:
        from voicequant.core.tts import audio as _audio

        return getattr(_audio, name)
    raise AttributeError(f"module 'voicequant.core.tts' has no attribute {name!r}")


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
