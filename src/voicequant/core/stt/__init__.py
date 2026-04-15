"""VoiceQuant STT — faster-whisper-backed speech-to-text."""

from voicequant.core.stt.config import STTConfig
from voicequant.core.stt.engine import STTEngine, TranscriptionResult

__all__ = ["STTConfig", "STTEngine", "TranscriptionResult"]
