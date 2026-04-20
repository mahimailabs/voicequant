"""Real TTS router — OpenAI-compatible speech synthesis endpoint."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_-]+")
_ALLOWED_FORMATS = {"wav", "pcm", "mp3"}


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: str | None = None
    model: str | None = None
    response_format: str = "wav"
    speed: float = 1.0

    @field_validator("response_format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        fmt = value.lower()
        if fmt not in _ALLOWED_FORMATS:
            raise ValueError("response_format must be one of: wav, pcm, mp3")
        return fmt


def _sanitize_token(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    sanitized = _SAFE_TOKEN.sub("", value.replace("\r", "").replace("\n", ""))
    return sanitized or fallback


def _content_type_for(fmt: str) -> str:
    mapping = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "pcm": "audio/pcm",
    }
    return mapping.get(fmt.lower(), "application/octet-stream")


def build_router(get_engine) -> APIRouter:
    """Build the real TTS router using a TTSEngine getter."""
    router = APIRouter(prefix="/v1", tags=["tts"])

    @router.post("/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="TTS engine not ready")

        try:
            result = engine.synthesize(
                text=request.input,
                voice=request.voice,
                output_format=request.response_format,
            )
        except ImportError as e:
            logger.exception("TTS synthesis dependency error")
            raise HTTPException(status_code=501, detail="Synthesis failed") from e
        except ValueError:
            logger.exception("TTS synthesis request validation error")
            raise HTTPException(status_code=400, detail="Synthesis failed")
        except RuntimeError:
            logger.exception("TTS runtime synthesis error")
            raise HTTPException(status_code=500, detail="Synthesis failed")
        except Exception:
            logger.exception("Unexpected TTS synthesis error")
            raise HTTPException(status_code=500, detail="Synthesis failed")

        ext = _sanitize_token(result.format.lower(), "bin")
        safe_voice = _sanitize_token(result.voice, "unknown")
        filename_ext = ext if ext in _ALLOWED_FORMATS else "bin"
        return Response(
            content=result.audio_bytes,
            media_type=_content_type_for(ext),
            headers={
                "Content-Disposition": f'attachment; filename="speech.{filename_ext}"',
                "X-VoiceQuant-Voice": safe_voice,
                "X-VoiceQuant-Sample-Rate": str(int(result.sample_rate)),
            },
        )

    @router.get("/audio/speech/voices")
    async def list_voices() -> dict[str, Any]:
        engine = get_engine()
        voices = engine.list_voices() if engine is not None else []
        return {
            "object": "list",
            "data": [
                {
                    "voice_id": v,
                    "description": "Kokoro built-in voice",
                }
                for v in voices
            ],
        }

    return router
