"""Real TTS router — OpenAI-compatible speech synthesis endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: str | None = None
    model: str | None = None
    response_format: str = "wav"
    speed: float = 1.0


def _content_type_for(fmt: str) -> str:
    mapping = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
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
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}") from e

        ext = result.format.lower()
        return Response(
            content=result.audio_bytes,
            media_type=_content_type_for(ext),
            headers={
                "Content-Disposition": f'attachment; filename="speech.{ext}"',
                "X-VoiceQuant-Voice": result.voice,
                "X-VoiceQuant-Sample-Rate": str(result.sample_rate),
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
