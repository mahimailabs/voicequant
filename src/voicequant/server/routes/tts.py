"""Real TTS router — OpenAI-compatible speech synthesis endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str | None = Field(default=None)
    model: str | None = Field(default=None)
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0)


_CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
}


def build_router(get_engine) -> APIRouter:
    """Build the real TTS router.

    `get_engine` returns the live TTSEngine instance (lazy).
    """
    router = APIRouter(prefix="/v1", tags=["tts"])

    @router.post("/audio/speech")
    async def speech(request: SpeechRequest) -> Any:
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="TTS engine not ready")

        fmt = (request.response_format or "wav").lower()
        try:
            result = engine.synthesize(
                request.input,
                voice=request.voice,
                output_format=fmt,
            )
        except ImportError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Synthesis failed: {e}"
            ) from e

        media_type = _CONTENT_TYPES.get(result.format, "application/octet-stream")
        filename = f"speech.{result.format}"
        return Response(
            content=result.audio_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/audio/speech/voices")
    async def list_voices() -> Any:
        engine = get_engine()
        if engine is None:
            return JSONResponse(content={"object": "list", "data": []})
        voices = engine.list_voices()
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": v["voice_id"],
                        "description": v.get("description", ""),
                        "object": "voice",
                    }
                    for v in voices
                ],
            }
        )

    return router
