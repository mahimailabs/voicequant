"""TTS stub router — returns 501 until voicequant[tts] is installed."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1", tags=["tts"])


@router.post("/audio/speech")
async def speech_stub():
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "TTS modality not installed. Install with pip install voicequant[tts]",
                "type": "not_implemented",
            }
        },
    )


@router.post("/audio/speech/stream")
async def speech_stream_stub():
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "TTS modality not installed. Install with pip install voicequant[tts]",
                "type": "not_implemented",
            }
        },
    )
