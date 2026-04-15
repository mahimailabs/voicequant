"""STT stub router — returns 501 until voicequant[stt] is installed."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1", tags=["stt"])


@router.post("/audio/transcriptions")
async def transcriptions_stub():
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "STT modality not installed. Install with pip install voicequant[stt]",
                "type": "not_implemented",
            }
        },
    )
