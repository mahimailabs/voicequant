"""Real STT router — OpenAI Whisper-compatible transcription endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse


def build_router(get_engine) -> APIRouter:
    """Build the real STT router.

    `get_engine` returns the live STTEngine instance (lazy).
    """
    router = APIRouter(prefix="/v1", tags=["stt"])

    @router.post("/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(None),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ) -> Any:
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="STT engine not ready")

        audio_bytes = await file.read()
        suffix = ""
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]
        else:
            suffix = ".wav"

        try:
            result = engine.transcribe_bytes(
                audio_bytes,
                language=language,
                response_format=response_format,
                suffix=suffix,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Transcription failed: {e}"
            ) from e

        if response_format == "text":
            return PlainTextResponse(result.text)

        if response_format == "verbose_json":
            return JSONResponse(
                content={
                    "task": "transcribe",
                    "language": result.language,
                    "duration": result.duration,
                    "text": result.text,
                    "segments": result.segments,
                }
            )

        return JSONResponse(content={"text": result.text})

    @router.get("/audio/transcriptions/models")
    async def list_stt_models() -> dict[str, Any]:
        engine = get_engine()
        configured = engine.config.model_name if engine is not None else None
        return {
            "object": "list",
            "data": [
                {
                    "id": configured or "not_configured",
                    "object": "model",
                    "owned_by": "voicequant",
                }
            ],
        }

    return router
