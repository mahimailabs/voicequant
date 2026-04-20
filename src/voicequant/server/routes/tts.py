"""Real TTS router — OpenAI-compatible speech synthesis endpoint."""

from __future__ import annotations

import base64
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str | None = Field(default=None)
    model: str | None = Field(default=None)
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0)
    stream: bool = Field(default=False)


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
    async def speech(request: SpeechRequest, http_request: Request) -> Any:
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="TTS engine not ready")

        if request.stream:
            return await speech_stream(request, http_request)

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

    def _streaming_format(fmt: str) -> str:
        """Map a request format to a streaming-safe payload format.

        Streaming responses must use raw, chunk-safe formats. WAV has a
        header that describes total length so we downgrade it to PCM,
        and any other/unknown format also falls back to PCM.
        """
        low = (fmt or "pcm").lower()
        if low == "pcm":
            return "pcm"
        return "pcm"

    async def _sse_generator(sync_stream):
        """Adapt a sync StreamingChunk iterator to SSE event strings."""
        total_ms = 0.0
        total_chunks = 0
        ttfa_ms = 0.0
        for chunk in sync_stream:
            if chunk.chunk_index == 0:
                ttfa_ms = chunk.timestamp_ms
            total_chunks = chunk.chunk_index + 1
            total_ms = chunk.timestamp_ms + chunk.duration_ms
            payload = {
                "chunk_index": chunk.chunk_index,
                "audio": base64.b64encode(chunk.audio_bytes).decode("ascii"),
                "is_final": chunk.is_final,
                "duration_ms": round(chunk.duration_ms, 3),
                "samples": chunk.samples_count,
                "timestamp_ms": round(chunk.timestamp_ms, 3),
            }
            yield f"event: audio\ndata: {json.dumps(payload)}\n\n"
        done = {
            "total_duration_ms": round(total_ms, 3),
            "total_chunks": total_chunks,
            "ttfa_ms": round(ttfa_ms, 3),
        }
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    async def _raw_generator(sync_stream):
        """Adapt a sync StreamingChunk iterator to raw audio bytes."""
        for chunk in sync_stream:
            yield chunk.audio_bytes

    @router.post("/audio/speech/stream")
    async def speech_stream(request: SpeechRequest, http_request: Request) -> Any:
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="TTS engine not ready")

        fmt = _streaming_format(request.response_format)
        accept = (http_request.headers.get("accept") or "").lower()
        use_sse = "text/event-stream" in accept

        try:
            from voicequant.core.tts.streaming import (
                StreamingSynthesizer,
                TTSStreamingConfig,
            )

            synth = StreamingSynthesizer(
                engine, TTSStreamingConfig(output_format=fmt)
            )
            chunk_iter = synth.stream(request.input, voice=request.voice)
            # Pull the first chunk eagerly so generator-start errors surface
            # here and get mapped to 501/500 instead of streaming a half-open
            # response.
            try:
                first_chunk = next(chunk_iter)
            except StopIteration:
                first_chunk = None
        except ImportError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Synthesis failed: {e}"
            ) from e

        def _prepend_first(first, rest):
            if first is not None:
                yield first
            yield from rest

        combined = _prepend_first(first_chunk, chunk_iter)

        if use_sse:
            return StreamingResponse(
                _sse_generator(combined),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        media_type = _CONTENT_TYPES.get(fmt, "application/octet-stream")
        return StreamingResponse(
            _raw_generator(combined),
            media_type=media_type,
            headers={"Cache-Control": "no-cache"},
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
