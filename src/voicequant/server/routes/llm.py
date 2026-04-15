"""LLM chat/completion routes."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[ChatCompletionChoice]
    usage: dict[str, int] | None = None


def build_router(config, get_engine) -> APIRouter:
    """Build the LLM router. `get_engine` returns the active LLM engine (or None)."""
    router = APIRouter(prefix="/v1", tags=["llm"])

    @router.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        engine = get_engine()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if request.stream:

            async def stream_response():
                async for chunk in engine.generate(
                    messages=messages,
                    max_tokens=request.max_tokens or config.default_max_tokens,
                    temperature=request.temperature
                    if request.temperature is not None
                    else config.temperature,
                    stream=True,
                    request_id=request_id,
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = None
        async for chunk in engine.generate(
            messages=messages,
            max_tokens=request.max_tokens or config.default_max_tokens,
            temperature=request.temperature
            if request.temperature is not None
            else config.temperature,
            stream=False,
            request_id=request_id,
        ):
            result = chunk
        return JSONResponse(content=result)

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model,
                    "object": "model",
                    "owned_by": "voicequant",
                    "permission": [],
                }
            ],
        }

    return router
