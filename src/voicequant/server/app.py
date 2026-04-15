"""FastAPI application with OpenAI-compatible endpoints for voice AI inference.

Provides standard OpenAI API endpoints plus voice-specific extensions
for capacity estimation and KV cache statistics.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from voicequant.server.config import ServerConfig, GPU_CAPACITY_ESTIMATES


# --- Request/Response Models ---

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int = 0
    message: ChatMessage | None = None
    delta: dict[str, str] | None = None
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[ChatCompletionChoice]
    usage: dict[str, int] | None = None


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    owned_by: str = "voicequant"


# --- Application ---

def create_app(config: ServerConfig) -> Any:
    """Create the FastAPI application with all endpoints.

    Args:
        config: Server configuration.

    Returns:
        FastAPI application instance.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse

    app = FastAPI(
        title="VoiceQuant",
        description="OpenAI-compatible inference server with TurboQuant KV cache compression",
        version="0.1.0",
    )

    engine = None

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal engine
        from voicequant.server.engine import VoiceQuantEngine
        engine = VoiceQuantEngine(config)
        await engine.initialize()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        """OpenAI-compatible chat completions endpoint with streaming support."""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if request.stream:
            async def stream_response():
                async for chunk in engine.generate(
                    messages=messages,
                    max_tokens=request.max_tokens or config.default_max_tokens,
                    temperature=request.temperature if request.temperature is not None else config.temperature,
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
            temperature=request.temperature if request.temperature is not None else config.temperature,
            stream=False,
            request_id=request_id,
        ):
            result = chunk

        return JSONResponse(content=result)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """List available models."""
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

    @app.get("/v1/health")
    async def health_check() -> dict[str, Any]:
        """Health check with GPU memory status."""
        if engine:
            return engine.get_health()
        return {"status": "starting", "model": config.model}

    @app.get("/v1/capacity")
    async def capacity() -> dict[str, Any]:
        """Estimate how many concurrent sessions can fit in remaining VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                mem_free = (torch.cuda.get_device_properties(0).total_mem
                            - torch.cuda.memory_allocated()) / (1024 ** 3)
                # Rough estimate: ~30MB per session at 4K context with TQ4
                tq4_sessions = int(mem_free * 1024 / 30)
                fp16_sessions = int(mem_free * 1024 / 150)
                return {
                    "gpu": gpu_name,
                    "free_memory_gb": round(mem_free, 2),
                    "estimated_sessions_tq4": tq4_sessions,
                    "estimated_sessions_fp16": fp16_sessions,
                    "kv_cache_dtype": config.kv_cache_dtype,
                }
        except Exception:
            pass

        return {
            "gpu": "unknown",
            "estimated_sessions": GPU_CAPACITY_ESTIMATES,
            "kv_cache_dtype": config.kv_cache_dtype,
        }

    @app.get("/v1/kv-stats")
    async def kv_stats() -> dict[str, Any]:
        """Current KV cache memory usage, compression ratio, active sessions."""
        metrics = engine.get_metrics() if engine else {}
        return {
            "kv_cache_dtype": config.kv_cache_dtype,
            "residual_window": config.residual_window,
            "active_requests": metrics.get("request_count", 0),
            "total_tokens_generated": metrics.get("total_tokens", 0),
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Any:
        """Prometheus-format metrics endpoint."""
        from fastapi.responses import PlainTextResponse

        metrics = engine.get_metrics() if engine else {}
        lines = [
            "# HELP voicequant_requests_total Total requests processed",
            "# TYPE voicequant_requests_total counter",
            f'voicequant_requests_total {metrics.get("request_count", 0)}',
            "# HELP voicequant_tokens_total Total tokens generated",
            "# TYPE voicequant_tokens_total counter",
            f'voicequant_tokens_total {metrics.get("total_tokens", 0)}',
            "# HELP voicequant_uptime_seconds Server uptime in seconds",
            "# TYPE voicequant_uptime_seconds gauge",
            f'voicequant_uptime_seconds {metrics.get("uptime_seconds", 0):.1f}',
            "# HELP voicequant_kv_cache_bits KV cache quantization bits",
            "# TYPE voicequant_kv_cache_bits gauge",
            f'voicequant_kv_cache_bits {config.kv_cache_dtype.replace("tq", "") if "tq" in config.kv_cache_dtype else "16"}',
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app


def start_server(
    model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    host: str = "0.0.0.0",
    port: int = 8000,
    tq_bits: int = 4,
    tq_residual_window: int = 256,
    max_concurrent: int = 64,
    gpu_memory: float = 0.90,
) -> None:
    """Start the VoiceQuant inference server.

    Args:
        model: HuggingFace model ID or path.
        host: Server bind host.
        port: Server bind port.
        tq_bits: TurboQuant quantization bits (3 or 4).
        tq_residual_window: FP16 residual window size.
        max_concurrent: Maximum concurrent sequences.
        gpu_memory: GPU memory utilization (0.0-1.0).
    """
    import uvicorn
    from rich.console import Console

    console = Console()

    config = ServerConfig(
        model=model,
        host=host,
        port=port,
        kv_cache_dtype=f"tq{tq_bits}",
        residual_window=tq_residual_window,
        max_num_seqs=max_concurrent,
        gpu_memory_utilization=gpu_memory,
    )

    console.print(f"\n[bold]VoiceQuant Server[/bold]")
    console.print(f"Model: {model}")
    console.print(f"KV cache: TQ{tq_bits} (residual window: {tq_residual_window})")
    console.print(f"Max concurrent: {max_concurrent}")
    console.print(f"GPU memory: {gpu_memory:.0%}")
    console.print(f"Endpoint: http://{host}:{port}/v1/chat/completions\n")

    fastapi_app = create_app(config)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
