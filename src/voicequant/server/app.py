"""FastAPI application with OpenAI-compatible endpoints for voice AI inference.

Routes are split into per-modality routers under server/routes/. STT and TTS
routers are mounted conditionally: if the real engine modules exist the real
router is mounted, otherwise a stub router returning HTTP 501 is mounted.
"""

from __future__ import annotations

from typing import Any

from voicequant.server.config import GPU_CAPACITY_ESTIMATES, ServerConfig  # noqa: F401


def create_app(config: ServerConfig) -> Any:
    """Create the FastAPI application with all endpoints."""
    from fastapi import FastAPI

    from voicequant.server.engine import EngineRegistry
    from voicequant.server.metrics import MetricsRegistry
    from voicequant.server.routes import capacity as capacity_routes
    from voicequant.server.routes import llm as llm_routes

    app = FastAPI(
        title="VoiceQuant",
        description="OpenAI-compatible inference server with TurboQuant KV cache compression",
        version="0.1.0",
    )

    engine = None
    registry = EngineRegistry()
    metrics_registry = MetricsRegistry()

    def _get_engine():
        return engine

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal engine
        from voicequant.server.engine import VoiceQuantEngine

        engine = VoiceQuantEngine(config)
        await engine.initialize()
        registry.register_engine("llm", engine)
        metrics_registry.register_modality("llm", engine.metrics)

    # Always-on routers
    app.include_router(llm_routes.build_router(config, _get_engine))
    app.include_router(
        capacity_routes.build_router(config, _get_engine, metrics_registry)
    )

    # Conditional STT mount: real engine if present, else stub
    try:
        from voicequant.core.stt.engine import STTEngine  # noqa: F401
        from voicequant.server.routes.stt_real import (
            router as stt_router,  # type: ignore
        )
    except ImportError:
        from voicequant.server.routes.stt import router as stt_router
    app.include_router(stt_router)

    # Conditional TTS mount
    try:
        from voicequant.core.tts.engine import TTSEngine  # noqa: F401
        from voicequant.server.routes.tts_real import (
            router as tts_router,  # type: ignore
        )
    except ImportError:
        from voicequant.server.routes.tts import router as tts_router
    app.include_router(tts_router)

    # Expose on app.state for tests / introspection
    app.state.registry = registry
    app.state.metrics_registry = metrics_registry

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
    """Start the VoiceQuant inference server."""
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

    console.print("\n[bold]VoiceQuant Server[/bold]")
    console.print(f"Model: {model}")
    console.print(f"KV cache: TQ{tq_bits} (residual window: {tq_residual_window})")
    console.print(f"Max concurrent: {max_concurrent}")
    console.print(f"GPU memory: {gpu_memory:.0%}")
    console.print(f"Endpoint: http://{host}:{port}/v1/chat/completions\n")

    fastapi_app = create_app(config)
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
