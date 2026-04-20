"""Health, capacity, kv-stats, and Prometheus metrics routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from voicequant.server.config import GPU_CAPACITY_ESTIMATES


def build_router(config, get_engine, metrics_registry) -> APIRouter:
    router = APIRouter(tags=["capacity"])

    @router.get("/v1/health")
    async def health_check() -> dict[str, Any]:
        engine = get_engine()
        if engine:
            return engine.get_health()
        return {"status": "starting", "model": config.model}

    @router.get("/v1/capacity")
    async def capacity() -> dict[str, Any]:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                mem_free = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                ) / (1024**3)
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

    @router.get("/v1/kv-stats")
    async def kv_stats() -> dict[str, Any]:
        engine = get_engine()
        metrics = engine.get_metrics() if engine else {}
        return {
            "kv_cache_dtype": config.kv_cache_dtype,
            "residual_window": config.residual_window,
            "active_requests": metrics.get("request_count", 0),
            "total_tokens_generated": metrics.get("total_tokens", 0),
        }

    @router.get("/metrics")
    async def prometheus_metrics() -> Any:
        engine = get_engine()
        metrics = engine.get_metrics() if engine else {}

        lines = [
            "# HELP voicequant_requests_total Total requests processed",
            "# TYPE voicequant_requests_total counter",
            f"voicequant_requests_total {metrics.get('request_count', 0)}",
            "# HELP voicequant_tokens_total Total tokens generated",
            "# TYPE voicequant_tokens_total counter",
            f"voicequant_tokens_total {metrics.get('total_tokens', 0)}",
            "# HELP voicequant_uptime_seconds Server uptime in seconds",
            "# TYPE voicequant_uptime_seconds gauge",
            f"voicequant_uptime_seconds {metrics.get('uptime_seconds', 0):.1f}",
            "# HELP voicequant_kv_cache_bits KV cache quantization bits",
            "# TYPE voicequant_kv_cache_bits gauge",
            f"voicequant_kv_cache_bits {config.kv_cache_dtype.replace('tq', '') if 'tq' in config.kv_cache_dtype else '16'}",
        ]

        # Append per-modality registered metrics
        if metrics_registry is not None:
            extra = metrics_registry.collect_all()
            for key, val in extra.items():
                lines.append(f"# HELP voicequant_{key} Registered modality metric")
                lines.append(f"# TYPE voicequant_{key} gauge")
                lines.append(f"voicequant_{key} {val}")

        return PlainTextResponse("\n".join(lines) + "\n")

    return router
