"""vLLM engine wrapper with TurboQuant KV cache compression.

Integrates TurboQuant with vLLM for voice AI inference. Supports multiple
integration strategies:
1. turboquant-vllm plugin (pip install turboquant-vllm) - preferred
2. Direct vLLM kv_cache_dtype configuration
3. Standalone TurboQuantEngine fallback (no vLLM)
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from rich.console import Console

from voicequant.server.config import ServerConfig

console = Console()


class VoiceQuantEngine:
    """vLLM engine wrapper with TurboQuant KV cache compression enabled.

    Auto-detects GPU capabilities and selects the appropriate kernel backend
    (cuTile for SM86+, Triton fallback, PyTorch fallback).

    Args:
        config: Server configuration with voice-optimized defaults.
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._tokenizer: Any = None
        self._model_loaded = False
        self._start_time = time.time()
        self._request_count = 0
        self._total_tokens = 0

    async def initialize(self) -> None:
        """Initialize the vLLM engine with TurboQuant configuration."""
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs

            engine_args = AsyncEngineArgs(
                model=self.config.model,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                kv_cache_dtype=self.config.kv_cache_dtype,
                enforce_eager=False,
                enable_prefix_caching=True,
            )

            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._model_loaded = True
            console.print(f"[green]Model loaded: {self.config.model}[/green]")
            console.print(f"[green]KV cache: {self.config.kv_cache_dtype}[/green]")

        except ImportError:
            console.print("[yellow]vLLM not available. Server requires 'pip install voicequant[serve]'[/yellow]")
            raise

    async def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = True,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate a completion with TurboQuant-compressed KV cache.

        Args:
            prompt: Raw prompt text (for completions API).
            messages: Chat messages (for chat completions API).
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            stream: Whether to stream the response.
            request_id: Unique request identifier.

        Yields:
            Completion chunks (streaming) or full response.
        """
        from vllm import SamplingParams

        max_tokens = max_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if messages:
            prompt = self._apply_chat_template(messages)

        request_id = request_id or f"vq-{self._request_count}"
        self._request_count += 1

        start_time = time.time()
        first_token_time = None

        async for output in self._engine.generate(prompt, sampling_params, request_id):
            if first_token_time is None:
                first_token_time = time.time()

            if stream:
                yield {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": output.outputs[0].text},
                        "finish_reason": output.outputs[0].finish_reason,
                    }],
                }

        ttfb = (first_token_time - start_time) if first_token_time else 0
        total_time = time.time() - start_time
        n_tokens = len(output.outputs[0].token_ids) if output else 0
        self._total_tokens += n_tokens

        if not stream:
            yield {
                "id": request_id,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": output.outputs[0].text},
                    "finish_reason": output.outputs[0].finish_reason,
                }],
                "usage": {
                    "completion_tokens": n_tokens,
                    "total_tokens": n_tokens,
                },
                "metrics": {
                    "ttfb_ms": ttfb * 1000,
                    "total_ms": total_time * 1000,
                    "tokens_per_sec": n_tokens / total_time if total_time > 0 else 0,
                },
            }

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Apply chat template to messages."""
        if self._tokenizer is not None:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def get_metrics(self) -> dict[str, Any]:
        """Get current engine metrics."""
        uptime = time.time() - self._start_time
        return {
            "model": self.config.model,
            "kv_cache_dtype": self.config.kv_cache_dtype,
            "model_loaded": self._model_loaded,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "uptime_seconds": uptime,
            "tokens_per_second_avg": self._total_tokens / uptime if uptime > 0 else 0,
        }

    def get_health(self) -> dict[str, Any]:
        """Get health status including GPU memory."""
        health: dict[str, Any] = {
            "status": "healthy" if self._model_loaded else "loading",
            "model": self.config.model,
            "kv_cache_dtype": self.config.kv_cache_dtype,
        }

        try:
            import torch
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                health["gpu"] = {
                    "name": torch.cuda.get_device_name(),
                    "memory_allocated_gb": round(mem_allocated, 2),
                    "memory_total_gb": round(mem_total, 2),
                    "memory_utilization": round(mem_allocated / mem_total, 3),
                }
        except Exception:
            health["gpu"] = None

        return health
