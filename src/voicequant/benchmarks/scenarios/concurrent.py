"""Concurrent session scaling benchmark.

Measures how VoiceQuant performance degrades as concurrent voice sessions
increase. Starts at 1 session and scales up through 5, 10, 20, 30, 40, 50
sessions. Each session runs a simulated 5-turn conversation.

Key metrics per concurrency level:
  - Per-session TTFB (p50, p95, p99)
  - Tokens/sec throughput
  - Total GPU memory usage
  - Breaking point: the concurrency level where p95 TTFB exceeds 500ms

When no live server is available, results are simulated using the analytical
GPU capacity model from ``voicequant.server.config.GPU_CAPACITY_ESTIMATES``.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from voicequant.server.config import GPU_CAPACITY_ESTIMATES, ServerConfig

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

console = Console()

# ---------------------------------------------------------------------------
# Simulated conversation for each session
# ---------------------------------------------------------------------------

_SESSION_MESSAGES = [
    {"role": "user", "content": "Hi, I need to book an appointment."},
    {"role": "user", "content": "How about next Tuesday at 3?"},
    {"role": "user", "content": "My name is Alex Rivera."},
    {"role": "user", "content": "Does my insurance cover this visit?"},
    {"role": "user", "content": "Great, book it. Thanks!"},
]

_CONCURRENCY_LEVELS = [1, 5, 10, 20, 30, 40, 50]

# Simulation constants
_TOKENS_PER_SESSION = 200  # ~5 turns * 40 tokens avg
_BASE_TTFB_MS = 18.0
_A100_MEMORY_GB = 80.0
_MODEL_MEMORY_GB = 4.0
_FP16_KV_PER_SESSION_MB = 32.0  # 4K context, 7B model
_TQ4_COMPRESSION = 5.33  # 16/3


def _detect_gpu() -> str:
    """Detect GPU type, defaulting to A100 for simulation."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).upper()
            for gpu in GPU_CAPACITY_ESTIMATES:
                if gpu in name:
                    return gpu
    except ImportError:
        pass
    return "A100"


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


class ConcurrentBenchmark:
    """Concurrent voice session scaling benchmark.

    Scales from 1 to max_sessions concurrent sessions, measuring latency
    and throughput at each level. Identifies the concurrency breaking
    point where p95 TTFB exceeds the 500ms voice latency budget.
    """

    name = "concurrent"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _get_client(self, config: ServerConfig) -> Any | None:
        if not _HAS_OPENAI:
            return None
        try:
            client = openai.OpenAI(
                base_url=f"http://{config.host}:{config.port}/v1",
                api_key="not-needed",
            )
            client.models.list()
            return client
        except Exception:
            return None

    def _run_session_live(
        self,
        client: Any,
        config: ServerConfig,
        session_id: int,
    ) -> dict[str, Any]:
        """Run a 5-turn session against a live server."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Keep responses brief.",
            },
        ]
        ttfbs: list[float] = []
        total_tokens = 0
        t_start = time.perf_counter()

        for user_msg in _SESSION_MESSAGES:
            messages.append(user_msg)
            t0 = time.perf_counter()
            first_token_time = None
            response_text = ""

            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.default_max_tokens,
                temperature=config.temperature,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    total_tokens += 1
                    response_text += chunk.choices[0].delta.content

            ttfb = ((first_token_time or time.perf_counter()) - t0) * 1000.0
            ttfbs.append(ttfb)
            messages.append({"role": "assistant", "content": response_text})

        t_end = time.perf_counter()
        elapsed = t_end - t_start

        return {
            "session_id": session_id,
            "ttfb_p50_ms": round(_percentile(ttfbs, 50), 2),
            "ttfb_p95_ms": round(_percentile(ttfbs, 95), 2),
            "ttfb_p99_ms": round(_percentile(ttfbs, 99), 2),
            "tokens_generated": total_tokens,
            "elapsed_s": round(elapsed, 2),
            "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
        }

    def _simulate_concurrency_level(
        self,
        n_sessions: int,
        kv_dtype: str,
        gpu: str,
    ) -> dict[str, Any]:
        """Simulate metrics for a given concurrency level."""
        gpu_info = GPU_CAPACITY_ESTIMATES.get(gpu, GPU_CAPACITY_ESTIMATES["A100"])
        max_sessions_key = "tq4_sessions" if kv_dtype == "tq4" else "fp16_sessions"
        capacity = gpu_info[max_sessions_key]

        # Load factor determines how much latency degrades
        load_factor = n_sessions / capacity
        overloaded = load_factor > 1.0

        # Memory
        if kv_dtype == "fp16":
            kv_per_session_mb = _FP16_KV_PER_SESSION_MB
        else:
            kv_per_session_mb = _FP16_KV_PER_SESSION_MB / _TQ4_COMPRESSION

        total_kv_mb = kv_per_session_mb * n_sessions
        total_memory_gb = _MODEL_MEMORY_GB + total_kv_mb / 1024.0

        # TTFB degrades non-linearly with load
        if overloaded:
            # Queuing delay dominates
            queue_factor = (load_factor - 1.0) * 500.0
            base = _BASE_TTFB_MS * (1.0 + load_factor * 0.3)
            ttfb_p50 = base + queue_factor * 0.5
            ttfb_p95 = base + queue_factor * 1.5
            ttfb_p99 = base + queue_factor * 2.5
        else:
            base = _BASE_TTFB_MS * (1.0 + load_factor * 0.3)
            ttfb_p50 = base
            ttfb_p95 = base * (1.0 + load_factor * 0.5)
            ttfb_p99 = base * (1.0 + load_factor * 0.8)

        # Tokens/sec: ~50 tok/s baseline per session, degrades at high load
        if overloaded:
            tps = max(5.0, 50.0 / load_factor)
        else:
            tps = 50.0 * (1.0 - load_factor * 0.2)

        return {
            "n_sessions": n_sessions,
            "kv_dtype": kv_dtype,
            "gpu": gpu,
            "ttfb_p50_ms": round(ttfb_p50, 2),
            "ttfb_p95_ms": round(ttfb_p95, 2),
            "ttfb_p99_ms": round(ttfb_p99, 2),
            "tokens_per_sec_per_session": round(tps, 2),
            "total_gpu_memory_gb": round(total_memory_gb, 2),
            "kv_memory_gb": round(total_kv_mb / 1024.0, 2),
            "load_factor": round(load_factor, 3),
            "overloaded": overloaded,
        }

    def run(
        self,
        model: str | None = None,
        config: ServerConfig | None = None,
        max_sessions: int = 50,
    ) -> dict[str, Any]:
        """Execute the concurrent session scaling benchmark.

        Args:
            model: HuggingFace model ID (overrides config).
            config: Server configuration.
            max_sessions: Maximum number of concurrent sessions to test.

        Returns:
            Dictionary with keys:
              - ``results``: list of dicts per concurrency level per kv_dtype.
              - ``breaking_point``: dict mapping kv_dtype to the session count
                where p95 TTFB first exceeds 500ms.
              - ``simulated``: whether results are simulated.
        """
        if config is None:
            config = ServerConfig()
        if model:
            config = config.model_copy(update={"model": model})

        client = self._get_client(config)
        is_simulated = client is None

        if is_simulated:
            console.print(
                "[yellow]No live server detected -- using simulated results.[/yellow]"
            )
        else:
            console.print(
                f"[green]Connected to live server at {config.host}:{config.port}[/green]"
            )

        levels = [n for n in _CONCURRENCY_LEVELS if n <= max_sessions]
        kv_dtypes = ["fp16", "tq4"]
        gpu = _detect_gpu()

        all_results: list[dict[str, Any]] = []
        breaking_points: dict[str, int | None] = {}

        for kv_dtype in kv_dtypes:
            console.print(f"\n[bold]Scaling test: {kv_dtype} on {gpu}[/bold]")
            breaking_point: int | None = None

            for n in levels:
                if is_simulated:
                    result = self._simulate_concurrency_level(n, kv_dtype, gpu)
                else:
                    # For live testing, run sessions concurrently using threads
                    import concurrent.futures

                    session_results: list[dict[str, Any]] = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                        futures = [
                            pool.submit(self._run_session_live, client, config, i)
                            for i in range(n)
                        ]
                        for f in concurrent.futures.as_completed(futures):
                            session_results.append(f.result())

                    all_ttfb_p95 = [s["ttfb_p95_ms"] for s in session_results]
                    all_tps = [s["tokens_per_sec"] for s in session_results]

                    result = {
                        "n_sessions": n,
                        "kv_dtype": kv_dtype,
                        "gpu": gpu,
                        "ttfb_p50_ms": round(
                            _percentile(
                                [s["ttfb_p50_ms"] for s in session_results], 50
                            ),
                            2,
                        ),
                        "ttfb_p95_ms": round(_percentile(all_ttfb_p95, 95), 2),
                        "ttfb_p99_ms": round(
                            _percentile(
                                [s["ttfb_p99_ms"] for s in session_results], 99
                            ),
                            2,
                        ),
                        "tokens_per_sec_per_session": round(
                            sum(all_tps) / len(all_tps), 2
                        )
                        if all_tps
                        else 0,
                        "total_gpu_memory_gb": 0,  # would need nvidia-smi
                        "kv_memory_gb": 0,
                        "load_factor": 0,
                        "overloaded": False,
                    }

                all_results.append(result)

                p95 = result["ttfb_p95_ms"]
                status = "[red]OVER 500ms[/red]" if p95 > 500 else "[green]OK[/green]"
                console.print(
                    f"  {n:3d} sessions | p95 TTFB={p95:7.1f}ms {status} | "
                    f"mem={result.get('total_gpu_memory_gb', 0):.1f}GB"
                )

                if breaking_point is None and p95 > 500.0:
                    breaking_point = n

            breaking_points[kv_dtype] = breaking_point

        return {
            "results": all_results,
            "breaking_point": breaking_points,
            "gpu": gpu,
            "simulated": is_simulated,
        }
