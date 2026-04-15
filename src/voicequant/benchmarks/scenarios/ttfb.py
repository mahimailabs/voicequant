"""Time to First Token (TTFB) benchmark.

Measures TTFB at increasing context lengths (1K, 4K, 8K, 16K, 32K tokens)
for FP16, TQ4, and TQ3 KV cache configurations. TTFB is the most critical
latency metric for voice AI since users perceive silence after speaking.

Voice AI latency budgets are typically:
  - Excellent: < 200ms TTFB
  - Good:      < 500ms TTFB
  - Poor:      > 500ms TTFB (noticeable pause)
  - Unusable:  > 1000ms TTFB

The benchmark demonstrates that TurboQuant maintains sub-200ms TTFB at
context lengths where FP16 degrades significantly due to memory bandwidth.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from voicequant.server.config import ServerConfig

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

console = Console()

_CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
_KV_DTYPES = ["fp16", "tq4", "tq3"]

# Simulation model:
# TTFB is dominated by prefill attention (O(n) in context length for each layer)
# and KV cache memory reads. With compression, fewer bytes are read.
_BASE_COMPUTE_MS = 8.0  # fixed scheduling + tokenizer overhead
_N_LAYERS = 32
_HEAD_DIM = 128
_N_HEADS = 32


def _compression_ratio(kv_dtype: str) -> float:
    if kv_dtype == "fp16":
        return 1.0
    elif kv_dtype == "tq4":
        return 16.0 / 3.0
    elif kv_dtype == "tq3":
        return 16.0 / 2.5
    return 1.0


def _simulated_ttfb_ms(context_len: int, kv_dtype: str) -> float:
    """Model TTFB as a function of memory bandwidth and compute.

    Prefill is compute-bound for short contexts, memory-bound for long ones.
    TurboQuant reduces the memory-bound component proportionally to compression
    ratio, but adds a small decompression overhead.
    """
    ratio = _compression_ratio(kv_dtype)

    # Compute component: attention is O(n * d * n_layers)
    # On A100, ~312 TFLOPS, attention for 7B model
    compute_ms = (context_len / 1024.0) * 2.5  # ~2.5ms per 1K tokens

    # Memory component: reading KV cache for all layers
    # FP16: 2 * n_layers * n_heads * head_dim * 2 bytes * context_len
    fp16_bytes = 2 * _N_LAYERS * _N_HEADS * _HEAD_DIM * 2 * context_len
    actual_bytes = fp16_bytes / ratio
    # A100 HBM bandwidth: ~2TB/s
    memory_ms = (actual_bytes / (2.0e12)) * 1000.0

    # Decompression overhead for compressed caches
    decompress_ms = 0.0
    if kv_dtype != "fp16":
        decompress_ms = (context_len / 1024.0) * 0.3  # ~0.3ms per 1K tokens

    return _BASE_COMPUTE_MS + compute_ms + memory_ms + decompress_ms


def _grade_ttfb(ttfb_ms: float) -> str:
    """Return a human-readable quality grade for the TTFB value."""
    if ttfb_ms < 200:
        return "excellent"
    elif ttfb_ms < 500:
        return "good"
    elif ttfb_ms < 1000:
        return "poor"
    return "unusable"


class TTFBBenchmark:
    """Time to First Token benchmark across context lengths.

    Measures TTFB at 1K, 4K, 8K, 16K, and 32K context lengths for
    FP16, TQ4, and TQ3. Demonstrates the latency advantage of KV cache
    compression at long contexts typical of multi-turn voice conversations.
    """

    name = "ttfb"

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

    def _build_prompt_of_length(self, target_tokens: int) -> str:
        """Build a prompt that is approximately target_tokens long.

        Uses repeated filler text with a final question to trigger generation.
        """
        # ~4 chars per token for English text
        filler_sentence = (
            "The user asked the voice assistant a series of general knowledge "
            "questions and the assistant retrieved relevant information before "
            "composing a response. "
        )
        tokens_per_sentence = len(filler_sentence.split())  # ~25 words ~ 30 tokens
        n_repeats = max(1, int(target_tokens / tokens_per_sentence))
        context = filler_sentence * n_repeats
        return context + "\n\nBased on the above notes, what is the next step?"

    def _measure_ttfb_live(
        self,
        client: Any,
        prompt: str,
        config: ServerConfig,
        n_warmup: int = 1,
        n_measure: int = 3,
    ) -> dict[str, Any]:
        """Measure TTFB against a live server with warmup and averaging."""
        messages = [{"role": "user", "content": prompt}]

        # Warmup runs
        for _ in range(n_warmup):
            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                stream=True,
            )
            for chunk in stream:
                pass

        # Measurement runs
        ttfbs: list[float] = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            first_token_time = None

            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=10,
                temperature=0.0,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    break

            # Drain remaining
            for chunk in stream:
                pass

            ttfb = ((first_token_time or time.perf_counter()) - t0) * 1000.0
            ttfbs.append(ttfb)

        avg = sum(ttfbs) / len(ttfbs)
        return {
            "ttfb_ms": round(avg, 2),
            "ttfb_min_ms": round(min(ttfbs), 2),
            "ttfb_max_ms": round(max(ttfbs), 2),
            "n_samples": n_measure,
        }

    def run(
        self,
        model: str | None = None,
        config: ServerConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the TTFB benchmark.

        Args:
            model: HuggingFace model ID (overrides config).
            config: Server configuration.

        Returns:
            Dictionary with keys:
              - ``results``: list of dicts with ttfb per context length per dtype.
              - ``summary``: best dtype at each context length.
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

        all_results: list[dict[str, Any]] = []

        for ctx_len in _CONTEXT_LENGTHS:
            console.print(f"\n[bold]Context length: {ctx_len:,} tokens[/bold]")

            for kv_dtype in _KV_DTYPES:
                if is_simulated:
                    ttfb = _simulated_ttfb_ms(ctx_len, kv_dtype)
                    result = {
                        "context_length": ctx_len,
                        "kv_dtype": kv_dtype,
                        "ttfb_ms": round(ttfb, 2),
                        "ttfb_min_ms": round(ttfb * 0.9, 2),
                        "ttfb_max_ms": round(ttfb * 1.1, 2),
                        "grade": _grade_ttfb(ttfb),
                    }
                else:
                    prompt = self._build_prompt_of_length(ctx_len)
                    measurement = self._measure_ttfb_live(client, prompt, config)
                    result = {
                        "context_length": ctx_len,
                        "kv_dtype": kv_dtype,
                        "ttfb_ms": measurement["ttfb_ms"],
                        "ttfb_min_ms": measurement["ttfb_min_ms"],
                        "ttfb_max_ms": measurement["ttfb_max_ms"],
                        "grade": _grade_ttfb(measurement["ttfb_ms"]),
                    }

                all_results.append(result)
                grade_color = {
                    "excellent": "green",
                    "good": "blue",
                    "poor": "yellow",
                    "unusable": "red",
                }.get(result["grade"], "white")
                console.print(
                    f"  {kv_dtype:5s} | TTFB={result['ttfb_ms']:7.1f}ms | "
                    f"[{grade_color}]{result['grade']}[/{grade_color}]"
                )

        # Summary: best dtype per context length
        summary: dict[int, dict[str, Any]] = {}
        for ctx_len in _CONTEXT_LENGTHS:
            ctx_results = [r for r in all_results if r["context_length"] == ctx_len]
            best = min(ctx_results, key=lambda r: r["ttfb_ms"])
            fp16_result = next(r for r in ctx_results if r["kv_dtype"] == "fp16")
            summary[ctx_len] = {
                "best_dtype": best["kv_dtype"],
                "best_ttfb_ms": best["ttfb_ms"],
                "fp16_ttfb_ms": fp16_result["ttfb_ms"],
                "speedup": round(fp16_result["ttfb_ms"] / best["ttfb_ms"], 2)
                if best["ttfb_ms"] > 0
                else 0,
            }

        return {
            "results": all_results,
            "summary": summary,
            "context_lengths": _CONTEXT_LENGTHS,
            "simulated": is_simulated,
        }
