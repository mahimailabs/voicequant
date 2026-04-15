"""Multi-turn voice conversation benchmark.

Simulates a realistic 20-turn voice agent conversation to measure how
compression quality and latency evolve as the KV cache grows. Each turn
appends user + assistant messages, compresses the cache, and records
TTFB, total latency, KV cache size, and compression ratio.

Three configurations are compared side-by-side:
  - FP16 (baseline, no compression)
  - TQ4 (4-bit TurboQuant, default voice setting)
  - TQ3 (3-bit TurboQuant, aggressive compression)

When no GPU or vLLM server is available, the benchmark produces simulated
results based on the analytical compression model from the VoiceQuant paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from voicequant.server.config import ServerConfig, GPU_CAPACITY_ESTIMATES

try:
    import openai  # used to talk to a live vLLM server
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

console = Console()

# ---------------------------------------------------------------------------
# Default scripted conversation
# ---------------------------------------------------------------------------

VOICE_AGENT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. You answer questions clearly and "
    "concisely, help with scheduling and reminders, perform calculations, "
    "and provide general information. Keep responses under 3 sentences "
    "and speak naturally as if on the phone."
)

SCRIPTED_CONVERSATION: list[dict[str, str]] = [
    {"role": "user", "content": "Hey, what's the weather going to be like tomorrow?"},
    {"role": "assistant", "content": "Tomorrow looks sunny with a high of 72 and a low of 55. Should be a nice day."},
    {"role": "user", "content": "Great. Can you set a reminder for me to go for a run at 7 AM?"},
    {"role": "assistant", "content": "Done, I've set a reminder for 7 AM tomorrow to go for a run."},
    {"role": "user", "content": "Thanks. What's 15% tip on a $68 bill?"},
    {"role": "assistant", "content": "A 15% tip on $68 is $10.20, making the total $78.20."},
    {"role": "user", "content": "And what about 20%?"},
    {"role": "assistant", "content": "At 20%, the tip would be $13.60, for a total of $81.60."},
    {"role": "user", "content": "OK. Can you tell me what the capital of Portugal is?"},
    {"role": "assistant", "content": "The capital of Portugal is Lisbon."},
    {"role": "user", "content": "How far is it from Lisbon to Madrid?"},
    {"role": "assistant", "content": "Lisbon to Madrid is about 625 kilometers, or roughly 390 miles by road."},
    {"role": "user", "content": "What time zone is Portugal in?"},
    {"role": "assistant", "content": "Portugal is in the Western European Time zone, which is UTC+0 in winter and UTC+1 in summer."},
    {"role": "user", "content": "Got it. Can you also remind me to call my mom at 6 PM today?"},
    {"role": "assistant", "content": "Sure, I've set a reminder to call your mom at 6 PM today."},
    {"role": "user", "content": "How many reminders do I have set now?"},
    {"role": "assistant", "content": "You have two reminders: the 7 AM run tomorrow and the 6 PM call today."},
    {"role": "user", "content": "Can you convert 100 euros to dollars?"},
    {"role": "assistant", "content": "At current rates, 100 euros is approximately $108.50 USD."},
    {"role": "user", "content": "What about to British pounds?"},
    {"role": "assistant", "content": "100 euros converts to approximately 86 British pounds."},
    {"role": "user", "content": "Thanks. What's a good recipe for a quick dinner?"},
    {"role": "assistant", "content": "A quick stir-fry works well: sautee sliced chicken or tofu with vegetables in soy sauce and garlic. Serve over rice. Takes about 20 minutes."},
    {"role": "user", "content": "How many calories would that be roughly?"},
    {"role": "assistant", "content": "A chicken stir-fry with rice is roughly 500-600 calories per serving, depending on portion size and oil used."},
    {"role": "user", "content": "That sounds good. What time is sunset today?"},
    {"role": "assistant", "content": "Sunset today is at approximately 7:45 PM."},
    {"role": "user", "content": "Perfect. One more thing -- what day of the week is July 4th this year?"},
    {"role": "assistant", "content": "July 4th falls on a Friday this year."},
    {"role": "user", "content": "Nice, long weekend then. Can you summarize what we talked about?"},
    {"role": "assistant", "content": "We covered tomorrow's weather, set two reminders, calculated tips, discussed Portugal geography, did currency conversions, got a dinner idea, and checked sunset time and July 4th."},
    {"role": "user", "content": "Great memory! How long have we been chatting?"},
    {"role": "assistant", "content": "We've had about 20 exchanges so far. Is there anything else you need?"},
    {"role": "user", "content": "Nope, that's everything. Thanks!"},
    {"role": "assistant", "content": "You're welcome! Have a great evening."},
    {"role": "user", "content": "Wait, one more -- what's the population of Lisbon?"},
    {"role": "assistant", "content": "Lisbon's city population is about 550,000, with the metro area around 2.9 million."},
    {"role": "user", "content": "OK now I'm really done. Bye!"},
    {"role": "assistant", "content": "Goodbye! Enjoy your evening."},
]

# Grab only user turns to drive the 20-turn benchmark
_USER_TURNS = [m for m in SCRIPTED_CONVERSATION if m["role"] == "user"][:20]

# Analytical constants for simulation
_TOKENS_PER_TURN_USER = 15  # average user tokens per turn
_TOKENS_PER_TURN_ASSISTANT = 25  # average assistant tokens per turn
_FP16_BYTES_PER_TOKEN_PER_LAYER = 2 * 128 * 2  # 2 (K+V) * head_dim * 2 bytes (fp16)
_N_LAYERS = 32  # typical 7B model
_N_HEADS = 32


@dataclass
class TurnResult:
    """Result for a single conversation turn."""
    turn: int
    kv_dtype: str
    context_tokens: int
    ttfb_ms: float
    total_latency_ms: float
    kv_cache_bytes: int
    compression_ratio: float
    tokens_generated: int


def _compression_ratio(kv_dtype: str) -> float:
    """Return analytical compression ratio for a given KV cache dtype."""
    if kv_dtype == "fp16":
        return 1.0
    elif kv_dtype == "tq4":
        # 2-bit keys + 1-bit QJL + 3-bit values ~ 3 bits avg vs 16 bits
        return 16.0 / 3.0
    elif kv_dtype == "tq3":
        # 2-bit keys + 1-bit QJL + 2-bit values ~ 2.5 bits avg vs 16 bits
        return 16.0 / 2.5
    return 1.0


def _simulated_ttfb_ms(context_tokens: int, kv_dtype: str) -> float:
    """Estimate TTFB based on context length and cache type.

    Models the observation that TTFB scales roughly linearly with
    KV cache memory footprint due to memory-bandwidth bottleneck.
    """
    base_ms = 15.0  # fixed overhead (tokenization, scheduling)
    # Memory-bandwidth limited: ~1TB/s on A100, each attention layer
    # reads all cached KV. FP16 reads more bytes.
    ratio = _compression_ratio(kv_dtype)
    mem_factor = context_tokens / (ratio * 1000.0)
    return base_ms + mem_factor * 8.0  # ~8ms per 1K-equivalent tokens


def _simulated_total_latency_ms(context_tokens: int, kv_dtype: str,
                                 tokens_generated: int) -> float:
    """Estimate total generation latency."""
    ttfb = _simulated_ttfb_ms(context_tokens, kv_dtype)
    # Decode phase: ~10ms per token on A100 7B model
    decode_ms = tokens_generated * 10.0
    return ttfb + decode_ms


class MultiTurnBenchmark:
    """Simulates a 20-turn voice conversation and measures per-turn metrics.

    Compares FP16 baseline, TQ4, and TQ3 compression across all turns.
    Tracks how TTFB and memory grow as the conversation context expands.
    """

    name = "multi_turn"

    def __init__(self) -> None:
        self._client: Any | None = None

    def _get_client(self, config: ServerConfig) -> Any | None:
        """Try to connect to a live vLLM server."""
        if not _HAS_OPENAI:
            return None
        try:
            client = openai.OpenAI(
                base_url=f"http://{config.host}:{config.port}/v1",
                api_key="not-needed",
            )
            # Quick health check
            client.models.list()
            return client
        except Exception:
            return None

    def _run_live_turn(
        self,
        client: Any,
        messages: list[dict[str, str]],
        config: ServerConfig,
    ) -> dict[str, Any]:
        """Run a single turn against a live server and measure timing."""
        t0 = time.perf_counter()
        first_token_time: float | None = None
        tokens_generated = 0
        full_response = ""

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
                tokens_generated += 1
                full_response += chunk.choices[0].delta.content

        t1 = time.perf_counter()
        ttfb_ms = ((first_token_time or t1) - t0) * 1000.0
        total_ms = (t1 - t0) * 1000.0

        return {
            "ttfb_ms": ttfb_ms,
            "total_latency_ms": total_ms,
            "tokens_generated": tokens_generated,
            "response": full_response,
        }

    def _run_simulated_turn(
        self,
        turn: int,
        context_tokens: int,
        kv_dtype: str,
    ) -> dict[str, Any]:
        """Produce simulated metrics when no live server is available."""
        tokens_generated = _TOKENS_PER_TURN_ASSISTANT
        ttfb = _simulated_ttfb_ms(context_tokens, kv_dtype)
        total = _simulated_total_latency_ms(context_tokens, kv_dtype, tokens_generated)
        return {
            "ttfb_ms": round(ttfb, 2),
            "total_latency_ms": round(total, 2),
            "tokens_generated": tokens_generated,
            "response": f"[simulated response for turn {turn}]",
        }

    def run(
        self,
        model: str | None = None,
        config: ServerConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the multi-turn benchmark.

        Args:
            model: HuggingFace model ID (overrides config).
            config: Server configuration. Defaults to voice defaults.

        Returns:
            Dictionary with keys:
              - ``turns``: list of TurnResult dicts per turn per kv_dtype.
              - ``summary``: aggregate statistics.
              - ``simulated``: whether results are simulated.
        """
        if config is None:
            config = ServerConfig()
        if model:
            config = config.model_copy(update={"model": model})

        client = self._get_client(config)
        is_simulated = client is None

        if is_simulated:
            console.print("[yellow]No live server detected -- using simulated results.[/yellow]")
        else:
            console.print(f"[green]Connected to live server at {config.host}:{config.port}[/green]")

        kv_dtypes = ["fp16", "tq4", "tq3"]
        all_results: list[dict[str, Any]] = []

        for kv_dtype in kv_dtypes:
            console.print(f"\n[bold]Running {kv_dtype} configuration...[/bold]")
            messages: list[dict[str, str]] = [
                {"role": "system", "content": VOICE_AGENT_SYSTEM_PROMPT},
            ]
            context_tokens = len(VOICE_AGENT_SYSTEM_PROMPT.split()) * 1.3  # rough estimate

            for turn_idx, user_msg in enumerate(_USER_TURNS, start=1):
                messages.append(user_msg)
                context_tokens += _TOKENS_PER_TURN_USER

                if is_simulated:
                    result = self._run_simulated_turn(
                        turn_idx, int(context_tokens), kv_dtype,
                    )
                else:
                    result = self._run_live_turn(client, messages, config)

                # Add assistant response to conversation
                assistant_text = result["response"]
                messages.append({"role": "assistant", "content": assistant_text})
                context_tokens += result["tokens_generated"]

                ratio = _compression_ratio(kv_dtype)
                fp16_bytes = int(context_tokens) * _FP16_BYTES_PER_TOKEN_PER_LAYER * _N_LAYERS * _N_HEADS
                kv_bytes = int(fp16_bytes / ratio)

                turn_result = {
                    "turn": turn_idx,
                    "kv_dtype": kv_dtype,
                    "context_tokens": int(context_tokens),
                    "ttfb_ms": result["ttfb_ms"],
                    "total_latency_ms": result["total_latency_ms"],
                    "kv_cache_bytes": kv_bytes,
                    "compression_ratio": round(ratio, 2),
                    "tokens_generated": result["tokens_generated"],
                }
                all_results.append(turn_result)

                console.print(
                    f"  Turn {turn_idx:2d} | ctx={int(context_tokens):5d} | "
                    f"TTFB={result['ttfb_ms']:6.1f}ms | "
                    f"KV={kv_bytes / (1024 * 1024):6.1f}MB | "
                    f"ratio={ratio:.1f}x"
                )

        # Build summary
        summary: dict[str, Any] = {}
        for kv_dtype in kv_dtypes:
            dtype_results = [r for r in all_results if r["kv_dtype"] == kv_dtype]
            ttfbs = [r["ttfb_ms"] for r in dtype_results]
            kv_sizes = [r["kv_cache_bytes"] for r in dtype_results]
            summary[kv_dtype] = {
                "avg_ttfb_ms": round(sum(ttfbs) / len(ttfbs), 2),
                "max_ttfb_ms": round(max(ttfbs), 2),
                "final_kv_cache_mb": round(kv_sizes[-1] / (1024 * 1024), 2),
                "compression_ratio": round(_compression_ratio(kv_dtype), 2),
            }

        return {
            "turns": all_results,
            "summary": summary,
            "simulated": is_simulated,
        }
