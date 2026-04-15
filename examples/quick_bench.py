"""Quick benchmark for a running VoiceQuant server.

Measures key voice AI metrics against a VoiceQuant (or any OpenAI-compatible)
server: time-to-first-byte (TTFB), token throughput, and concurrent session
scaling.

Usage:
    # Basic benchmark against localhost
    python examples/quick_bench.py

    # Custom server URL
    python examples/quick_bench.py --url http://gpu-server:8000/v1

    # More concurrent sessions
    python examples/quick_bench.py --max-concurrent 100

    # Quick smoke test (fewer iterations)
    python examples/quick_bench.py --iterations 5

Requirements:
    pip install httpx rich

What it measures:
    1. TTFB (Time to First Byte): How quickly the first token arrives.
       Critical for voice AI where users are waiting in real-time.
    2. Throughput (tokens/sec): Sustained generation speed per request.
    3. Concurrent scaling: How TTFB and throughput change as concurrent
       sessions increase from 1 to --max-concurrent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "default"
DEFAULT_ITERATIONS = 20
DEFAULT_MAX_CONCURRENT = 50
DEFAULT_MAX_TOKENS = 100

# A realistic voice AI prompt: multi-turn with system prompt
BENCH_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a helpful voice assistant for a restaurant. "
            "Keep responses brief and conversational, ideally 1-2 sentences."
        ),
    },
    {"role": "user", "content": "Hi, do you have any tables available tonight?"},
    {
        "role": "assistant",
        "content": "Yes, we have a few tables available tonight! What time were you thinking, and how many guests?",
    },
    {"role": "user", "content": "Around 7pm for four people please."},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Result from a single benchmark request."""
    ttfb_ms: float = 0.0
    total_ms: float = 0.0
    tokens: int = 0
    tokens_per_sec: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregated results for a concurrency level."""
    concurrency: int = 1
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.error is None]

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def stat(self, attr: str) -> tuple[float, float, float]:
        """Return (median, p50, p99) for a given attribute."""
        values = [getattr(r, attr) for r in self.successful]
        if not values:
            return (0.0, 0.0, 0.0)
        values.sort()
        median = statistics.median(values)
        p50 = values[len(values) // 2]
        p99_idx = min(int(len(values) * 0.99), len(values) - 1)
        p99 = values[p99_idx]
        return (median, p50, p99)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def single_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    max_tokens: int,
) -> RequestResult:
    """Send a single streaming chat completion and measure latency."""
    result = RequestResult()
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        async with client.stream(
            "POST",
            f"{url}/chat/completions",
            json={
                "model": model,
                "messages": BENCH_MESSAGES,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            },
            timeout=60.0,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        result.error = str(e)
        return result

    end = time.perf_counter()

    result.total_ms = (end - start) * 1000
    result.tokens = token_count

    if first_token_time is not None:
        result.ttfb_ms = (first_token_time - start) * 1000

    # Throughput: tokens per second (excluding TTFB)
    generation_time = end - (first_token_time or start)
    if generation_time > 0 and token_count > 1:
        result.tokens_per_sec = (token_count - 1) / generation_time

    return result


async def run_concurrent_batch(
    url: str,
    model: str,
    concurrency: int,
    iterations: int,
    max_tokens: int,
) -> BenchmarkResult:
    """Run a batch of concurrent requests and collect results."""
    bench_result = BenchmarkResult(concurrency=concurrency)

    async with httpx.AsyncClient() as client:
        for _ in range(iterations):
            tasks = [
                single_request(client, url, model, max_tokens)
                for _ in range(concurrency)
            ]
            results = await asyncio.gather(*tasks)
            bench_result.results.extend(results)

    return bench_result


async def check_server_health(url: str) -> dict | None:
    """Check if the server is healthy and return health info."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


async def get_capacity(url: str) -> dict | None:
    """Get server capacity estimates."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{url}/capacity")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(all_results: list[BenchmarkResult], url: str) -> None:
    """Print benchmark results using Rich tables."""

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]VoiceQuant Quick Benchmark[/bold]\nServer: {url}",
            style="blue",
        )
    )

    # Results table
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Concurrent", justify="right", style="bold")
    table.add_column("Requests", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("TTFB p50", justify="right")
    table.add_column("TTFB p99", justify="right")
    table.add_column("Tok/s p50", justify="right")
    table.add_column("Tok/s p99", justify="right")
    table.add_column("Total p50", justify="right")

    for result in all_results:
        ttfb_med, ttfb_p50, ttfb_p99 = result.stat("ttfb_ms")
        tps_med, tps_p50, tps_p99 = result.stat("tokens_per_sec")
        total_med, total_p50, _ = result.stat("total_ms")

        # Color TTFB based on voice AI thresholds
        ttfb_color = "green" if ttfb_p50 < 200 else "yellow" if ttfb_p50 < 500 else "red"
        ttfb_p99_color = "green" if ttfb_p99 < 500 else "yellow" if ttfb_p99 < 1000 else "red"

        error_style = "red" if result.error_count > 0 else "green"

        table.add_row(
            str(result.concurrency),
            str(len(result.results)),
            f"[{error_style}]{result.error_count}[/{error_style}]",
            f"[{ttfb_color}]{ttfb_p50:.0f}ms[/{ttfb_color}]",
            f"[{ttfb_p99_color}]{ttfb_p99:.0f}ms[/{ttfb_p99_color}]",
            f"{tps_p50:.1f}",
            f"{tps_p99:.1f}",
            f"{total_p50:.0f}ms",
        )

    console.print(table)

    # Voice AI quality thresholds
    console.print()
    legend = Text()
    legend.append("Voice AI thresholds: ", style="bold")
    legend.append("TTFB < 200ms ", style="green")
    legend.append("(excellent)  ", style="dim")
    legend.append("< 500ms ", style="yellow")
    legend.append("(acceptable)  ", style="dim")
    legend.append("> 500ms ", style="red")
    legend.append("(degraded)", style="dim")
    console.print(legend)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick benchmark for a running VoiceQuant server",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"VoiceQuant server base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name to request (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help=f"Iterations per concurrency level (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum concurrent sessions to test (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per request (default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    # Check server health
    console.print("[bold]Checking server health...[/bold]")
    health = await check_server_health(args.url)
    if health is None:
        console.print(f"[red]Server at {args.url} is not reachable.[/red]")
        console.print("Start a VoiceQuant server first:")
        console.print("  voicequant serve --model Qwen/Qwen2.5-7B-Instruct-AWQ")
        return

    console.print(f"[green]Server healthy:[/green] {json.dumps(health, indent=2)}")

    # Get capacity info
    capacity = await get_capacity(args.url)
    if capacity:
        console.print(f"[blue]Capacity:[/blue] {json.dumps(capacity, indent=2)}")

    # Define concurrency levels to test
    # Ramp from 1 up to max_concurrent in reasonable steps
    levels = [1]
    step = max(1, args.max_concurrent // 5)
    current = step
    while current <= args.max_concurrent:
        levels.append(current)
        current += step
    if levels[-1] != args.max_concurrent:
        levels.append(args.max_concurrent)

    all_results: list[BenchmarkResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for level in levels:
            task = progress.add_task(
                f"Testing {level} concurrent session(s)...",
                total=None,
            )

            result = await run_concurrent_batch(
                url=args.url,
                model=args.model,
                concurrency=level,
                iterations=max(1, args.iterations // level),  # Scale iterations down
                max_tokens=args.max_tokens,
            )
            all_results.append(result)

            progress.update(task, completed=True)
            progress.remove_task(task)

    print_results(all_results, args.url)


if __name__ == "__main__":
    asyncio.run(main())
