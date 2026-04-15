"""Benchmark orchestrator for VoiceQuant scenarios.

Discovers, configures, and runs benchmark scenarios. Collects results and
optionally generates a markdown report.

Usage::

    from voicequant.benchmarks.runner import run_benchmarks

    results = run_benchmarks(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        scenarios=["ttfb", "multi_turn", "concurrent"],
        compare=["fp16", "tq4"],
        max_sessions=50,
        report_path="benchmark_report.md",
    )
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.table import Table

from voicequant.server.config import ServerConfig

console = Console()

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

_SCENARIO_CLASSES: dict[str, type] = {}
_SCENARIO_LOAD_ERRORS: dict[str, str] = {}


def _load_scenarios() -> None:
    """Lazily import all scenario classes."""
    if _SCENARIO_CLASSES:
        return

    scenario_imports = {
        "multi_turn": (
            "voicequant.benchmarks.scenarios.multi_turn",
            "MultiTurnBenchmark",
        ),
        "concurrent": (
            "voicequant.benchmarks.scenarios.concurrent",
            "ConcurrentBenchmark",
        ),
        "ttfb": ("voicequant.benchmarks.scenarios.ttfb", "TTFBBenchmark"),
        "system_prompt": (
            "voicequant.benchmarks.scenarios.system_prompt",
            "SystemPromptBenchmark",
        ),
        "tool_calling": (
            "voicequant.benchmarks.scenarios.tool_calling",
            "ToolCallingBenchmark",
        ),
        "quality": ("voicequant.benchmarks.scenarios.quality", "QualityBenchmark"),
    }

    for name, (module_path, class_name) in scenario_imports.items():
        try:
            import importlib

            mod = importlib.import_module(module_path)
            _SCENARIO_CLASSES[name] = getattr(mod, class_name)
        except Exception as e:
            _SCENARIO_LOAD_ERRORS[name] = str(e)


def list_scenarios() -> list[str]:
    """Return names of all available benchmark scenarios."""
    _load_scenarios()
    return sorted(_SCENARIO_CLASSES.keys())


def _print_summary_table(all_results: dict[str, dict[str, Any]]) -> None:
    """Print a Rich summary table of all scenario results."""
    table = Table(title="VoiceQuant Benchmark Results", show_lines=True)
    table.add_column("Scenario", style="bold cyan", min_width=15)
    table.add_column("Status", min_width=10)
    table.add_column("Simulated", min_width=10)
    table.add_column("Key Findings", min_width=40)

    for scenario_name, result in all_results.items():
        if "error" in result:
            table.add_row(
                scenario_name,
                "[red]FAILED[/red]",
                "-",
                result["error"],
            )
            continue

        simulated = (
            "[yellow]Yes[/yellow]"
            if result.get("simulated", True)
            else "[green]No[/green]"
        )
        findings = _extract_key_findings(scenario_name, result)
        table.add_row(scenario_name, "[green]OK[/green]", simulated, findings)

    console.print(table)


def _extract_key_findings(scenario_name: str, result: dict[str, Any]) -> str:
    """Extract a one-line summary from scenario results."""
    if scenario_name == "multi_turn":
        summary = result.get("summary", {})
        parts = []
        for dtype, stats in summary.items():
            parts.append(f"{dtype}: avg TTFB={stats.get('avg_ttfb_ms', 0):.1f}ms")
        return " | ".join(parts) if parts else "No summary available"

    elif scenario_name == "concurrent":
        bp = result.get("breaking_point", {})
        parts = []
        for dtype, point in bp.items():
            if point is not None:
                parts.append(f"{dtype}: breaks at {point} sessions")
            else:
                parts.append(f"{dtype}: no breakpoint found")
        return " | ".join(parts) if parts else "No breaking point data"

    elif scenario_name == "ttfb":
        summary = result.get("summary", {})
        parts = []
        for ctx_len, stats in summary.items():
            parts.append(
                f"{ctx_len}: {stats.get('best_dtype', '?')} {stats.get('best_ttfb_ms', 0):.1f}ms"
            )
        return " | ".join(parts[:3]) if parts else "No summary"

    elif scenario_name == "system_prompt":
        mem = result.get("system_prompt_memory", {})
        return (
            f"FP16: {mem.get('fp16_mb', 0):.1f}MB | "
            f"TQ4: {mem.get('tq4_mb', 0):.1f}MB | "
            f"Savings: {mem.get('savings_mb', 0):.1f}MB/session"
        )

    elif scenario_name == "tool_calling":
        summary = result.get("summary", {})
        parts = []
        for dtype, stats in summary.items():
            parts.append(f"{dtype}: recall={stats.get('avg_recall', 0):.1%}")
        return " | ".join(parts) if parts else "No summary"

    elif scenario_name == "quality":
        summary = result.get("summary", {})
        parts = []
        for dtype, stats in summary.items():
            parts.append(
                f"{dtype}: ROUGE-L={stats.get('avg_rouge_l', 0):.3f}, "
                f"CosSim={stats.get('avg_cosine_similarity', 0):.3f}"
            )
        return " | ".join(parts) if parts else "No summary"

    return "Results collected"


def _print_detailed_results(scenario_name: str, result: dict[str, Any]) -> None:
    """Print detailed per-scenario results as Rich tables."""
    if scenario_name == "ttfb":
        results = result.get("results", [])
        if not results:
            return
        table = Table(title="TTFB by Context Length", show_lines=True)
        table.add_column("Context", justify="right")
        table.add_column("FP16 (ms)", justify="right")
        table.add_column("TQ4 (ms)", justify="right")
        table.add_column("TQ3 (ms)", justify="right")
        table.add_column("TQ4 Speedup", justify="right")

        ctx_lengths = sorted(set(r["context_length"] for r in results))
        for ctx in ctx_lengths:
            ctx_results = {
                r["kv_dtype"]: r for r in results if r["context_length"] == ctx
            }
            fp16 = ctx_results.get("fp16", {}).get("ttfb_ms", 0)
            tq4 = ctx_results.get("tq4", {}).get("ttfb_ms", 0)
            tq3 = ctx_results.get("tq3", {}).get("ttfb_ms", 0)
            speedup = f"{fp16 / tq4:.2f}x" if tq4 > 0 else "-"
            table.add_row(
                f"{ctx:,}", f"{fp16:.1f}", f"{tq4:.1f}", f"{tq3:.1f}", speedup
            )

        console.print(table)

    elif scenario_name == "concurrent":
        results = result.get("results", [])
        if not results:
            return
        table = Table(title="Concurrent Session Scaling", show_lines=True)
        table.add_column("Sessions", justify="right")
        table.add_column("KV Dtype")
        table.add_column("p95 TTFB (ms)", justify="right")
        table.add_column("Tok/s/session", justify="right")
        table.add_column("GPU Memory", justify="right")

        for r in results:
            p95_style = "red" if r["ttfb_p95_ms"] > 500 else "green"
            table.add_row(
                str(r["n_sessions"]),
                r["kv_dtype"],
                f"[{p95_style}]{r['ttfb_p95_ms']:.1f}[/{p95_style}]",
                f"{r.get('tokens_per_sec_per_session', 0):.1f}",
                f"{r.get('total_gpu_memory_gb', 0):.1f} GB",
            )

        console.print(table)

    elif scenario_name == "quality":
        summary = result.get("summary", {})
        if not summary:
            return
        table = Table(title="Quality Comparison vs FP16 Baseline", show_lines=True)
        table.add_column("Metric")
        table.add_column("TQ4", justify="right")
        table.add_column("TQ3", justify="right")

        metrics = ["avg_rouge_l", "avg_cosine_similarity", "exact_match_rate"]
        labels = ["ROUGE-L", "Cosine Similarity", "Exact Match Rate"]
        for label, metric in zip(labels, metrics, strict=False):
            tq4_val = summary.get("tq4", {}).get(metric, 0)
            tq3_val = summary.get("tq3", {}).get(metric, 0)
            table.add_row(label, f"{tq4_val:.4f}", f"{tq3_val:.4f}")

        console.print(table)


def run_benchmarks(
    model: str | None = None,
    scenarios: list[str] | None = None,
    compare: list[str] | None = None,
    max_sessions: int = 50,
    report_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run selected benchmark scenarios and collect results.

    Args:
        model: HuggingFace model ID. Defaults to config default.
        scenarios: List of scenario names to run. None runs all.
        compare: KV cache dtypes to compare (unused when scenarios
            handle comparison internally).
        max_sessions: Maximum concurrent sessions for the concurrent benchmark.
        report_path: If provided, generate a markdown report at this path.

    Returns:
        Dictionary mapping scenario name to its result dict.
    """
    _load_scenarios()

    if scenarios is None:
        scenarios = list(_SCENARIO_CLASSES.keys())

    config = ServerConfig()
    if model:
        config = config.model_copy(update={"model": model})

    console.print("\n[bold underline]VoiceQuant Benchmark Suite[/bold underline]")
    console.print(f"Model: {config.model}")
    console.print(f"Scenarios: {', '.join(scenarios)}")
    console.print(f"Max sessions: {max_sessions}\n")

    all_results: dict[str, dict[str, Any]] = {}

    for scenario_name in scenarios:
        if scenario_name in _SCENARIO_LOAD_ERRORS:
            console.print(
                f"[red]Skipping {scenario_name}: {_SCENARIO_LOAD_ERRORS[scenario_name]}[/red]"
            )
            all_results[scenario_name] = {"error": _SCENARIO_LOAD_ERRORS[scenario_name]}
            continue

        if scenario_name not in _SCENARIO_CLASSES:
            console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
            all_results[scenario_name] = {"error": f"Unknown scenario: {scenario_name}"}
            continue

        console.rule(f"[bold]{scenario_name}[/bold]")
        cls = _SCENARIO_CLASSES[scenario_name]
        instance = cls()

        t0 = time.perf_counter()
        try:
            if scenario_name == "concurrent":
                result = instance.run(
                    model=model, config=config, max_sessions=max_sessions
                )
            else:
                result = instance.run(model=model, config=config)

            elapsed = time.perf_counter() - t0
            result["elapsed_s"] = round(elapsed, 2)
            all_results[scenario_name] = result

            console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")

            # Print detailed results for select scenarios
            _print_detailed_results(scenario_name, result)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            console.print(f"[red]Error in {scenario_name}: {e}[/red]")
            all_results[scenario_name] = {
                "error": str(e),
                "elapsed_s": round(elapsed, 2),
            }

    # Summary table
    console.print()
    _print_summary_table(all_results)

    # Generate report if requested
    if report_path:
        try:
            from voicequant.benchmarks.report import generate_report

            generate_report(all_results, report_path)
            console.print(f"\n[green]Report written to {report_path}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to generate report: {e}[/red]")

    return all_results
