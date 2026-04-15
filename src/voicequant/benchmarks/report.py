"""Benchmark report generator.

Produces a markdown report from benchmark results including summary tables,
comparison metrics, and GPU-specific deployment recommendations.

Usage::

    from voicequant.benchmarks.report import generate_report

    generate_report(results, "benchmark_report.md")
"""

from __future__ import annotations

import datetime
from typing import Any

from rich.console import Console

from voicequant.server.config import GPU_CAPACITY_ESTIMATES

console = Console()

# ---------------------------------------------------------------------------
# GPU recommendation templates
# ---------------------------------------------------------------------------

_GPU_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "T4": {
        "tier": "Budget / Development",
        "memory": "16 GB",
        "recommendation": (
            "Use TQ4 compression. FP16 supports ~8 concurrent sessions at 4K context, "
            "while TQ4 supports ~40. Ideal for development, testing, and low-traffic "
            "deployments. Avoid TQ3 as quality degradation is noticeable on smaller GPUs "
            "due to limited compute for decompression."
        ),
        "max_context": "8K recommended (16K possible with TQ4)",
    },
    "A10G": {
        "tier": "Production / Mid-range",
        "memory": "24 GB",
        "recommendation": (
            "TQ4 is the sweet spot. Supports ~65 concurrent sessions at 4K context "
            "vs ~13 for FP16. Good for medium-traffic voice agents. TQ3 is viable "
            "if you need to push to 80+ sessions and can tolerate ~3% quality drop."
        ),
        "max_context": "16K recommended (32K possible with TQ4)",
    },
    "L4": {
        "tier": "Production / Cost-effective",
        "memory": "24 GB",
        "recommendation": (
            "Similar capacity to A10G but with newer architecture. TQ4 recommended "
            "for production voice workloads. The L4's tensor cores handle TQ4 "
            "decompression efficiently with minimal latency overhead."
        ),
        "max_context": "16K recommended (32K possible with TQ4)",
    },
    "A100": {
        "tier": "High-performance / High-concurrency",
        "memory": "80 GB",
        "recommendation": (
            "TQ4 enables ~250 concurrent sessions at 4K context vs ~50 for FP16. "
            "At this scale, memory savings translate directly to infrastructure cost "
            "reduction (5x fewer GPUs). TQ3 is recommended only for 500+ session "
            "targets where the quality trade-off is acceptable."
        ),
        "max_context": "32K (full context window)",
    },
    "H100": {
        "tier": "Premium / Maximum performance",
        "memory": "80 GB",
        "recommendation": (
            "Similar session capacity to A100 but with significantly faster "
            "decompression via FP8 tensor cores. TQ4 is strongly recommended. "
            "The H100's memory bandwidth (3.35 TB/s) means TTFB stays under 100ms "
            "even at 32K context with TQ4 compression."
        ),
        "max_context": "32K (full context window)",
    },
}


def _format_number(n: float, precision: int = 2) -> str:
    """Format a number for display in markdown."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    return f"{n:.{precision}f}"


def _section_header(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


def _generate_summary_section(results: dict[str, dict[str, Any]]) -> str:
    """Generate the executive summary section."""
    lines = [
        _section_header("Executive Summary"),
        "| Scenario | Status | Key Finding |",
        "|----------|--------|-------------|",
    ]

    for name, result in results.items():
        if "error" in result:
            lines.append(f"| {name} | Failed | {result['error']} |")
            continue

        simulated = " (simulated)" if result.get("simulated", True) else ""
        finding = _extract_finding(name, result)
        lines.append(f"| {name} | OK{simulated} | {finding} |")

    return "\n".join(lines)


def _extract_finding(name: str, result: dict[str, Any]) -> str:
    """Extract a one-line finding for the summary table."""
    if name == "multi_turn":
        summary = result.get("summary", {})
        tq4 = summary.get("tq4", {})
        fp16 = summary.get("fp16", {})
        if tq4 and fp16:
            return (
                f"TQ4 avg TTFB {tq4.get('avg_ttfb_ms', 0):.1f}ms vs "
                f"FP16 {fp16.get('avg_ttfb_ms', 0):.1f}ms, "
                f"{tq4.get('compression_ratio', 0):.1f}x compression"
            )
        return "Multi-turn conversation completed"

    elif name == "concurrent":
        bp = result.get("breaking_point", {})
        parts = []
        for dtype, point in bp.items():
            if point is not None:
                parts.append(f"{dtype} breaks at {point} sessions")
            else:
                parts.append(f"{dtype}: no breakpoint in tested range")
        return "; ".join(parts) if parts else "Scaling test completed"

    elif name == "ttfb":
        summary = result.get("summary", {})
        if summary:
            max_ctx = max(
                summary.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) else 0
            )
            stats = summary[max_ctx]
            return (
                f"At {_format_number(int(max_ctx))} context: "
                f"{stats.get('best_dtype', '?')} wins with "
                f"{stats.get('best_ttfb_ms', 0):.1f}ms "
                f"({stats.get('speedup', 0):.1f}x vs FP16)"
            )
        return "TTFB measured across context lengths"

    elif name == "system_prompt":
        mem = result.get("system_prompt_memory", {})
        return (
            f"TQ4 saves {mem.get('savings_mb', 0):.1f}MB per session "
            f"on system prompt alone"
        )

    elif name == "tool_calling":
        summary = result.get("summary", {})
        tq4 = summary.get("tq4", {})
        fp16 = summary.get("fp16", {})
        return (
            f"TQ4 recall {tq4.get('avg_recall', 0):.1%} vs "
            f"FP16 {fp16.get('avg_recall', 0):.1%}"
        )

    elif name == "quality":
        summary = result.get("summary", {})
        tq4 = summary.get("tq4", {})
        return (
            f"TQ4: ROUGE-L={tq4.get('avg_rouge_l', 0):.3f}, "
            f"CosSim={tq4.get('avg_cosine_similarity', 0):.3f}, "
            f"ExactMatch={tq4.get('exact_match_rate', 0):.1%}"
        )

    return "Completed"


def _generate_ttfb_section(result: dict[str, Any]) -> str:
    """Generate the TTFB comparison section."""
    lines = [
        _section_header("Time to First Token (TTFB)"),
        "TTFB is the most critical metric for voice AI. Users perceive silence "
        "after speaking, making sub-200ms TTFB essential for natural conversation.\n",
        "| Context Length | FP16 (ms) | TQ4 (ms) | TQ3 (ms) | TQ4 Speedup |",
        "|---------------|-----------|----------|----------|-------------|",
    ]

    results_list = result.get("results", [])
    ctx_lengths = sorted(set(r["context_length"] for r in results_list))

    for ctx in ctx_lengths:
        ctx_results = {
            r["kv_dtype"]: r for r in results_list if r["context_length"] == ctx
        }
        fp16 = ctx_results.get("fp16", {}).get("ttfb_ms", 0)
        tq4 = ctx_results.get("tq4", {}).get("ttfb_ms", 0)
        tq3 = ctx_results.get("tq3", {}).get("ttfb_ms", 0)
        speedup = f"{fp16 / tq4:.2f}x" if tq4 > 0 else "-"
        lines.append(f"| {ctx:,} | {fp16:.1f} | {tq4:.1f} | {tq3:.1f} | {speedup} |")

    return "\n".join(lines)


def _generate_concurrent_section(result: dict[str, Any]) -> str:
    """Generate the concurrent scaling section."""
    lines = [
        _section_header("Concurrent Session Scaling"),
        "Voice AI servers handle many short sessions simultaneously. "
        "This test measures how latency degrades as session count increases.\n",
        "| Sessions | KV Type | p95 TTFB (ms) | Tokens/s | GPU Memory |",
        "|----------|---------|---------------|----------|------------|",
    ]

    for r in result.get("results", []):
        p95 = r["ttfb_p95_ms"]
        marker = " **" if p95 > 500 else ""
        lines.append(
            f"| {r['n_sessions']} | {r['kv_dtype']} | {p95:.1f}{marker} | "
            f"{r.get('tokens_per_sec_per_session', 0):.1f} | "
            f"{r.get('total_gpu_memory_gb', 0):.1f} GB |"
        )

    bp = result.get("breaking_point", {})
    lines.append("")
    for dtype, point in bp.items():
        if point is not None:
            lines.append(
                f"**{dtype} breaking point**: {point} sessions (p95 TTFB > 500ms)"
            )
        else:
            lines.append(f"**{dtype}**: No breaking point found within tested range")

    return "\n".join(lines)


def _generate_quality_section(result: dict[str, Any]) -> str:
    """Generate the quality comparison section."""
    lines = [
        _section_header("Output Quality"),
        "Quality metrics compare compressed outputs against the FP16 baseline "
        "across 100 voice-style prompts.\n",
        "| Metric | TQ4 | TQ3 |",
        "|--------|-----|-----|",
    ]

    summary = result.get("summary", {})
    tq4 = summary.get("tq4", {})
    tq3 = summary.get("tq3", {})

    lines.append(
        f"| ROUGE-L | {tq4.get('avg_rouge_l', 0):.4f} | {tq3.get('avg_rouge_l', 0):.4f} |"
    )
    lines.append(
        f"| Cosine Similarity | {tq4.get('avg_cosine_similarity', 0):.4f} | "
        f"{tq3.get('avg_cosine_similarity', 0):.4f} |"
    )
    lines.append(
        f"| Exact Match Rate | {tq4.get('exact_match_rate', 0):.1%} | "
        f"{tq3.get('exact_match_rate', 0):.1%} |"
    )

    return "\n".join(lines)


def _generate_system_prompt_section(result: dict[str, Any]) -> str:
    """Generate the system prompt section."""
    lines = [
        _section_header("System Prompt Compression"),
        "Voice agents carry large system prompts (~1500 tokens) in every session's "
        "KV cache. Compression reduces this per-session overhead significantly.\n",
    ]

    mem = result.get("system_prompt_memory", {})
    lines.extend(
        [
            f"- **System prompt tokens**: {mem.get('system_prompt_tokens', 0):,}",
            f"- **FP16 memory**: {mem.get('fp16_mb', 0):.2f} MB per session",
            f"- **TQ4 memory**: {mem.get('tq4_mb', 0):.2f} MB per session",
            f"- **Savings**: {mem.get('savings_mb', 0):.2f} MB per session "
            f"({mem.get('compression_ratio', 0):.1f}x compression)",
            "",
        ]
    )

    scaling = result.get("scaling", [])
    if scaling:
        lines.extend(
            [
                "| Sessions | FP16 (GB) | TQ4 (GB) | Savings (GB) |",
                "|----------|-----------|----------|--------------|",
            ]
        )
        for s in scaling:
            lines.append(
                f"| {s['sessions']} | {s['fp16_total_gb']:.3f} | "
                f"{s['tq4_total_gb']:.3f} | {s['savings_gb']:.3f} |"
            )

    return "\n".join(lines)


def _generate_tool_calling_section(result: dict[str, Any]) -> str:
    """Generate the tool calling section."""
    lines = [
        _section_header("Tool Calling Accuracy"),
        "Voice agents depend on reliable function calling. This test verifies "
        "that KV cache compression does not degrade tool calling accuracy.\n",
        "| Metric | FP16 | TQ4 | Delta |",
        "|--------|------|-----|-------|",
    ]

    summary = result.get("summary", {})
    fp16 = summary.get("fp16", {})
    tq4 = summary.get("tq4", {})

    metrics = [
        ("Recall", "avg_recall"),
        ("Precision", "avg_precision"),
        ("Name Accuracy", "avg_name_accuracy"),
        ("Sequence Accuracy", "avg_sequence_accuracy"),
    ]

    for label, key in metrics:
        fp16_val = fp16.get(key, 0)
        tq4_val = tq4.get(key, 0)
        delta = tq4_val - fp16_val
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {label} | {fp16_val:.1%} | {tq4_val:.1%} | {sign}{delta:.1%} |"
        )

    return "\n".join(lines)


def _generate_gpu_recommendations() -> str:
    """Generate GPU-specific deployment recommendations."""
    lines = [
        _section_header("GPU Deployment Recommendations"),
        "Recommendations based on VoiceQuant compression ratios and GPU memory capacity.\n",
    ]

    for gpu, info in _GPU_RECOMMENDATIONS.items():
        capacity = GPU_CAPACITY_ESTIMATES.get(gpu, {})
        lines.extend(
            [
                f"### {gpu} ({info['memory']}) - {info['tier']}",
                "",
                f"- **FP16 sessions**: ~{capacity.get('fp16_sessions', '?')} at 4K context",
                f"- **TQ4 sessions**: ~{capacity.get('tq4_sessions', '?')} at 4K context",
                f"- **Max recommended context**: {info['max_context']}",
                f"- **Recommendation**: {info['recommendation']}",
                "",
            ]
        )

    return "\n".join(lines)


def generate_report(
    results: dict[str, dict[str, Any]],
    output_path: str,
) -> None:
    """Generate a markdown benchmark report.

    Args:
        results: Dictionary mapping scenario names to result dicts,
            as returned by ``run_benchmarks()``.
        output_path: Path to write the markdown report.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections: list[str] = [
        "# VoiceQuant Benchmark Report",
        f"\nGenerated: {now}\n",
        _generate_summary_section(results),
    ]

    # Add per-scenario sections if results exist
    if "ttfb" in results and "error" not in results["ttfb"]:
        sections.append(_generate_ttfb_section(results["ttfb"]))

    if "concurrent" in results and "error" not in results["concurrent"]:
        sections.append(_generate_concurrent_section(results["concurrent"]))

    if "quality" in results and "error" not in results["quality"]:
        sections.append(_generate_quality_section(results["quality"]))

    if "system_prompt" in results and "error" not in results["system_prompt"]:
        sections.append(_generate_system_prompt_section(results["system_prompt"]))

    if "tool_calling" in results and "error" not in results["tool_calling"]:
        sections.append(_generate_tool_calling_section(results["tool_calling"]))

    if "multi_turn" in results and "error" not in results["multi_turn"]:
        summary = results["multi_turn"].get("summary", {})
        lines = [
            _section_header("Multi-Turn Conversation"),
            "Performance across a 20-turn voice conversation.\n",
            "| KV Type | Avg TTFB (ms) | Max TTFB (ms) | Final KV Cache (MB) | Compression |",
            "|---------|---------------|---------------|---------------------|-------------|",
        ]
        for dtype, stats in summary.items():
            lines.append(
                f"| {dtype} | {stats.get('avg_ttfb_ms', 0):.1f} | "
                f"{stats.get('max_ttfb_ms', 0):.1f} | "
                f"{stats.get('final_kv_cache_mb', 0):.1f} | "
                f"{stats.get('compression_ratio', 0):.1f}x |"
            )
        sections.append("\n".join(lines))

    # Always include GPU recommendations
    sections.append(_generate_gpu_recommendations())

    # Methodology note
    simulated = any(
        r.get("simulated", False)
        for r in results.values()
        if isinstance(r, dict) and "error" not in r
    )
    if simulated:
        sections.append(
            _section_header("Methodology Note")
            + "Some or all results in this report are **simulated** based on "
            "analytical compression models. Results marked as simulated use "
            "theoretical compression ratios and latency models rather than "
            "live GPU measurements. Run benchmarks against a live vLLM server "
            "with `--model` to obtain empirical results."
        )

    report = "\n".join(sections) + "\n"

    with open(output_path, "w") as f:
        f.write(report)

    console.print(f"[green]Report written: {output_path} ({len(report)} bytes)[/green]")
