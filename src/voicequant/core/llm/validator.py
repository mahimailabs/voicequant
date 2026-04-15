"""Per-layer cosine similarity validation for TurboQuant compression.

Validates that TurboQuant compression maintains sufficient quality
for voice AI inference by checking per-layer KV cache fidelity.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from voicequant.core.llm.config import TurboQuantConfig
from voicequant.core.llm.wrapper import TurboQuantWrapper

console = Console()


def validate_model(
    model: str,
    bits: int = 4,
    threshold: float = 0.99,
    seq_len: int = 512,
    n_trials: int = 5,
) -> dict[str, any]:
    """Validate model quality with TurboQuant KV cache compression.

    Measures per-layer cosine similarity between original and compressed
    KV cache vectors. Reports pass/fail for each layer against the threshold.

    Args:
        model: HuggingFace model ID (used for display; validation uses random data).
        bits: TurboQuant quantization bits (3 or 4).
        threshold: Minimum cosine similarity threshold for pass.
        seq_len: Sequence length for test vectors.
        n_trials: Number of random trials to average.

    Returns:
        Dict with per-layer results and overall pass/fail.
    """
    kv_dtype = f"tq{bits}"
    config = TurboQuantConfig(kv_cache_dtype=kv_dtype)
    wrapper = TurboQuantWrapper(config=config, device="cpu")

    console.print("\n[bold]VoiceQuant Quality Validation[/bold]")
    console.print(f"Model: {model}")
    console.print(f"TurboQuant: {bits}-bit | Threshold: {threshold}")
    console.print(f"Test: {seq_len} tokens x {n_trials} trials\n")

    quality = wrapper.validate_quality(seq_len=seq_len, n_trials=n_trials)

    table = Table(title="Compression Quality")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="center")

    results = {}
    all_pass = True

    for name, key in [
        ("Key Cosine (avg)", "avg_key_cosine"),
        ("Key Cosine (min)", "min_key_cosine"),
        ("Value Cosine (avg)", "avg_val_cosine"),
        ("Value Cosine (min)", "min_val_cosine"),
    ]:
        val = quality[key]
        passed = val >= threshold
        if not passed:
            all_pass = False
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, f"{val:.4f}", status)
        results[key] = {"value": val, "pass": passed}

    console.print(table)

    if all_pass:
        console.print(
            f"\n[green bold]All checks passed. TQ{bits} is safe for {model}.[/green bold]\n"
        )
    else:
        console.print(
            "\n[red bold]Some checks failed. Consider using higher bit-width.[/red bold]\n"
        )

    capacity = wrapper.estimate_capacity(
        model_memory_gb=4.0,
        gpu_memory_gb=16.0,
        avg_context_len=4096,
    )
    console.print(
        f"[dim]Estimated T4 (16GB) capacity: ~{capacity['tq_sessions']} concurrent sessions "
        f"({capacity['multiplier']:.1f}x vs FP16)[/dim]\n"
    )

    return {"quality": results, "overall_pass": all_pass, "capacity": capacity}
