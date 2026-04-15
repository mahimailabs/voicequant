"""VoiceQuant CLI — TurboQuant KV cache compression for voice AI inference."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="voicequant",
    help="TurboQuant KV cache compression for voice AI inference — 5x more concurrent sessions on the same GPU.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID or path"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    tq_bits: int = typer.Option(4, help="TurboQuant quantization bits (3 or 4)"),
    tq_residual_window: int = typer.Option(256, help="Number of recent tokens kept in FP16"),
    max_concurrent: int = typer.Option(64, help="Max concurrent sequences"),
    gpu_memory: float = typer.Option(0.90, help="GPU memory utilization (0.0-1.0)"),
) -> None:
    """Start an OpenAI-compatible VoiceQuant inference server."""
    from voicequant.server.app import start_server

    start_server(
        model=model,
        host=host,
        port=port,
        tq_bits=tq_bits,
        tq_residual_window=tq_residual_window,
        max_concurrent=max_concurrent,
        gpu_memory=gpu_memory,
    )


@app.command()
def bench(
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID or path"),
    scenario: str = typer.Option(None, help="Specific scenario: multi_turn, concurrent, ttfb, system_prompt, tool_calling, quality"),
    all_scenarios: bool = typer.Option(False, "--all", help="Run all benchmark scenarios"),
    compare: list[str] = typer.Option(None, help="Compare configurations: fp16, tq4, tq3"),
    max_sessions: int = typer.Option(50, help="Max concurrent sessions for concurrent benchmark"),
    report: str = typer.Option(None, help="Output path for markdown benchmark report"),
    viz: bool = typer.Option(False, "--visualize", "--viz", help="Generate comparison charts"),
    chart_output: str = typer.Option("./charts", "--chart-output", help="Output directory for charts"),
) -> None:
    """Run voice AI benchmarks with TurboQuant compression."""
    from voicequant.benchmarks.runner import run_benchmarks

    scenarios = None
    if all_scenarios:
        scenarios = ["multi_turn", "concurrent", "ttfb", "system_prompt", "tool_calling", "quality"]
    elif scenario:
        scenarios = [scenario]
    else:
        console.print("[red]Specify --scenario or --all[/red]")
        raise typer.Exit(1)

    run_benchmarks(
        model=model,
        scenarios=scenarios,
        compare=compare or ["fp16", "tq4"],
        max_sessions=max_sessions,
        report_path=report,
    )

    if viz:
        from voicequant.benchmarks.visualize import generate_analytical_charts
        generate_analytical_charts(output_dir=chart_output)


@app.command()
def verify(
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID or path"),
    bits: int = typer.Option(4, help="TurboQuant bits to validate"),
    threshold: float = typer.Option(0.99, help="Minimum cosine similarity threshold"),
) -> None:
    """Validate model quality with TurboQuant KV cache compression."""
    from voicequant.core.validator import validate_model

    validate_model(model=model, bits=bits, threshold=threshold)


@app.command()
def visualize(
    output: str = typer.Option("./charts", help="Output directory for charts"),
    fmt: str = typer.Option("png", "--format", help="Output format: png, svg, html"),
) -> None:
    """Generate FP16 vs TurboQuant efficiency comparison charts (no GPU needed)."""
    from voicequant.benchmarks.visualize import generate_analytical_charts

    generate_analytical_charts(output_dir=output, fmt=fmt)


@app.command()
def deploy(
    target: str = typer.Argument(help="Deployment target: modal, runpod, docker"),
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID"),
    gpu: str = typer.Option("T4", help="GPU type: T4, A10G, L4, A100, H100"),
    output: str = typer.Option("./deploy", help="Output directory for deployment files"),
) -> None:
    """Deploy VoiceQuant to Modal, RunPod, or Docker."""
    from voicequant.deploy import generate_deployment

    generate_deployment(target=target, model=model, gpu=gpu, output_dir=output)


if __name__ == "__main__":
    app()
