"""VoiceQuant CLI — TurboQuant KV cache compression for voice AI inference."""

from __future__ import annotations

import sys
import time

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
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID or path"
    ),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    tq_bits: int = typer.Option(4, help="TurboQuant quantization bits (3 or 4)"),
    tq_residual_window: int = typer.Option(
        256, help="Number of recent tokens kept in FP16"
    ),
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
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID or path"
    ),
    scenario: str = typer.Option(
        None,
        help="Specific scenario: multi_turn, concurrent, ttfb, system_prompt, tool_calling, quality",
    ),
    all_scenarios: bool = typer.Option(
        False, "--all", help="Run all benchmark scenarios"
    ),
    compare: list[str] = typer.Option(
        None, help="Compare configurations: fp16, tq4, tq3"
    ),
    max_sessions: int = typer.Option(
        50, help="Max concurrent sessions for concurrent benchmark"
    ),
    report: str = typer.Option(None, help="Output path for markdown benchmark report"),
    viz: bool = typer.Option(
        False, "--visualize", "--viz", help="Generate comparison charts"
    ),
    chart_output: str = typer.Option(
        "./charts", "--chart-output", help="Output directory for charts"
    ),
) -> None:
    """Run voice AI benchmarks with TurboQuant compression."""
    from voicequant.benchmarks.runner import run_benchmarks

    scenarios = None
    if all_scenarios:
        scenarios = [
            "multi_turn",
            "concurrent",
            "ttfb",
            "system_prompt",
            "tool_calling",
            "quality",
        ]
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
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID or path"
    ),
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
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct-AWQ", help="HuggingFace model ID"
    ),
    gpu: str = typer.Option("T4", help="GPU type: T4, A10G, L4, A100, H100"),
    output: str = typer.Option(
        "./deploy", help="Output directory for deployment files"
    ),
) -> None:
    """Deploy VoiceQuant to Modal, RunPod, or Docker."""
    from voicequant.deploy import generate_deployment

    generate_deployment(target=target, model=model, gpu=gpu, output_dir=output)


stt_app = typer.Typer(help="Speech-to-text (faster-whisper) commands.")
app.add_typer(stt_app, name="stt")


@stt_app.command("models")
def stt_models() -> None:
    """List available STT models."""
    from rich.table import Table

    from voicequant.core.stt.compile import list_available_models

    table = Table(title="Available STT models")
    table.add_column("Name", style="cyan")
    table.add_column("HuggingFace ID")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Compute type")
    for m in list_available_models():
        table.add_row(m["name"], m["hf_id"], str(m["size_mb"]), m["compute_type"])
    console.print(table)


@stt_app.command("download")
def stt_download(
    model_name: str = typer.Argument(..., help="Model name or HuggingFace ID"),
    output_dir: str = typer.Option(None, "--output", help="Custom output directory"),
) -> None:
    """Download an STT model."""
    from voicequant.core.stt.compile import download_model

    console.print(f"[cyan]Downloading {model_name}...[/cyan]")
    path = download_model(model_name, output_dir=output_dir)
    console.print(f"[green]Downloaded to: {path}[/green]")


@stt_app.command("transcribe")
def stt_transcribe(
    audio_file: str = typer.Argument(..., help="Path to audio file"),
    model: str = typer.Option("Systran/faster-whisper-large-v3", "--model"),
    language: str = typer.Option(None, "--language"),
    fmt: str = typer.Option("json", "--format", help="json | verbose_json | text"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Transcribe an audio file."""
    import json as _json

    from voicequant.core.stt.config import STTConfig
    from voicequant.core.stt.engine import STTEngine

    cfg = STTConfig(model_name=model, device=device)
    engine = STTEngine(cfg)
    result = engine.transcribe(audio_file, language=language, response_format=fmt)

    if fmt == "text":
        console.print(result.text)
    elif fmt == "verbose_json":
        console.print_json(
            _json.dumps(
                {
                    "task": "transcribe",
                    "language": result.language,
                    "duration": result.duration,
                    "text": result.text,
                    "segments": result.segments,
                }
            )
        )
    else:
        console.print_json(_json.dumps({"text": result.text}))


tts_app = typer.Typer(help="Text-to-speech (Kokoro ONNX) commands.")
app.add_typer(tts_app, name="tts")


@tts_app.command("speak")
def tts_speak(
    text: str = typer.Argument(..., help="Text to synthesize. Use '-' to read stdin."),
    voice: str = typer.Option("af_heart", "--voice"),
    fmt: str = typer.Option("wav", "--format", help="wav | pcm | mp3 | opus"),
    output: str | None = typer.Option(None, "--output", help="Output file path"),
    device: str = typer.Option("auto", "--device", help="auto | cuda | cpu"),
) -> None:
    """Synthesize text to speech and write audio output."""
    from voicequant.core.tts.config import TTSConfig
    from voicequant.core.tts.engine import TTSEngine

    content = sys.stdin.read() if text == "-" else text
    cfg = TTSConfig(device=device, default_voice=voice, output_format=fmt)
    engine = TTSEngine(cfg)
    started = time.time()
    result = engine.synthesize(content, voice=voice, output_format=fmt)
    elapsed_ms = (time.time() - started) * 1000

    out_path = output or f"voicequant_tts.{result.format}"
    with open(out_path, "wb") as f:
        f.write(result.audio_bytes)

    console.print(f"[green]Saved audio to:[/green] {out_path}")
    console.print(
        f"Voice={result.voice} | Format={result.format} | "
        f"SampleRate={result.sample_rate}Hz | Duration={result.duration_seconds:.2f}s | "
        f"Latency={elapsed_ms:.1f}ms"
    )


@tts_app.command("voices")
def tts_voices(
    device: str = typer.Option("auto", "--device", help="auto | cuda | cpu"),
) -> None:
    """List available TTS voices."""
    from rich.table import Table

    from voicequant.core.tts.config import TTSConfig
    from voicequant.core.tts.engine import TTSEngine

    engine = TTSEngine(TTSConfig(device=device))
    voices = engine.list_voices()

    table = Table(title="Available TTS voices")
    table.add_column("voice_id", style="cyan")
    table.add_column("description")
    for voice_id in voices:
        table.add_row(voice_id, "Kokoro built-in voice")
    console.print(table)


@tts_app.command("benchmark-quick")
def tts_benchmark_quick(
    voice: str = typer.Option("af_heart", "--voice"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Run a quick TTS latency sanity check."""
    from voicequant.core.tts.config import TTSConfig
    from voicequant.core.tts.engine import TTSEngine

    samples = [
        "Hello from VoiceQuant.",
        "This is a quick latency benchmark for the Kokoro TTS engine.",
        "Voice interfaces should feel responsive and natural.",
    ]

    engine = TTSEngine(TTSConfig(device=device, default_voice=voice))
    latencies: list[float] = []
    durations: list[float] = []
    for text in samples:
        t0 = time.time()
        result = engine.synthesize(text, voice=voice)
        latencies.append((time.time() - t0) * 1000)
        durations.append(result.duration_seconds)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    console.print("[bold]Quick TTS benchmark[/bold]")
    console.print(f"Runs: {len(samples)}")
    console.print(f"Average latency: {avg_latency:.1f} ms")
    console.print(f"Average audio duration: {avg_duration:.2f} s")


if __name__ == "__main__":
    app()
