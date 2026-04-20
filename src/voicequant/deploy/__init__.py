"""VoiceQuant deployment templates for Modal, RunPod, and Docker."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_deployment(
    target: str,
    model: str,
    gpu: str,
    output_dir: str,
) -> None:
    """Generate deployment files for the specified target platform."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if target == "modal":
        _generate_modal(model, gpu, output_path)
    elif target == "runpod":
        _generate_runpod(model, gpu, output_path)
    elif target == "docker":
        _generate_docker(model, gpu, output_path)
    else:
        console.print(f"[red]Unknown target: {target}. Use: modal, runpod, docker[/red]")
        raise SystemExit(1)


def _generate_modal(model: str, gpu: str, output_path: Path) -> None:
    """Generate Modal deployment files."""
    import shutil
    src = Path(__file__).parent / "modal_deploy.py"
    dst = output_path / "modal_deploy.py"
    shutil.copy2(src, dst)
    console.print(f"[green]Modal deployment script written to {dst}[/green]")
    console.print(f"\nTo deploy:\n  modal deploy {dst}")
    console.print("\nEnvironment variables:")
    console.print(f"  VOICEQUANT_MODEL={model}")
    console.print(f"  VOICEQUANT_GPU={gpu}")


def _generate_runpod(model: str, gpu: str, output_path: Path) -> None:
    """Generate RunPod deployment files."""
    import shutil
    src = Path(__file__).parent / "runpod_handler.py"
    dst = output_path / "runpod_handler.py"
    shutil.copy2(src, dst)
    console.print(f"[green]RunPod handler written to {dst}[/green]")
    console.print("\nUpload to RunPod serverless with:")
    console.print(f"  Model: {model}")
    console.print(f"  GPU: {gpu}")


def _generate_docker(model: str, gpu: str, output_path: Path) -> None:
    """Generate Docker deployment files."""
    import shutil
    for fname in ["Dockerfile", "docker-compose.yml"]:
        src = _TEMPLATES_DIR / fname
        dst = output_path / fname
        shutil.copy2(src, dst)
        console.print(f"[green]{fname} written to {dst}[/green]")
    console.print("\nTo build and run:")
    console.print("  docker compose up --build")
    console.print("\nEnvironment variables:")
    console.print(f"  VOICEQUANT_MODEL={model}")
    console.print(f"  VOICEQUANT_GPU={gpu}")
