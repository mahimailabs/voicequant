"""VoiceQuant deployment script for Modal Labs.

Usage:
    # Deploy with default settings (T4, Qwen 7B AWQ)
    modal deploy src/voicequant/deploy/modal_deploy.py

    # Run locally for testing
    modal run src/voicequant/deploy/modal_deploy.py

    # Hit the endpoint
    curl -X POST https://<your-app>.modal.run/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'

Supported GPUs (set via VOICEQUANT_GPU env var):
    T4    — 16 GB, ~40 concurrent voice sessions with TQ4
    A10G  — 24 GB, ~65 concurrent voice sessions with TQ4
    L4    — 24 GB, ~65 concurrent voice sessions with TQ4
    A100  — 80 GB, ~250 concurrent voice sessions with TQ4
    H100  — 80 GB, ~250 concurrent voice sessions with TQ4

Environment variables:
    VOICEQUANT_MODEL           — HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
    VOICEQUANT_GPU             — GPU type (default: T4)
    VOICEQUANT_TQ_BITS         — TurboQuant quantization bits, 3 or 4 (default: 4)
    VOICEQUANT_RESIDUAL_WINDOW — FP16 residual window size (default: 256)
    VOICEQUANT_MAX_CONCURRENT  — Max concurrent sequences (default: 64)
    HF_TOKEN                   — HuggingFace token for gated models

Session estimates (7B AWQ model, 4K context, TQ4):
    T4  (16 GB):  ~40 concurrent sessions  (~5x vs FP16)
    A10G (24 GB): ~65 concurrent sessions  (~5x vs FP16)
    L4  (24 GB):  ~65 concurrent sessions  (~5x vs FP16)
    A100 (80 GB): ~250 concurrent sessions (~5x vs FP16)
    H100 (80 GB): ~250 concurrent sessions (~5x vs FP16)
"""

from __future__ import annotations

import os

import modal

# ---------------------------------------------------------------------------
# Configuration from environment (with sensible defaults for voice AI)
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("VOICEQUANT_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
GPU_TYPE = os.environ.get("VOICEQUANT_GPU", "T4")
TQ_BITS = int(os.environ.get("VOICEQUANT_TQ_BITS", "4"))
RESIDUAL_WINDOW = int(os.environ.get("VOICEQUANT_RESIDUAL_WINDOW", "256"))
MAX_CONCURRENT = int(os.environ.get("VOICEQUANT_MAX_CONCURRENT", "64"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("VOICEQUANT_GPU_MEMORY", "0.90"))

# Map string GPU names to Modal GPU specs
_GPU_MAP = {
    "T4": modal.gpu.T4(),
    "A10G": modal.gpu.A10G(),
    "L4": modal.gpu.L4(),
    "A100": modal.gpu.A100(size="40GB"),
    "A100-80GB": modal.gpu.A100(size="80GB"),
    "H100": modal.gpu.H100(),
}

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("voicequant")

# Persistent volume for caching model weights across cold starts
model_cache = modal.Volume.from_name("voicequant-model-cache", create_if_missing=True)

# Container image: CUDA base with vLLM and VoiceQuant pre-installed
voicequant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "vllm>=0.8",
        "voicequant[cuda]",
        "httpx",
        "fastapi",
        "uvicorn[standard]",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "VOICEQUANT_MODEL": MODEL_ID,
            "VOICEQUANT_TQ_BITS": str(TQ_BITS),
        }
    )
)


def download_model_weights() -> None:
    """Download model weights at image build time so cold starts are fast.

    This function runs once when the Modal image is built and the weights
    are persisted in the attached volume.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        local_dir=f"/cache/huggingface/hub/{MODEL_ID.replace('/', '--')}",
        token=os.environ.get("HF_TOKEN"),
    )


# Build step: download weights into the volume during image build
voicequant_image = voicequant_image.run_function(
    download_model_weights,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)


# ---------------------------------------------------------------------------
# VoiceQuant inference class
# ---------------------------------------------------------------------------

@app.cls(
    image=voicequant_image,
    gpu=_GPU_MAP.get(GPU_TYPE, modal.gpu.T4()),
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    container_idle_timeout=300,
    allow_concurrent_inputs=MAX_CONCURRENT,
)
class VoiceQuant:
    """Modal class running a VoiceQuant-accelerated vLLM server.

    The server starts in __enter__ (container warm-up) and handles
    requests via the web endpoint.
    """

    @modal.enter()
    def start_server(self) -> None:
        """Start the vLLM server with TurboQuant KV cache compression.

        Called once when the container starts. Subsequent requests reuse
        the same engine instance.
        """
        import subprocess
        import time

        import httpx

        # Launch VoiceQuant server as a background process
        self._server_proc = subprocess.Popen(
            [
                "python", "-m", "voicequant.cli", "serve",
                "--model", MODEL_ID,
                "--tq-bits", str(TQ_BITS),
                "--tq-residual-window", str(RESIDUAL_WINDOW),
                "--max-concurrent", str(MAX_CONCURRENT),
                "--gpu-memory", str(GPU_MEMORY_UTILIZATION),
                "--host", "127.0.0.1",
                "--port", "8000",
            ],
            env={**os.environ, "HF_HOME": "/cache/huggingface"},
        )

        # Wait for the server to become healthy (up to 120s for model loading)
        client = httpx.Client(timeout=5.0)
        for _ in range(120):
            try:
                resp = client.get("http://127.0.0.1:8000/v1/health")
                if resp.status_code == 200:
                    health = resp.json()
                    if health.get("status") in ("healthy", "ready"):
                        print(f"VoiceQuant server ready: {health}")
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(1.0)

        raise RuntimeError("VoiceQuant server failed to start within 120 seconds")

    @modal.exit()
    def stop_server(self) -> None:
        """Gracefully terminate the inference server on container shutdown."""
        if hasattr(self, "_server_proc") and self._server_proc.poll() is None:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=10)

    @modal.web_endpoint(method="POST", docs=True)
    def chat(self, request: dict) -> dict:
        """OpenAI-compatible /v1/chat/completions endpoint.

        Proxies the request to the internal vLLM server.

        Args:
            request: OpenAI chat completion request body.

        Returns:
            OpenAI chat completion response.
        """
        import httpx

        # Forward the request to the internal server
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                "http://127.0.0.1:8000/v1/chat/completions",
                json=request,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()

    @modal.web_endpoint(method="GET", docs=True)
    def health(self) -> dict:
        """Health check endpoint for Modal readiness probes.

        Returns server status, model info, and capacity estimates.
        """
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get("http://127.0.0.1:8000/v1/health")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass

        return {"status": "unhealthy", "model": MODEL_ID}

    @modal.web_endpoint(method="GET", docs=True)
    def capacity(self) -> dict:
        """Return estimated concurrent session capacity for the current GPU."""
        import httpx

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get("http://127.0.0.1:8000/v1/capacity")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass

        return {"gpu": GPU_TYPE, "model": MODEL_ID, "tq_bits": TQ_BITS}


# ---------------------------------------------------------------------------
# Local entrypoint for testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """Test the deployment locally with a simple chat completion."""
    import json

    vq = VoiceQuant()

    # Health check
    health = vq.health.remote()
    print(f"Health: {json.dumps(health, indent=2)}")

    # Test chat completion
    response = vq.chat.remote(
        {
            "model": "default",
            "messages": [
                {"role": "system", "content": "You are a helpful voice assistant. Keep responses brief."},
                {"role": "user", "content": "What is VoiceQuant?"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }
    )
    print(f"Response: {json.dumps(response, indent=2)}")
