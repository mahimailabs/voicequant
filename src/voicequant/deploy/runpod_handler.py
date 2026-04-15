"""VoiceQuant serverless handler for RunPod.

Usage:
    1. Build a Docker image with this handler and VoiceQuant installed.
    2. Upload to RunPod as a serverless endpoint.
    3. Send requests in OpenAI chat completion format.

    # Example request via RunPod API
    curl -X POST "https://api.runpod.ai/v2/<endpoint_id>/runsync" \\
        -H "Authorization: Bearer <RUNPOD_API_KEY>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a voice assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": false
            }
        }'

    # Streaming: set "stream": true to receive chunks via RunPod streaming.

Environment variables (set in RunPod template):
    VOICEQUANT_MODEL           — HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
    VOICEQUANT_TQ_BITS         — TurboQuant quantization bits (default: 4)
    VOICEQUANT_RESIDUAL_WINDOW — FP16 residual window size (default: 256)
    VOICEQUANT_MAX_CONCURRENT  — Max concurrent sequences (default: 64)
    VOICEQUANT_GPU_MEMORY      — GPU memory utilization 0.0-1.0 (default: 0.90)
    HF_TOKEN                   — HuggingFace token for gated models

This handler follows the RunPod pattern:
    - On first invocation (cold start), the vLLM server starts and stays warm.
    - Subsequent requests reuse the warm server for low-latency inference.
    - Supports both synchronous (runsync) and streaming responses.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Generator

import httpx
import runpod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("VOICEQUANT_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
TQ_BITS = int(os.environ.get("VOICEQUANT_TQ_BITS", "4"))
RESIDUAL_WINDOW = int(os.environ.get("VOICEQUANT_RESIDUAL_WINDOW", "256"))
MAX_CONCURRENT = int(os.environ.get("VOICEQUANT_MAX_CONCURRENT", "64"))
GPU_MEMORY = float(os.environ.get("VOICEQUANT_GPU_MEMORY", "0.90"))

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Global handle for the background server process
_server_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _ensure_server_running() -> None:
    """Start the VoiceQuant server if it is not already running.

    On RunPod serverless, the container persists between invocations as long
    as requests keep arriving within the idle timeout. This function starts
    the server on the first request and keeps it warm for subsequent ones.
    """
    global _server_proc

    # Already running?
    if _server_proc is not None and _server_proc.poll() is None:
        return

    print(f"Starting VoiceQuant server: model={MODEL_ID}, tq_bits={TQ_BITS}")

    _server_proc = subprocess.Popen(
        [
            "python", "-m", "voicequant.cli", "serve",
            "--model", MODEL_ID,
            "--tq-bits", str(TQ_BITS),
            "--tq-residual-window", str(RESIDUAL_WINDOW),
            "--max-concurrent", str(MAX_CONCURRENT),
            "--gpu-memory", str(GPU_MEMORY),
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
        ],
    )

    # Wait for the server to be ready (allow up to 180s for large models)
    client = httpx.Client(timeout=5.0)
    deadline = time.monotonic() + 180

    while time.monotonic() < deadline:
        try:
            resp = client.get(f"{SERVER_URL}/v1/health")
            if resp.status_code == 200:
                status = resp.json().get("status", "")
                if status in ("healthy", "ready"):
                    print("VoiceQuant server is ready.")
                    return
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(1.0)

    raise RuntimeError(
        "VoiceQuant server did not become healthy within 180 seconds. "
        "Check logs for model loading errors."
    )


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _build_openai_request(event_input: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible request from the RunPod event input.

    Accepts either a full OpenAI request body or a simplified format
    with just ``messages`` at the top level.
    """
    return {
        "model": event_input.get("model", "default"),
        "messages": event_input.get("messages", []),
        "max_tokens": event_input.get("max_tokens", 150),
        "temperature": event_input.get("temperature", 0.7),
        "stream": event_input.get("stream", False),
    }


def _sync_inference(request_body: dict[str, Any]) -> dict[str, Any]:
    """Send a synchronous chat completion request and return the response."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=request_body,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def _stream_inference(
    request_body: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Send a streaming chat completion request and yield SSE chunks.

    Each yielded dict is a partial OpenAI chat.completion.chunk object.
    RunPod's streaming infrastructure forwards these to the caller.
    """
    request_body["stream"] = True

    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{SERVER_URL}/v1/chat/completions",
            json=request_body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    return

                try:
                    chunk = json.loads(data)
                    # Extract the text delta for RunPod streaming
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield {"text": content, "chunk": chunk}
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict | Generator:
    """RunPod serverless handler entry point.

    Args:
        event: RunPod event dict with ``input`` containing the request.
            Expected input format (OpenAI chat completion):
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": false
            }

    Returns:
        For non-streaming: dict with the full OpenAI chat completion response.
        For streaming: generator yielding partial text chunks.
    """
    try:
        # Ensure the inference server is running (warm container pattern)
        _ensure_server_running()

        event_input = event.get("input", {})
        request_body = _build_openai_request(event_input)
        is_streaming = request_body.get("stream", False)

        if is_streaming:
            # Return a generator for RunPod streaming support
            return _stream_inference(request_body)

        # Synchronous response
        response = _sync_inference(request_body)

        # Extract the assistant message for a clean RunPod response
        choices = response.get("choices", [])
        output_text = ""
        if choices:
            message = choices[0].get("message", {})
            output_text = message.get("content", "")

        return {
            "text": output_text,
            "usage": response.get("usage", {}),
            "model": response.get("model", MODEL_ID),
            "full_response": response,
        }

    except Exception as e:
        return {"error": str(e), "error_type": type(e).__name__}


# ---------------------------------------------------------------------------
# RunPod entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Starting RunPod handler: model={MODEL_ID}, tq_bits={TQ_BITS}")
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )
