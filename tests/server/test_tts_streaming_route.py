"""Tests for the streaming TTS endpoint (M5 Phase 2)."""

from __future__ import annotations

import base64
import json
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from voicequant.core.tts.engine import SynthesisResult, TTSEngine
from voicequant.core.tts.streaming import StreamingChunk, StreamingSynthesizer
from voicequant.server.app import create_app
from voicequant.server.config import ServerConfig


@pytest.fixture
def fake_kokoro_onnx(monkeypatch):
    """Install a stub kokoro_onnx so tts_available resolves True."""
    if "kokoro_onnx" in sys.modules:
        yield
        return
    stub = ModuleType("kokoro_onnx")
    stub.Kokoro = object
    monkeypatch.setitem(sys.modules, "kokoro_onnx", stub)
    yield
    monkeypatch.delitem(sys.modules, "kokoro_onnx", raising=False)


def _fake_chunks(n: int = 3) -> list[StreamingChunk]:
    return [
        StreamingChunk(
            audio_bytes=b"\x11\x22" * 100,
            chunk_index=i,
            is_final=(i == n - 1),
            timestamp_ms=float(i * 40),
            samples_count=100,
            duration_ms=4.16,
        )
        for i in range(n)
    ]


def test_streaming_endpoint_returns_chunked_audio(fake_kokoro_onnx):
    app = create_app(ServerConfig())
    chunks = _fake_chunks(3)

    with patch.object(StreamingSynthesizer, "stream", lambda self, text, voice=None: iter(chunks)):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech/stream",
                json={"input": "hello", "response_format": "pcm"},
            )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/pcm")
    # Concatenated chunk bytes
    assert r.content == b"".join(c.audio_bytes for c in chunks)


def test_streaming_endpoint_sse_when_accept_event_stream(fake_kokoro_onnx):
    app = create_app(ServerConfig())
    chunks = _fake_chunks(2)

    with patch.object(StreamingSynthesizer, "stream", lambda self, text, voice=None: iter(chunks)):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech/stream",
                json={"input": "hi"},
                headers={"Accept": "text/event-stream"},
            )
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert "event: audio" in body
    assert "event: done" in body
    # Grab one audio event payload and confirm structure
    audio_lines = [line for line in body.splitlines() if line.startswith("data: ")]
    assert audio_lines, "no SSE data lines found"
    # First data line should be a valid JSON object with required fields
    payload = json.loads(audio_lines[0][len("data: "):])
    assert {"chunk_index", "audio", "is_final", "duration_ms"} <= set(payload.keys())
    # audio is base64-encoded bytes
    decoded = base64.b64decode(payload["audio"])
    assert isinstance(decoded, bytes) and len(decoded) > 0


def test_non_streaming_endpoint_still_returns_full_audio(fake_kokoro_onnx):
    """stream=false (default) must still hit the normal synthesize path."""
    app = create_app(ServerConfig())
    result = SynthesisResult(
        audio_bytes=b"RIFFfakepayload",
        sample_rate=24000,
        duration_seconds=1.0,
        format="wav",
        voice="af_heart",
    )
    with patch.object(TTSEngine, "synthesize", lambda self, *a, **k: result):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hello"},
            )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content == b"RIFFfakepayload"


def test_speech_endpoint_stream_true_routes_to_stream(fake_kokoro_onnx):
    """When the request body has stream=true, /audio/speech streams."""
    app = create_app(ServerConfig())
    chunks = _fake_chunks(2)

    with patch.object(StreamingSynthesizer, "stream", lambda self, text, voice=None: iter(chunks)):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "stream": True, "response_format": "pcm"},
            )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.content == b"".join(c.audio_bytes for c in chunks)


def test_streaming_stub_returns_501(monkeypatch):
    """When kokoro_onnx is absent the stub 501 endpoint is mounted."""
    monkeypatch.delitem(sys.modules, "kokoro_onnx", raising=False)
    import builtins

    original_import = builtins.__import__

    def _patched(name, *a, **k):
        if name == "kokoro_onnx":
            raise ImportError("simulated missing kokoro_onnx")
        return original_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _patched)

    app = create_app(ServerConfig())
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/audio/speech/stream", json={"input": "hi"})
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["type"] == "not_implemented"
