"""TTS router tests (real + stub)."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from voicequant.core.tts.engine import SynthesisResult, TTSEngine
from voicequant.server.app import create_app
from voicequant.server.config import ServerConfig


@pytest.fixture
def fake_kokoro_onnx(monkeypatch):
    """Install a stub kokoro_onnx module so tts_available resolves True."""
    if "kokoro_onnx" in sys.modules:
        yield
        return
    stub = ModuleType("kokoro_onnx")
    stub.Kokoro = object  # unused when synthesize is patched
    monkeypatch.setitem(sys.modules, "kokoro_onnx", stub)
    yield
    monkeypatch.delitem(sys.modules, "kokoro_onnx", raising=False)


def test_tts_stub_returns_501_when_backend_absent(monkeypatch):
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
    r = client.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 501
    body = r.json()
    assert "error" in body
    assert "TTS modality not installed" in body["error"]["message"]
    assert body["error"]["type"] == "not_implemented"


def test_tts_real_endpoint_returns_audio(fake_kokoro_onnx):
    app = create_app(ServerConfig())

    fake = SynthesisResult(
        audio_bytes=b"RIFF0000WAVEfakeaudio",
        sample_rate=24000,
        duration_seconds=1.0,
        format="wav",
        voice="af_heart",
    )

    with patch.object(TTSEngine, "synthesize", lambda self, *a, **k: fake):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hello world", "voice": "af_heart"},
            )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.content == b"RIFF0000WAVEfakeaudio"
    assert "attachment" in r.headers.get("content-disposition", "")


def test_tts_real_endpoint_missing_input_returns_422(fake_kokoro_onnx):
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        r = client.post("/v1/audio/speech", json={"voice": "af_heart"})
    assert r.status_code == 422


def test_tts_real_endpoint_pcm_content_type(fake_kokoro_onnx):
    app = create_app(ServerConfig())

    fake = SynthesisResult(
        audio_bytes=b"\x00" * 1000,
        sample_rate=24000,
        duration_seconds=0.02,
        format="pcm",
        voice="af_heart",
    )

    with patch.object(TTSEngine, "synthesize", lambda self, *a, **k: fake):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "response_format": "pcm"},
            )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/pcm"


def test_tts_voices_endpoint_returns_list(fake_kokoro_onnx):
    app = create_app(ServerConfig())
    with TestClient(app) as client:
        r = client.get("/v1/audio/speech/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0
    assert "id" in body["data"][0]


def test_tts_real_router_mounts_voices_endpoint(fake_kokoro_onnx):
    app = create_app(ServerConfig())
    paths = {r.path for r in app.routes}
    assert "/v1/audio/speech/voices" in paths


def test_tts_stub_has_no_voices_endpoint(monkeypatch):
    monkeypatch.delitem(sys.modules, "kokoro_onnx", raising=False)
    import builtins

    original_import = builtins.__import__

    def _patched(name, *a, **k):
        if name == "kokoro_onnx":
            raise ImportError("simulated missing kokoro_onnx")
        return original_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _patched)

    app = create_app(ServerConfig())
    paths = {r.path for r in app.routes}
    assert "/v1/audio/speech/voices" not in paths


def test_registry_has_tts_on_startup(fake_kokoro_onnx):
    app = create_app(ServerConfig())
    with TestClient(app):
        reg = app.state.registry
        assert reg.has("tts")


def test_cli_tts_help_works():
    from typer.testing import CliRunner

    from voicequant.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["tts", "--help"])
    assert result.exit_code == 0
    assert "voices" in result.stdout
    assert "speak" in result.stdout


def test_cli_tts_voices_runs():
    from typer.testing import CliRunner

    from voicequant.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["tts", "voices"])
    assert result.exit_code == 0
    assert "af_heart" in result.stdout
