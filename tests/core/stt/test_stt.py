"""M2 STT foundation tests."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from voicequant.core.protocol import CapacityReport, HealthStatus, ModalityEngine
from voicequant.core.stt.compile import list_available_models
from voicequant.core.stt.config import STTConfig
from voicequant.core.stt.engine import STTEngine, TranscriptionResult
from voicequant.server.app import create_app
from voicequant.server.config import ServerConfig


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Install a stub faster_whisper module so `stt_available` resolves True."""
    if "faster_whisper" in sys.modules:
        yield
        return
    stub = ModuleType("faster_whisper")
    stub.WhisperModel = object  # unused when transcribe_bytes is patched
    stub.download_model = lambda *a, **k: ""
    monkeypatch.setitem(sys.modules, "faster_whisper", stub)
    yield
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)


def test_stt_config_defaults():
    cfg = STTConfig()
    assert cfg.model_name == "Systran/faster-whisper-large-v3"
    assert cfg.beam_size == 1
    assert cfg.vad_filter is True
    assert cfg.condition_on_previous_text is False
    assert cfg.device in {"cpu", "cuda"}


def test_stt_config_device_auto_resolves():
    cfg = STTConfig(device="auto")
    assert cfg.device in {"cpu", "cuda"}


def test_stt_config_custom_overrides():
    cfg = STTConfig(device="cpu", beam_size=5, compute_type="int8")
    assert cfg.device == "cpu"
    assert cfg.beam_size == 5
    assert cfg.compute_type == "int8"


def test_stt_engine_lazy_construction():
    engine = STTEngine(STTConfig(device="cpu"))
    assert engine._model is None
    assert engine._model_loaded is False


def test_stt_engine_satisfies_protocol():
    engine: ModalityEngine = STTEngine(STTConfig(device="cpu"))
    h = engine.health()
    assert isinstance(h, HealthStatus)
    assert h.modality == "stt"
    assert h.healthy is False
    assert h.detail == "not loaded"

    c = engine.capacity()
    assert isinstance(c, CapacityReport)
    assert c.active == 0
    assert c.saturated is False

    m = engine.metrics()
    assert set(m.keys()) >= {
        "transcriptions_total",
        "avg_latency_ms",
        "active_sessions",
        "model_loaded",
    }
    assert m["model_loaded"] == 0.0

    assert engine.shutdown() is None


def test_transcription_result_dataclass():
    r = TranscriptionResult(text="hello", language="en", duration=1.2, segments=[])
    assert r.text == "hello"
    assert r.language == "en"
    assert r.duration == 1.2
    assert r.segments == []


def test_stt_endpoint_json(fake_faster_whisper):
    app = create_app(ServerConfig())

    fake = TranscriptionResult(
        text="hello world",
        language="en",
        duration=1.5,
        segments=[{"id": 0, "start": 0.0, "end": 1.5, "text": "hello world"}],
    )

    with patch.object(STTEngine, "transcribe_bytes", lambda self, *a, **k: fake):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("t.wav", b"RIFFfakeWAV", "audio/wav")},
            )
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hello world"}


def test_stt_endpoint_verbose_json(fake_faster_whisper):
    app = create_app(ServerConfig())

    fake = TranscriptionResult(
        text="hi",
        language="en",
        duration=0.5,
        segments=[{"id": 0, "start": 0.0, "end": 0.5, "text": "hi"}],
    )

    with patch.object(STTEngine, "transcribe_bytes", lambda self, *a, **k: fake):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("t.wav", b"xxx", "audio/wav")},
                data={"response_format": "verbose_json"},
            )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["text"] == "hi"
    assert body["duration"] == 0.5
    assert body["segments"][0]["text"] == "hi"


def test_stt_endpoint_text(fake_faster_whisper):
    app = create_app(ServerConfig())

    fake = TranscriptionResult(
        text="plain text", language="en", duration=0.1, segments=[]
    )

    with patch.object(STTEngine, "transcribe_bytes", lambda self, *a, **k: fake):
        with TestClient(app) as client:
            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("t.wav", b"xxx", "audio/wav")},
                data={"response_format": "text"},
            )
    assert r.status_code == 200
    assert r.text == "plain text"


def test_stt_real_router_mounts_models_endpoint(fake_faster_whisper):
    app = create_app(ServerConfig())
    paths = {r.path for r in app.routes}
    assert "/v1/audio/transcriptions/models" in paths


def test_stt_stub_mounted_when_backend_absent(monkeypatch):
    # Ensure faster_whisper is NOT importable
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    import builtins as _b

    original_import = _b.__import__

    def _patched_import(name, *a, **k):
        if name == "faster_whisper":
            raise ImportError("simulated missing faster_whisper")
        return original_import(name, *a, **k)

    monkeypatch.setattr(_b, "__import__", _patched_import)

    app = create_app(ServerConfig())
    paths = {r.path for r in app.routes}
    # Stub router does NOT expose /models
    assert "/v1/audio/transcriptions/models" not in paths
    # Stub still returns 501
    from starlette.testclient import TestClient as _TC

    client = _TC(app, raise_server_exceptions=False)
    r = client.post("/v1/audio/transcriptions")
    assert r.status_code == 501


def test_list_available_models_nonempty():
    models = list_available_models()
    assert len(models) >= 3
    for m in models:
        assert "name" in m and "hf_id" in m and "size_mb" in m and "compute_type" in m


def test_cli_stt_help_works():
    from voicequant.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["stt", "--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.stdout
    assert "models" in result.stdout
    assert "download" in result.stdout


def test_cli_stt_models_runs():
    from voicequant.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["stt", "models"])
    assert result.exit_code == 0
    assert "large-v3" in result.stdout


def test_registry_has_stt_on_startup(fake_faster_whisper):
    app = create_app(ServerConfig())
    with TestClient(app):
        reg = app.state.registry
        assert reg.has("stt")
        eng = reg.get_engine("stt")
        assert isinstance(eng.health(), HealthStatus)
