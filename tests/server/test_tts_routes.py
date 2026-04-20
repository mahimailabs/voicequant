from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from voicequant.server.routes.tts import build_router as build_tts_router
from voicequant.server.routes.tts_stub import router as tts_stub_router


class _MockTTSEngine:
    def synthesize(self, text: str, voice: str | None = None, output_format: str | None = None):
        return type(
            "R",
            (),
            {
                "audio_bytes": b"RIFF....",
                "sample_rate": 24000,
                "duration_seconds": 0.1,
                "format": output_format or "wav",
                "voice": voice or "af_heart",
            },
        )()

    def list_voices(self):
        return ["af_heart", "bf_ember"]


def test_tts_stub_returns_501():
    app = FastAPI()
    app.include_router(tts_stub_router)
    client = TestClient(app)
    r = client.post("/v1/audio/speech")
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["type"] == "not_implemented"


def test_real_tts_post_returns_audio_bytes():
    app = FastAPI()
    app.include_router(build_tts_router(lambda: _MockTTSEngine()))
    client = TestClient(app)
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "af_heart", "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert len(r.content) > 0


def test_real_tts_missing_input_field_422():
    app = FastAPI()
    app.include_router(build_tts_router(lambda: _MockTTSEngine()))
    client = TestClient(app)
    r = client.post("/v1/audio/speech", json={"voice": "af_heart"})
    assert r.status_code == 422


def test_real_tts_voices_list():
    app = FastAPI()
    app.include_router(build_tts_router(lambda: _MockTTSEngine()))
    client = TestClient(app)
    r = client.get("/v1/audio/speech/voices")
    assert r.status_code == 200
    payload = r.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) >= 1


def test_app_mounts_real_tts_router_when_backend_importable(monkeypatch):
    import sys
    from types import ModuleType

    from voicequant.server.app import create_app
    from voicequant.server.config import ServerConfig

    mod = ModuleType("kokoro_onnx")
    mod.Kokoro = object
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)

    app = create_app(ServerConfig())
    paths = {r.path for r in app.routes}
    assert "/v1/audio/speech/voices" in paths
