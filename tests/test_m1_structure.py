"""M1 structural tests — engine contract, routers, registry, config, metrics."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from voicequant.core.protocol import CapacityReport, HealthStatus, ModalityEngine
from voicequant.server.app import create_app
from voicequant.server.config import ServerConfig
from voicequant.server.engine import EngineRegistry
from voicequant.server.metrics import MetricsRegistry


class MockEngine:
    def __init__(self, modality: str = "mock") -> None:
        self._modality = modality

    def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, modality=self._modality, detail="ok")

    def capacity(self) -> CapacityReport:
        return CapacityReport(
            active=1, headroom=10, saturated=False, latency_metric=42.0
        )

    def metrics(self) -> dict[str, float]:
        return {"foo": 1.0, "bar": 2.0}

    def shutdown(self) -> None:
        pass


def test_protocol_dataclasses():
    h = HealthStatus(healthy=True, modality="llm")
    assert h.healthy is True
    assert h.modality == "llm"
    assert h.detail is None

    c = CapacityReport(active=5, headroom=10, saturated=False, latency_metric=0.2)
    assert c.active == 5 and c.headroom == 10 and c.saturated is False
    assert c.latency_metric == 0.2


def test_modality_engine_protocol_satisfied_by_mock():
    m: ModalityEngine = MockEngine()
    assert isinstance(m.health(), HealthStatus)
    assert isinstance(m.capacity(), CapacityReport)
    assert m.metrics() == {"foo": 1.0, "bar": 2.0}
    assert m.shutdown() is None


def test_engine_registry_register_get():
    reg = EngineRegistry()
    e = MockEngine("stt")
    reg.register_engine("stt", e)
    assert reg.get_engine("stt") is e


def test_engine_registry_keyerror_for_unknown():
    reg = EngineRegistry()
    with pytest.raises(KeyError, match="Modality not registered"):
        reg.get_engine("nope")


def test_engine_registry_health_all_single():
    reg = EngineRegistry()
    reg.register_engine("llm", MockEngine("llm"))
    result = reg.health_all()
    assert set(result.keys()) == {"llm"}
    assert result["llm"].modality == "llm"


def test_engine_registry_health_all_multi():
    reg = EngineRegistry()
    reg.register_engine("llm", MockEngine("llm"))
    reg.register_engine("stt", MockEngine("stt"))
    result = reg.health_all()
    assert set(result.keys()) == {"llm", "stt"}


def test_engine_registry_capacity_all():
    reg = EngineRegistry()
    reg.register_engine("llm", MockEngine("llm"))
    reg.register_engine("tts", MockEngine("tts"))
    result = reg.capacity_all()
    assert set(result.keys()) == {"llm", "tts"}
    assert all(isinstance(v, CapacityReport) for v in result.values())


def test_engine_registry_metrics_all_prefixed():
    reg = EngineRegistry()
    reg.register_engine("llm", MockEngine("llm"))
    reg.register_engine("stt", MockEngine("stt"))
    merged = reg.metrics_all()
    assert merged == {
        "llm_foo": 1.0,
        "llm_bar": 2.0,
        "stt_foo": 1.0,
        "stt_bar": 2.0,
    }


def test_server_config_defaults_llm_only():
    cfg = ServerConfig()
    assert cfg.enabled_modalities == ["llm"]


def test_server_config_stt_adds_modality():
    cfg = ServerConfig(stt_config={"model": "whisper"})
    assert "stt" in cfg.enabled_modalities
    assert "llm" in cfg.enabled_modalities


def test_server_config_tts_adds_modality():
    cfg = ServerConfig(tts_config={"model": "kokoro"})
    assert "tts" in cfg.enabled_modalities


def test_server_config_both_modalities():
    cfg = ServerConfig(stt_config={"a": 1}, tts_config={"b": 2})
    assert set(cfg.enabled_modalities) >= {"llm", "stt", "tts"}


def test_metrics_registry_collect_all_prefixes_keys():
    reg = MetricsRegistry()
    reg.register_modality("llm", lambda: {"tokens": 100.0})
    reg.register_modality("stt", lambda: {"rtf": 0.15})
    merged = reg.collect_all()
    assert merged == {"llm_tokens": 100.0, "stt_rtf": 0.15}


def test_stt_stub_returns_501():
    app = create_app(ServerConfig())
    client = TestClient(app)
    r = client.post("/v1/audio/transcriptions")
    assert r.status_code == 501
    body = r.json()
    assert "error" in body
    assert "STT modality not installed" in body["error"]["message"]
    assert body["error"]["type"] == "not_implemented"


def test_tts_stub_returns_501():
    app = create_app(ServerConfig())
    client = TestClient(app)
    r = client.post("/v1/audio/speech")
    assert r.status_code == 501
    body = r.json()
    assert "error" in body
    assert "TTS modality not installed" in body["error"]["message"]
    assert body["error"]["type"] == "not_implemented"


def test_existing_endpoints_still_registered():
    app = create_app(ServerConfig())
    paths = {r.path for r in app.routes}
    for expected in (
        "/v1/chat/completions",
        "/v1/models",
        "/v1/health",
        "/v1/capacity",
        "/v1/kv-stats",
        "/metrics",
        "/v1/audio/transcriptions",
        "/v1/audio/speech",
    ):
        assert expected in paths, f"missing route {expected}"
