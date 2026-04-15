"""Tests for the server engine and config modules."""

from voicequant.server.config import (
    GPU_CAPACITY_ESTIMATES,
    VOICE_DEFAULTS,
    ServerConfig,
)


class TestServerConfig:
    def test_default_config(self):
        config = ServerConfig()
        assert config.model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
        assert config.kv_cache_dtype == "tq4"
        assert config.port == 8000
        assert config.max_num_seqs == 64

    def test_custom_config(self):
        config = ServerConfig(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            kv_cache_dtype="tq3",
            port=9000,
        )
        assert config.model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.kv_cache_dtype == "tq3"
        assert config.port == 9000

    def test_voice_defaults_exist(self):
        assert "kv_cache_dtype" in VOICE_DEFAULTS
        assert "residual_window" in VOICE_DEFAULTS
        assert "default_max_tokens" in VOICE_DEFAULTS
        assert VOICE_DEFAULTS["default_max_tokens"] <= 200

    def test_gpu_capacity_estimates(self):
        assert "T4" in GPU_CAPACITY_ESTIMATES
        assert "A100" in GPU_CAPACITY_ESTIMATES
        t4 = GPU_CAPACITY_ESTIMATES["T4"]
        assert t4["tq4_sessions"] > t4["fp16_sessions"]


class TestServerApp:
    def test_create_app(self):
        from voicequant.server.app import create_app

        config = ServerConfig()
        app = create_app(config)
        assert app is not None
        assert app.title == "VoiceQuant"

    def test_app_has_routes(self):
        from voicequant.server.app import create_app

        config = ServerConfig()
        app = create_app(config)
        routes = [r.path for r in app.routes]
        assert "/v1/chat/completions" in routes
        assert "/v1/models" in routes
        assert "/v1/health" in routes
        assert "/v1/capacity" in routes
        assert "/v1/kv-stats" in routes
        assert "/metrics" in routes
