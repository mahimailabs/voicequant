"""Tests for benchmark infrastructure."""

import json
from pathlib import Path


PROMPTS_DIR = Path(__file__).parent.parent / "src" / "voicequant" / "benchmarks" / "prompts"


class TestBenchmarkPrompts:
    def test_short_system_prompt_exists(self):
        assert (PROMPTS_DIR / "short_system.txt").exists()

    def test_medium_system_prompt_exists(self):
        assert (PROMPTS_DIR / "medium_system.txt").exists()

    def test_long_system_prompt_exists(self):
        assert (PROMPTS_DIR / "long_system.txt").exists()

    def test_short_system_prompt_length(self):
        text = (PROMPTS_DIR / "short_system.txt").read_text()
        assert len(text) > 100

    def test_long_system_prompt_length(self):
        text = (PROMPTS_DIR / "long_system.txt").read_text()
        # Should be substantial (~1500 tokens = ~6000 chars)
        assert len(text) > 2000

    def test_conversation_files_exist(self):
        conv_dir = PROMPTS_DIR / "conversations"
        assert (conv_dir / "short_5_turn.json").exists()
        assert (conv_dir / "medium_10_turn.json").exists()
        assert (conv_dir / "long_20_turn.json").exists()

    def test_conversation_json_valid(self):
        conv_dir = PROMPTS_DIR / "conversations"
        for fname in ["short_5_turn.json", "medium_10_turn.json", "long_20_turn.json"]:
            data = json.loads((conv_dir / fname).read_text())
            assert "name" in data
            assert "turns" in data
            assert len(data["turns"]) >= 10  # 5 turns = 10 messages (user+assistant)

    def test_conversation_turn_counts(self):
        conv_dir = PROMPTS_DIR / "conversations"
        short = json.loads((conv_dir / "short_5_turn.json").read_text())
        medium = json.loads((conv_dir / "medium_10_turn.json").read_text())
        long_ = json.loads((conv_dir / "long_20_turn.json").read_text())
        assert len(short["turns"]) >= 10    # 5 turns = 10 messages
        assert len(medium["turns"]) >= 20   # 10 turns = 20 messages
        assert len(long_["turns"]) >= 40    # 20 turns = 40 messages


class TestBenchmarkScenarios:
    def test_multi_turn_importable(self):
        from voicequant.benchmarks.scenarios.multi_turn import MultiTurnBenchmark
        bench = MultiTurnBenchmark()
        assert bench.name == "multi_turn"

    def test_concurrent_importable(self):
        from voicequant.benchmarks.scenarios.concurrent import ConcurrentBenchmark
        bench = ConcurrentBenchmark()
        assert bench.name == "concurrent"

    def test_ttfb_importable(self):
        from voicequant.benchmarks.scenarios.ttfb import TTFBBenchmark
        bench = TTFBBenchmark()
        assert bench.name == "ttfb"

    def test_system_prompt_importable(self):
        from voicequant.benchmarks.scenarios.system_prompt import SystemPromptBenchmark
        bench = SystemPromptBenchmark()
        assert bench.name == "system_prompt"

    def test_tool_calling_importable(self):
        from voicequant.benchmarks.scenarios.tool_calling import ToolCallingBenchmark
        bench = ToolCallingBenchmark()
        assert bench.name == "tool_calling"

    def test_quality_importable(self):
        from voicequant.benchmarks.scenarios.quality import QualityBenchmark
        bench = QualityBenchmark()
        assert bench.name == "quality"


class TestBenchmarkRunner:
    def test_runner_importable(self):
        from voicequant.benchmarks.runner import run_benchmarks
        assert callable(run_benchmarks)


class TestBenchmarkReport:
    def test_report_importable(self):
        from voicequant.benchmarks.report import generate_report
        assert callable(generate_report)
