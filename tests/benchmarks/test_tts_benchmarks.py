"""Tests for M5 TTS benchmark scenarios and fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicequant.benchmarks.runner import _load_scenarios, list_scenarios
from voicequant.benchmarks.scenarios.tts import (
    ConcurrentTTSScenario,
    MOSQualityScenario,
    SpeakerCacheHitScenario,
    StreamingJitterScenario,
    TTFAScenario,
)

SCENARIO_CLASSES = [
    TTFAScenario,
    StreamingJitterScenario,
    MOSQualityScenario,
    ConcurrentTTSScenario,
    SpeakerCacheHitScenario,
]


@pytest.mark.parametrize("cls", SCENARIO_CLASSES)
def test_scenario_instantiates(cls):
    scenario = cls()
    assert scenario is not None


@pytest.mark.parametrize("cls", SCENARIO_CLASSES)
def test_scenario_run_returns_expected_keys(cls):
    result = cls().run()
    assert isinstance(result, dict)
    assert "results" in result
    assert "summary" in result
    assert "simulated" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


def test_runner_registers_tts_scenarios():
    _load_scenarios()
    names = list_scenarios()
    for expected in (
        "tts_ttfa",
        "tts_streaming_jitter",
        "tts_mos_quality",
        "tts_concurrent",
        "tts_speaker_cache_hit",
    ):
        assert expected in names, f"missing scenario: {expected}"


def test_runner_still_has_llm_scenarios():
    _load_scenarios()
    names = list_scenarios()
    for expected in ("multi_turn", "concurrent", "ttfb"):
        assert expected in names


def test_tts_concurrent_has_headline_ratio():
    result = ConcurrentTTSScenario().run()
    assert "headline_ratio_tq4_over_fp16" in result
    # Orpheus TQ4 should fit more than FP16
    assert result["headline_ratio_tq4_over_fp16"] >= 1.0


def test_tts_ttfa_compares_modes():
    result = TTFAScenario().run()
    rows = result["results"]
    modes = {r["mode"] for r in rows}
    assert modes == {"streaming", "non-streaming"}
    # Streaming TTFA should be strictly lower than non-streaming for same model/length
    for r in rows:
        if r["mode"] == "streaming":
            pair = [
                o
                for o in rows
                if o["model"] == r["model"]
                and o["text_length"] == r["text_length"]
                and o["mode"] == "non-streaming"
            ]
            assert pair and r["ttfa_ms"] <= pair[0]["ttfa_ms"]


def test_test_sentences_fixture_exists_and_has_15_plus_entries():
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "voicequant"
        / "benchmarks"
        / "prompts"
        / "tts"
        / "test_sentences.json"
    )
    assert path.exists(), f"missing fixture: {path}"
    data = json.loads(path.read_text())
    total = sum(len(v) for v in data.values())
    assert total >= 15, f"fewer than 15 sentences: {total}"
    # Every entry has required fields
    for bucket in data.values():
        for entry in bucket:
            assert "text" in entry and isinstance(entry["text"], str)
            assert "expected_duration_range" in entry
            assert len(entry["expected_duration_range"]) == 2


def test_report_generates_tts_section(tmp_path):
    from voicequant.benchmarks.report import generate_report

    results = {
        "tts_ttfa": TTFAScenario().run(),
        "tts_concurrent": ConcurrentTTSScenario().run(),
        "tts_mos_quality": MOSQualityScenario().run(),
    }
    out = tmp_path / "report.md"
    generate_report(results, str(out))
    text = out.read_text()
    assert "TTS (Text-to-Speech)" in text
    assert "Time-to-First-Audio" in text
    assert "Concurrent Voice Streams per GPU" in text


def test_visualize_tts_charts(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    assert matplotlib is not None
    from voicequant.benchmarks.visualize import generate_tts_charts

    paths = generate_tts_charts(str(tmp_path), fmt="png", include_hero=True)
    assert len(paths) == 6
    names = [Path(p).name for p in paths]
    assert "tts_ttfa.png" in names
    assert "tts_concurrent_streams.png" in names
    assert "cross_modality_hero.png" in names
    for p in paths:
        assert Path(p).exists()
