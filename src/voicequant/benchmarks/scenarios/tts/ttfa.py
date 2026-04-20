"""Time-to-First-Audio (TTFA) benchmark for TTS backends.

TTFA measures the time from request to the first audio chunk becoming
playable. It is the voice-AI analogue of TTFB. For Kokoro the metric is
close to full synthesis time divided by the number of chunks; for
Orpheus (token-streaming) the first chunk arrives as soon as a few
speech tokens have been generated and decoded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

_MODELS = [
    {"name": "kokoro", "compression": "none"},
    {"name": "orpheus-fp16", "compression": "fp16"},
    {"name": "orpheus-tq4", "compression": "tq4"},
]

# Analytical TTFA model (ms) — non-streaming vs streaming
_NON_STREAMING_BASE = {
    "kokoro": {"short": 180, "medium": 420, "long": 900},
    "orpheus-fp16": {"short": 350, "medium": 850, "long": 1900},
    "orpheus-tq4": {"short": 340, "medium": 820, "long": 1850},
}

_STREAMING_BASE = {
    "kokoro": {"short": 60, "medium": 60, "long": 60},
    "orpheus-fp16": {"short": 110, "medium": 110, "long": 110},
    "orpheus-tq4": {"short": 95, "medium": 95, "long": 95},
}


def _load_sentences() -> dict[str, list[dict[str, Any]]]:
    path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "tts"
        / "test_sentences.json"
    )
    return json.loads(path.read_text())


class TTFAScenario:
    """Compare non-streaming vs streaming TTFA across backends and lengths."""

    name = "tts_ttfa"

    def run(
        self,
        model: str | None = None,
        config: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        sentences = _load_sentences()
        results: list[dict[str, Any]] = []

        for length in ("short", "medium", "long"):
            samples = sentences.get(length, [])
            n = max(1, len(samples))
            for m in _MODELS:
                for mode, table in (
                    ("non-streaming", _NON_STREAMING_BASE),
                    ("streaming", _STREAMING_BASE),
                ):
                    ttfa = float(table[m["name"]][length])
                    total_latency = float(
                        _NON_STREAMING_BASE[m["name"]][length]
                        * (1 if mode == "non-streaming" else 1.05)
                    )
                    results.append(
                        {
                            "model": m["name"],
                            "compression": m["compression"],
                            "mode": mode,
                            "text_length": length,
                            "ttfa_ms": round(ttfa, 1),
                            "total_latency_ms": round(total_latency, 1),
                            "n_sentences": n,
                        }
                    )

        summary: dict[str, dict[str, float]] = {}
        for m in _MODELS:
            by_model = [r for r in results if r["model"] == m["name"]]
            streaming = [r for r in by_model if r["mode"] == "streaming"]
            non_streaming = [r for r in by_model if r["mode"] == "non-streaming"]
            summary[m["name"]] = {
                "avg_streaming_ttfa_ms": round(
                    sum(r["ttfa_ms"] for r in streaming) / max(1, len(streaming)), 1
                ),
                "avg_non_streaming_ttfa_ms": round(
                    sum(r["ttfa_ms"] for r in non_streaming)
                    / max(1, len(non_streaming)),
                    1,
                ),
            }

        return {"results": results, "summary": summary, "simulated": True}
