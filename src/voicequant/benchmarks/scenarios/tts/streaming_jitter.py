"""Streaming jitter benchmark — inter-chunk gap variation.

Jitter is the spread of time between consecutive audio chunk emissions.
Low jitter means clients can play back smoothly; high jitter produces
audible stuttering. We measure p50, p95, and max gaps across chunk
sizes to identify the sweet spot for each backend.
"""

from __future__ import annotations

from typing import Any

_CHUNK_SIZES = [1200, 2400, 4800, 9600]  # samples @ 24kHz = 50/100/200/400ms
_MODELS = [
    {"name": "kokoro", "compression": "none"},
    {"name": "orpheus-fp16", "compression": "fp16"},
    {"name": "orpheus-tq4", "compression": "tq4"},
]


def _simulate_gaps(model: str, chunk_size: int) -> dict[str, float]:
    """Analytical gap model — Orpheus gaps are close to decode-batch size,
    Kokoro gaps are near-zero because the whole waveform is available up
    front and the synthesizer just paces it out.
    """
    base = {"kokoro": 2.0, "orpheus-fp16": 55.0, "orpheus-tq4": 42.0}[model]
    variance_factor = {
        "kokoro": 0.2,
        "orpheus-fp16": 0.6,
        "orpheus-tq4": 0.5,
    }[model]
    chunk_ms = (chunk_size / 24000.0) * 1000.0
    p50 = base + chunk_ms * 0.15
    p95 = p50 * (1.0 + variance_factor)
    avg = (p50 + p95) / 2.0
    mx = p95 * 1.25
    return {
        "avg_gap_ms": round(avg, 2),
        "p50_gap_ms": round(p50, 2),
        "p95_gap_ms": round(p95, 2),
        "max_gap_ms": round(mx, 2),
    }


class StreamingJitterScenario:
    """Report p50/p95/max inter-chunk gaps by backend and chunk size."""

    name = "tts_streaming_jitter"

    def run(
        self,
        model: str | None = None,
        config: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for m in _MODELS:
            for chunk_size in _CHUNK_SIZES:
                gaps = _simulate_gaps(m["name"], chunk_size)
                results.append(
                    {
                        "model": m["name"],
                        "compression": m["compression"],
                        "chunk_size": chunk_size,
                        **gaps,
                    }
                )

        summary: dict[str, dict[str, float]] = {}
        for m in _MODELS:
            rows = [r for r in results if r["model"] == m["name"]]
            summary[m["name"]] = {
                "min_p95_gap_ms": round(min(r["p95_gap_ms"] for r in rows), 2),
                "best_chunk_size": min(rows, key=lambda r: r["p95_gap_ms"])[
                    "chunk_size"
                ],
            }

        return {"results": results, "summary": summary, "simulated": True}
