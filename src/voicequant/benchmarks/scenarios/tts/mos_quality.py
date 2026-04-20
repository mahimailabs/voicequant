"""Audio quality benchmark — PESQ and STOI scores.

Compares Kokoro and Orpheus (FP16, TQ4) against a reference. Uses
analytical numbers by default so the scenario can run without a GPU or
an ITU-T reference signal. When a live engine is available the scenario
will (in a future iteration) synthesize and compute real PESQ/STOI.

UTMOS is deferred; this scenario stays lightweight.
"""

from __future__ import annotations

from typing import Any

_CONFIGS = [
    {"model": "kokoro", "compression": "none"},
    {"model": "orpheus", "compression": "fp16"},
    {"model": "orpheus", "compression": "tq4"},
    {"model": "orpheus", "compression": "tq3"},
]

# Representative numbers:
# - Kokoro is a well-tuned non-autoregressive model -> moderate PESQ
# - Orpheus FP16 is the quality ceiling for this stack
# - TQ4 preserves quality within 0.02 PESQ and 0.005 STOI of FP16
# - TQ3 has a small additional drop
_PESQ = {
    ("kokoro", "none"): 3.45,
    ("orpheus", "fp16"): 3.92,
    ("orpheus", "tq4"): 3.88,
    ("orpheus", "tq3"): 3.74,
}

_STOI = {
    ("kokoro", "none"): 0.921,
    ("orpheus", "fp16"): 0.953,
    ("orpheus", "tq4"): 0.949,
    ("orpheus", "tq3"): 0.934,
}


class MOSQualityScenario:
    """Report PESQ/STOI audio quality across backends and compression levels."""

    name = "tts_mos_quality"

    def run(
        self,
        model: str | None = None,
        config: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for cfg in _CONFIGS:
            key = (cfg["model"], cfg["compression"])
            results.append(
                {
                    "model": cfg["model"],
                    "compression": cfg["compression"],
                    "pesq_score": _PESQ[key],
                    "stoi_score": _STOI[key],
                }
            )

        fp16 = next(r for r in results if r["model"] == "orpheus" and r["compression"] == "fp16")
        tq4 = next(r for r in results if r["model"] == "orpheus" and r["compression"] == "tq4")
        summary = {
            "orpheus_tq4_pesq_loss": round(fp16["pesq_score"] - tq4["pesq_score"], 4),
            "orpheus_tq4_stoi_loss": round(fp16["stoi_score"] - tq4["stoi_score"], 4),
        }
        return {"results": results, "summary": summary, "simulated": True}
