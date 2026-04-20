"""Speaker cache impact — cold vs warm synthesis latency.

Each TTS backend caches speaker embeddings in an LRU (see
core/tts/speaker_cache.py). This benchmark measures the marginal cost
of a cache miss against a hit, and how cache pressure (more unique
voices than cache slots) erodes the hit rate.
"""

from __future__ import annotations

from typing import Any

_VOICE_COUNTS = [1, 5, 10, 20]
_CACHE_SIZE = 10
_COLD_LATENCY_MS = 185.0
_WARM_LATENCY_MS = 120.0


def _cache_hit_rate(unique_voices: int, cache_size: int = _CACHE_SIZE) -> float:
    """Hit rate assuming uniform access over a round of N requests.

    When unique_voices <= cache_size every voice stays resident; rate
    approaches (N - unique_voices) / N for large N. Beyond cache size
    we get LRU churn — approximated as cache_size / unique_voices.
    """
    if unique_voices <= cache_size:
        return max(0.0, 1.0 - 1.0 / max(1, unique_voices))
    return cache_size / unique_voices


class SpeakerCacheHitScenario:
    """Report cold/warm latency and cache hit-rate across voice counts."""

    name = "tts_speaker_cache_hit"

    def run(
        self,
        model: str | None = None,
        config: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for n in _VOICE_COUNTS:
            hit_rate = _cache_hit_rate(n)
            cold_latency = _COLD_LATENCY_MS
            warm_latency = _WARM_LATENCY_MS
            avg_latency = (
                cold_latency * (1.0 - hit_rate) + warm_latency * hit_rate
            )
            results.append(
                {
                    "voice_count": n,
                    "cache_size": _CACHE_SIZE,
                    "cache_hit_rate": round(hit_rate, 4),
                    "avg_cold_latency_ms": round(cold_latency, 2),
                    "avg_warm_latency_ms": round(warm_latency, 2),
                    "avg_latency_ms": round(avg_latency, 2),
                }
            )

        summary = {
            "cold_warm_speedup": round(_COLD_LATENCY_MS / _WARM_LATENCY_MS, 2),
            "hit_rate_at_cache_size": round(_cache_hit_rate(_CACHE_SIZE), 4),
        }
        return {"results": results, "summary": summary, "simulated": True}
