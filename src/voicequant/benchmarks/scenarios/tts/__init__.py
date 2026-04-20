"""TTS benchmark scenarios.

Each scenario class exposes a ``run(model, config, **kwargs)`` method and
returns a dict with ``results``, a ``summary``, and a ``simulated`` flag
so the shared runner/report/visualize pipeline can consume them without
special-casing.
"""

from voicequant.benchmarks.scenarios.tts.concurrent import ConcurrentTTSScenario
from voicequant.benchmarks.scenarios.tts.mos_quality import MOSQualityScenario
from voicequant.benchmarks.scenarios.tts.speaker_cache_hit import (
    SpeakerCacheHitScenario,
)
from voicequant.benchmarks.scenarios.tts.streaming_jitter import (
    StreamingJitterScenario,
)
from voicequant.benchmarks.scenarios.tts.ttfa import TTFAScenario

__all__ = [
    "TTFAScenario",
    "StreamingJitterScenario",
    "MOSQualityScenario",
    "ConcurrentTTSScenario",
    "SpeakerCacheHitScenario",
]
