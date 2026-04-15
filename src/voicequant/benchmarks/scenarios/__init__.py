"""Benchmark scenarios for voice AI workloads."""

from voicequant.benchmarks.scenarios.concurrent import ConcurrentBenchmark
from voicequant.benchmarks.scenarios.multi_turn import MultiTurnBenchmark
from voicequant.benchmarks.scenarios.quality import QualityBenchmark
from voicequant.benchmarks.scenarios.system_prompt import SystemPromptBenchmark
from voicequant.benchmarks.scenarios.tool_calling import ToolCallingBenchmark
from voicequant.benchmarks.scenarios.ttfb import TTFBBenchmark

__all__ = [
    "MultiTurnBenchmark",
    "ConcurrentBenchmark",
    "TTFBBenchmark",
    "SystemPromptBenchmark",
    "ToolCallingBenchmark",
    "QualityBenchmark",
]
