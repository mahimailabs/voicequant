"""VoiceQuant LLM plugin for LiveKit Agents (Phase 3).

Wraps a HuggingFace model + TurboQuantEngine to implement
the livekit.agents.llm.LLM abstract interface.
"""

from __future__ import annotations


class LLM:
    """VoiceQuant-compressed LLM for LiveKit Agents.

    Phase 3: Subclass livekit.agents.llm.LLM, implement chat() -> LLMStream,
    maintain CacheSession per conversation for compressed KV state.
    """

    def __init__(self, model: str = "", total_bits: int = 3,
                 device: str = "cuda", **kwargs):
        raise NotImplementedError(
            "VoiceQuant LiveKit LLM plugin is planned for Phase 3. "
            "Install livekit-agents>=1.4 and check back for updates."
        )
