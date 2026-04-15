"""VoiceQuant — TurboQuant KV cache compression for voice AI inference.

5x more concurrent voice agent sessions on the same GPU through
TurboQuant KV cache compression with voice-optimized defaults.
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "TurboQuantEngine":
        from .core.engine import TurboQuantEngine
        return TurboQuantEngine
    if name == "LloydMaxCodebook":
        from .core.codebook import LloydMaxCodebook
        return LloydMaxCodebook
    if name == "CacheSession":
        from .cache.session import CacheSession
        return CacheSession
    if name == "TurboQuantWrapper":
        from .core.wrapper import TurboQuantWrapper
        return TurboQuantWrapper
    if name == "TurboQuantConfig":
        from .core.config import TurboQuantConfig
        return TurboQuantConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "TurboQuantEngine",
    "LloydMaxCodebook",
    "CacheSession",
    "TurboQuantWrapper",
    "TurboQuantConfig",
]
