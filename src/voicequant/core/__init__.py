"""VoiceQuant core — TurboQuant compression engine."""

import sys as _sys
import importlib as _importlib

# Eagerly alias non-cuTile submodules so legacy
# "from voicequant.core.<name> import X" paths keep resolving.
# compress/decompress/attention import cuTile at module load; they're kept
# lazy (engine.py imports them inside methods, as it always did).
for _name in ("engine", "codebook", "config", "constants", "wrapper", "validator"):
    _sys.modules[f"voicequant.core.{_name}"] = _importlib.import_module(
        f"voicequant.core.llm.{_name}"
    )


def __getattr__(name: str):
    if name in ("compress", "decompress", "attention"):
        mod = _importlib.import_module(f"voicequant.core.llm.{name}")
        _sys.modules[f"voicequant.core.{name}"] = mod
        return mod
    if name == "TurboQuantEngine":
        from voicequant.core.llm.engine import TurboQuantEngine
        return TurboQuantEngine
    if name == "LloydMaxCodebook":
        from voicequant.core.llm.codebook import LloydMaxCodebook
        return LloydMaxCodebook
    if name == "TurboQuantConfig":
        from voicequant.core.llm.config import TurboQuantConfig
        return TurboQuantConfig
    if name == "TurboQuantWrapper":
        from voicequant.core.llm.wrapper import TurboQuantWrapper
        return TurboQuantWrapper
    raise AttributeError(f"module 'voicequant.core' has no attribute {name!r}")


from voicequant.core.llm.engine import TurboQuantEngine
from voicequant.core.llm.codebook import LloydMaxCodebook
from voicequant.core.llm.config import TurboQuantConfig
from voicequant.core.llm.wrapper import TurboQuantWrapper

__all__ = [
    "TurboQuantEngine",
    "LloydMaxCodebook",
    "TurboQuantConfig",
    "TurboQuantWrapper",
]
