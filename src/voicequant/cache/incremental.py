"""Incremental compression for streaming token-by-token (Phase 2)."""


class IncrementalCompressor:
    """Compress new KV entries without reprocessing existing ones.

    Phase 2: Accept single-token KV updates, maintain running
    compressed state, support append-only and full-recompress modes.
    """

    def __init__(self, engine):
        self._engine = engine
        raise NotImplementedError("IncrementalCompressor is planned for Phase 2")
