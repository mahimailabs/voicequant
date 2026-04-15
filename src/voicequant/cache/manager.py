"""Multi-session GPU memory manager (Phase 2)."""


class GPUMemoryManager:
    """Manages GPU memory budget across multiple CacheSessions.

    Phase 2: Track total GPU memory, auto-downgrade compression,
    LRU eviction when budget exceeded.
    """

    def __init__(self, budget_bytes: int | None = None):
        self._budget = budget_bytes
        self._sessions: dict[str, object] = {}
        raise NotImplementedError("GPUMemoryManager is planned for Phase 2")
