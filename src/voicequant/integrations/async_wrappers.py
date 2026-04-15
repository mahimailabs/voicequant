"""Async wrappers that run CUDA ops in executor threads (Phase 2)."""


async def async_compress(engine, past_key_values):
    raise NotImplementedError("async_compress is planned for Phase 2")


async def async_build_cache(engine, compressed):
    raise NotImplementedError("async_build_cache is planned for Phase 2")
