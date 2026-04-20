from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from voicequant.core.tts.speaker_cache import SpeakerCache


def test_cache_instantiation_custom_max_size():
    c = SpeakerCache(10)
    assert c.max_size == 10


def test_put_and_get_roundtrip():
    c = SpeakerCache(2)
    c.put("v1", [1, 2, 3])
    assert c.get("v1") == [1, 2, 3]


def test_lru_eviction_oldest_removed():
    c = SpeakerCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_get_updates_recency():
    c = SpeakerCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # a becomes newest
    c.put("c", 3)  # should evict b
    assert c.get("b") is None
    assert c.get("a") == 1


def test_clear_empties_cache():
    c = SpeakerCache(2)
    c.put("a", 1)
    c.clear()
    stats = c.stats()
    assert stats["size"] == 0


def test_stats_hit_miss_and_rate():
    c = SpeakerCache(2)
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("x") is None
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["hit_rate"] == 0.5


def test_thread_safety_concurrent_access():
    c = SpeakerCache(20)

    def worker(i: int) -> None:
        key = f"v{i % 8}"
        c.put(key, i)
        _ = c.get(key)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(500)))

    stats = c.stats()
    assert stats["size"] <= 20
    assert stats["hits"] >= 0
    assert stats["misses"] >= 0
