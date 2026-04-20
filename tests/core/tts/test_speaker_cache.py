"""SpeakerCache tests."""

from __future__ import annotations

import threading

from voicequant.core.tts.speaker_cache import SpeakerCache


def test_cache_instantiates_with_custom_size():
    c = SpeakerCache(max_size=7)
    assert c.stats()["max_size"] == 7
    assert c.stats()["size"] == 0


def test_put_and_get_round_trip():
    c = SpeakerCache(max_size=4)
    c.put("v1", [1, 2, 3])
    assert c.get("v1") == [1, 2, 3]


def test_get_miss_returns_none():
    c = SpeakerCache(max_size=4)
    assert c.get("nope") is None


def test_lru_eviction_drops_oldest():
    c = SpeakerCache(max_size=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.put("d", 4)  # evicts "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3
    assert c.get("d") == 4


def test_get_updates_recency():
    c = SpeakerCache(max_size=3)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    # Access "a" so it is most recently used.
    assert c.get("a") == 1
    c.put("d", 4)  # evicts "b" (least recent), not "a"
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3
    assert c.get("d") == 4


def test_clear_empties_cache():
    c = SpeakerCache(max_size=3)
    c.put("a", 1)
    c.put("b", 2)
    c.clear()
    assert c.stats()["size"] == 0
    assert c.get("a") is None


def test_stats_tracks_hits_and_misses():
    c = SpeakerCache(max_size=3)
    c.put("a", 1)
    c.get("a")  # hit
    c.get("a")  # hit
    c.get("b")  # miss
    s = c.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_rate"] == 2 / 3


def test_stats_hit_rate_zero_on_empty():
    c = SpeakerCache(max_size=3)
    assert c.stats()["hit_rate"] == 0.0


def test_thread_safety():
    c = SpeakerCache(max_size=64)
    errors: list[Exception] = []

    def worker(start: int) -> None:
        try:
            for i in range(100):
                key = f"v{(start + i) % 32}"
                c.put(key, i)
                c.get(key)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert c.stats()["size"] > 0
