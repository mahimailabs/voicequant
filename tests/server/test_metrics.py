"""Test MetricsCollector."""

from voicequant.server.metrics import CompressionMetrics, MetricsCollector


def test_collector_empty():
    c = MetricsCollector()
    assert c.last is None
    assert c.summary() == {}
    assert c.history == []


def test_collector_record_and_last():
    c = MetricsCollector()
    m = CompressionMetrics(seq_len=512, ratio=5.02, compress_time_ms=1.5)
    c.record(m)

    assert c.last == m
    assert len(c.history) == 1


def test_collector_summary():
    c = MetricsCollector()
    c.record(CompressionMetrics(ratio=5.0, compress_time_ms=1.0))
    c.record(CompressionMetrics(ratio=4.0, compress_time_ms=2.0))

    s = c.summary()
    assert s["count"] == 2
    assert s["avg_ratio"] == 4.5
    assert s["avg_compress_ms"] == 1.5


def test_history_is_copy():
    c = MetricsCollector()
    c.record(CompressionMetrics(ratio=5.0))
    h = c.history
    h.append(CompressionMetrics(ratio=1.0))
    assert len(c.history) == 1  # original unchanged
