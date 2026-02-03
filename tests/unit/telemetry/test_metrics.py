"""Tests for metrics collection."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from layerzero.telemetry.metrics import MetricsCollector


class TestMetricsCollectorCreation:
    """Test MetricsCollector creation."""

    def test_collector_instantiation(self) -> None:
        """MetricsCollector can be instantiated."""
        collector = MetricsCollector()
        assert collector is not None

    def test_collector_initial_state(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Fresh collector has zero counts."""
        assert metrics_collector.total_selections == 0
        assert metrics_collector.cache_hit_rate == 0.0
        assert len(metrics_collector.kernel_usage_counts) == 0


class TestMetricsRecording:
    """Test metrics recording."""

    def test_record_selection(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """record_selection increments counts."""
        metrics_collector.record_selection(
            kernel_id="flash_attn.fwd",
            latency_ns=1000,
            cache_hit=True,
        )
        assert metrics_collector.total_selections == 1

    def test_record_multiple_selections(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Multiple selections are recorded."""
        for i in range(10):
            metrics_collector.record_selection(
                kernel_id="flash_attn.fwd",
                latency_ns=1000 + i * 100,
                cache_hit=i % 2 == 0,
            )
        assert metrics_collector.total_selections == 10

    def test_record_different_kernels(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Different kernels are tracked separately."""
        metrics_collector.record_selection("kernel_a", 1000, True)
        metrics_collector.record_selection("kernel_b", 1000, True)
        metrics_collector.record_selection("kernel_a", 1000, True)

        counts = metrics_collector.kernel_usage_counts
        assert counts["kernel_a"] == 2
        assert counts["kernel_b"] == 1


class TestSelectionLatency:
    """Test selection latency tracking."""

    def test_selection_latency_tracked(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Selection latency is tracked."""
        latencies = populated_collector.selection_latencies
        assert len(latencies) > 0

    def test_selection_latency_histogram(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Selection latency histogram available."""
        histogram = populated_collector.selection_latency_histogram
        assert "p50" in histogram
        assert "p95" in histogram
        assert "p99" in histogram

    def test_latency_histogram_values(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Latency histogram values are reasonable."""
        histogram = populated_collector.selection_latency_histogram
        # p99 should be >= p95 >= p50
        assert histogram["p99"] >= histogram["p95"]
        assert histogram["p95"] >= histogram["p50"]
        assert histogram["p50"] > 0

    def test_latency_mean_and_std(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Mean and std deviation available."""
        histogram = populated_collector.selection_latency_histogram
        assert "mean" in histogram
        assert "std" in histogram
        assert histogram["mean"] > 0


class TestCacheHitRate:
    """Test cache hit rate tracking."""

    def test_cache_hit_rate_calculated(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Cache hit rate is calculated."""
        hit_rate = populated_collector.cache_hit_rate
        assert 0.0 <= hit_rate <= 1.0

    def test_cache_hit_rate_correct(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Cache hit rate calculation is correct."""
        # 3 hits, 1 miss = 75% hit rate
        metrics_collector.record_selection("a", 1000, cache_hit=True)
        metrics_collector.record_selection("a", 1000, cache_hit=True)
        metrics_collector.record_selection("a", 1000, cache_hit=True)
        metrics_collector.record_selection("a", 1000, cache_hit=False)

        assert metrics_collector.cache_hit_rate == 0.75

    def test_cache_hit_rate_zero_selections(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Cache hit rate is 0.0 with no selections."""
        assert metrics_collector.cache_hit_rate == 0.0


class TestKernelUsageCount:
    """Test kernel usage count tracking."""

    def test_kernel_usage_counts(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Kernel usage counts tracked."""
        counts = populated_collector.kernel_usage_counts
        assert isinstance(counts, dict)
        assert len(counts) > 0

    def test_kernel_usage_counts_correct(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Kernel usage counts are correct."""
        counts = populated_collector.kernel_usage_counts
        assert counts["flash_attn.fwd"] == 3
        assert counts["torch.sdpa"] == 2


class TestThreadSafety:
    """Test thread-safe metrics collection."""

    def test_concurrent_recording(
        self,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Concurrent recording is thread-safe."""
        num_threads = 10
        recordings_per_thread = 100

        def record_many() -> None:
            for i in range(recordings_per_thread):
                metrics_collector.record_selection(
                    kernel_id=f"kernel_{threading.current_thread().name}",
                    latency_ns=1000 + i,
                    cache_hit=i % 2 == 0,
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_many) for _ in range(num_threads)]
            for f in futures:
                f.result()

        # Should have recorded all without data loss
        expected_total = num_threads * recordings_per_thread
        assert metrics_collector.total_selections == expected_total


class TestMetricsReset:
    """Test metrics reset functionality."""

    def test_reset_clears_all(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Reset clears all metrics."""
        assert populated_collector.total_selections > 0

        populated_collector.reset()

        assert populated_collector.total_selections == 0
        assert populated_collector.cache_hit_rate == 0.0
        assert len(populated_collector.kernel_usage_counts) == 0

    def test_reset_allows_new_recording(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Reset allows new recordings."""
        populated_collector.reset()
        populated_collector.record_selection("new_kernel", 500, True)

        assert populated_collector.total_selections == 1
        assert "new_kernel" in populated_collector.kernel_usage_counts


class TestMetricsSnapshot:
    """Test metrics snapshot functionality."""

    def test_snapshot_returns_dict(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Snapshot returns dictionary."""
        snapshot = populated_collector.snapshot()
        assert isinstance(snapshot, dict)

    def test_snapshot_contains_all_metrics(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Snapshot contains all metric types."""
        snapshot = populated_collector.snapshot()
        assert "total_selections" in snapshot
        assert "cache_hit_rate" in snapshot
        assert "kernel_usage_counts" in snapshot
        assert "latency_histogram" in snapshot

    def test_snapshot_is_copy(
        self,
        populated_collector: MetricsCollector,
    ) -> None:
        """Snapshot is a copy, not a reference."""
        snapshot1 = populated_collector.snapshot()
        populated_collector.record_selection("new", 1000, True)
        snapshot2 = populated_collector.snapshot()

        assert snapshot1["total_selections"] != snapshot2["total_selections"]
