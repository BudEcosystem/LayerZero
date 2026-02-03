"""
Metrics Collection

Thread-safe metrics collection for kernel selection telemetry.
Tracks selection latency, cache hit rates, and per-kernel usage.
"""
from __future__ import annotations

import statistics
import threading
from collections import defaultdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class MetricsCollector:
    """Thread-safe metrics collection for kernel selection.

    Collects and aggregates metrics about kernel selection
    including latency, cache hits, and per-kernel usage counts.

    Thread Safety:
        All methods are thread-safe and can be called concurrently
        from multiple threads.

    Example:
        ```python
        collector = MetricsCollector()

        # Record selections
        collector.record_selection("flash_attn.fwd", 1000, cache_hit=True)
        collector.record_selection("torch.sdpa", 2000, cache_hit=False)

        # Get metrics
        print(f"Hit rate: {collector.cache_hit_rate:.2%}")
        print(f"P99 latency: {collector.selection_latency_histogram['p99']}ns")
        ```
    """

    def __init__(self) -> None:
        """Initialize the collector."""
        self._lock = threading.Lock()
        self._total_selections: int = 0
        self._cache_hits: int = 0
        self._latencies: list[int] = []
        self._kernel_counts: dict[str, int] = defaultdict(int)

    def record_selection(
        self,
        kernel_id: str,
        latency_ns: int,
        cache_hit: bool,
    ) -> None:
        """Record a kernel selection event.

        Args:
            kernel_id: ID of the selected kernel.
            latency_ns: Selection latency in nanoseconds.
            cache_hit: Whether this was a cache hit.
        """
        with self._lock:
            self._total_selections += 1
            self._latencies.append(latency_ns)
            self._kernel_counts[kernel_id] += 1
            if cache_hit:
                self._cache_hits += 1

    @property
    def total_selections(self) -> int:
        """Get total number of selections recorded."""
        with self._lock:
            return self._total_selections

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0).

        Returns:
            Cache hit rate, or 0.0 if no selections recorded.
        """
        with self._lock:
            if self._total_selections == 0:
                return 0.0
            return self._cache_hits / self._total_selections

    @property
    def kernel_usage_counts(self) -> dict[str, int]:
        """Get per-kernel usage counts.

        Returns:
            Dictionary mapping kernel ID to selection count.
        """
        with self._lock:
            return dict(self._kernel_counts)

    @property
    def selection_latencies(self) -> list[int]:
        """Get all recorded latencies.

        Returns:
            List of latencies in nanoseconds.
        """
        with self._lock:
            return list(self._latencies)

    @property
    def selection_latency_histogram(self) -> dict[str, float]:
        """Get latency histogram with percentiles.

        Returns:
            Dictionary with p50, p95, p99, mean, std, min, max.
        """
        with self._lock:
            if not self._latencies:
                return {
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }

            sorted_latencies = sorted(self._latencies)
            n = len(sorted_latencies)

            def percentile(p: float) -> float:
                idx = int(n * p / 100)
                idx = min(idx, n - 1)
                return float(sorted_latencies[idx])

            mean = statistics.mean(self._latencies)
            std = statistics.stdev(self._latencies) if n > 1 else 0.0

            return {
                "p50": percentile(50),
                "p95": percentile(95),
                "p99": percentile(99),
                "mean": mean,
                "std": std,
                "min": float(min(self._latencies)),
                "max": float(max(self._latencies)),
            }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._total_selections = 0
            self._cache_hits = 0
            self._latencies.clear()
            self._kernel_counts.clear()

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of all metrics.

        Returns:
            Dictionary containing all metrics.
        """
        with self._lock:
            return {
                "total_selections": self._total_selections,
                "cache_hits": self._cache_hits,
                "cache_hit_rate": (
                    self._cache_hits / self._total_selections
                    if self._total_selections > 0 else 0.0
                ),
                "kernel_usage_counts": dict(self._kernel_counts),
                "latency_histogram": self._compute_histogram_unsafe(),
                "latency_count": len(self._latencies),
            }

    def _compute_histogram_unsafe(self) -> dict[str, float]:
        """Compute histogram without lock (internal use only)."""
        if not self._latencies:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            return float(sorted_latencies[idx])

        mean = statistics.mean(self._latencies)
        std = statistics.stdev(self._latencies) if n > 1 else 0.0

        return {
            "p50": percentile(50),
            "p95": percentile(95),
            "p99": percentile(99),
            "mean": mean,
            "std": std,
            "min": float(min(self._latencies)),
            "max": float(max(self._latencies)),
        }


# Global metrics collector instance
_global_collector: MetricsCollector | None = None
_global_lock = threading.Lock()


def get_global_collector() -> MetricsCollector:
    """Get the global metrics collector.

    Creates a new collector if one doesn't exist.

    Returns:
        Global MetricsCollector instance.
    """
    global _global_collector
    with _global_lock:
        if _global_collector is None:
            _global_collector = MetricsCollector()
        return _global_collector


def reset_global_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    with _global_lock:
        if _global_collector is not None:
            _global_collector.reset()
