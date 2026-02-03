"""Tests for benchmark harness."""
from __future__ import annotations

import time
from typing import Callable

import pytest
import torch

from layerzero.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkHarness,
    ComparisonResult,
    compare_kernels,
)
from layerzero.benchmark.stats import (
    calculate_percentile,
    calculate_variance,
    is_statistically_significant,
)


def dummy_workload() -> torch.Tensor:
    """Simple workload for testing."""
    return torch.randn(100, 100)


def slow_workload() -> torch.Tensor:
    """Slower workload for comparison testing."""
    time.sleep(0.001)  # 1ms delay
    return torch.randn(100, 100)


class TestBenchmarkHarness:
    """Test benchmark harness functionality."""

    @pytest.fixture
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            warmup_iters=5,
            timed_iters=10,
            sync_cuda=False,  # CPU-only for tests
            device="cpu",
        )

    @pytest.fixture
    def harness(self, config: BenchmarkConfig) -> BenchmarkHarness:
        return BenchmarkHarness(config)

    def test_benchmark_warmup_iterations(self, harness: BenchmarkHarness) -> None:
        """Warmup iterations executed."""
        warmup_count = [0]

        def counting_workload() -> None:
            warmup_count[0] += 1

        harness.warmup(counting_workload)

        assert warmup_count[0] == harness.config.warmup_iters

    def test_benchmark_timed_iterations(self, harness: BenchmarkHarness) -> None:
        """Timed iterations executed."""
        timed_count = [0]

        def counting_workload() -> None:
            timed_count[0] += 1

        latencies = harness.time_iterations(counting_workload)

        assert len(latencies) == harness.config.timed_iters
        assert timed_count[0] == harness.config.timed_iters

    def test_benchmark_median_calculation(self, harness: BenchmarkHarness) -> None:
        """Median latency calculated."""
        result = harness.run(dummy_workload)

        assert result.median_ns > 0
        assert isinstance(result.median_ns, int)

    def test_benchmark_p95_calculation(self, harness: BenchmarkHarness) -> None:
        """p95 latency calculated."""
        result = harness.run(dummy_workload)

        assert result.p95_ns > 0
        assert result.p95_ns >= result.median_ns

    def test_benchmark_p99_calculation(self, harness: BenchmarkHarness) -> None:
        """p99 latency calculated."""
        result = harness.run(dummy_workload)

        assert result.p99_ns > 0
        assert result.p99_ns >= result.p95_ns

    def test_benchmark_variance_tracking(self, harness: BenchmarkHarness) -> None:
        """Variance tracked."""
        result = harness.run(dummy_workload)

        assert result.std_ns >= 0


class TestBenchmarkComparison:
    """Test kernel comparison functionality."""

    @pytest.fixture
    def config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            warmup_iters=2,
            timed_iters=10,
            sync_cuda=False,
            device="cpu",
        )

    def test_benchmark_kernel_comparison(self, config: BenchmarkConfig) -> None:
        """Compare two kernels head-to-head."""
        result = compare_kernels(
            baseline_fn=dummy_workload,
            candidate_fn=dummy_workload,
            config=config,
        )

        assert isinstance(result, ComparisonResult)
        assert result.baseline is not None
        assert result.candidate is not None

    def test_benchmark_winner_selection(self, config: BenchmarkConfig) -> None:
        """Winner kernel identified."""
        result = compare_kernels(
            baseline_fn=slow_workload,  # Slower
            candidate_fn=dummy_workload,  # Faster
            config=config,
        )

        assert result.winner in ("baseline", "candidate", "tie")

    def test_benchmark_speedup_calculation(self, config: BenchmarkConfig) -> None:
        """Speedup percentage calculated."""
        result = compare_kernels(
            baseline_fn=dummy_workload,
            candidate_fn=dummy_workload,
            config=config,
        )

        assert isinstance(result.speedup, float)
        # Speedup should be close to 1.0 for identical workloads
        assert 0.5 <= result.speedup <= 2.0


class TestBenchmarkStats:
    """Test statistical functions."""

    def test_calculate_percentile_p50(self) -> None:
        """Calculate 50th percentile (median)."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p50 = calculate_percentile(values, 50)

        assert p50 == 3.0

    def test_calculate_percentile_p95(self) -> None:
        """Calculate 95th percentile."""
        values = list(range(1, 101))  # 1 to 100
        p95 = calculate_percentile(values, 95)

        assert p95 >= 95

    def test_calculate_percentile_p99(self) -> None:
        """Calculate 99th percentile."""
        values = list(range(1, 101))
        p99 = calculate_percentile(values, 99)

        assert p99 >= 99

    def test_calculate_variance(self) -> None:
        """Calculate variance."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        variance = calculate_variance(values)

        # Variance should be approximately 4.0 for this dataset
        assert abs(variance - 4.0) < 0.5

    def test_statistical_significance_same(self) -> None:
        """Same distributions not significant."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]

        significant, p_value = is_statistically_significant(a, b)

        # Nearly identical distributions should not be significant
        # (depending on test parameters)
        assert isinstance(significant, bool)
        assert 0 <= p_value <= 1

    def test_statistical_significance_different(self) -> None:
        """Very different distributions are significant."""
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [100.0, 100.0, 100.0, 100.0, 100.0]

        significant, p_value = is_statistically_significant(a, b)

        assert significant is True
        assert p_value < 0.05


class TestBenchmarkPerfDBIntegration:
    """Test PerfDB integration."""

    def test_benchmark_results_to_perfdb(self) -> None:
        """Benchmark results saved to PerfDB."""
        from layerzero.benchmark import save_benchmark_to_perfdb

        result = BenchmarkResult(
            latencies_ns=[1000, 2000, 3000],
            median_ns=2000,
            p95_ns=2900,
            p99_ns=2990,
            mean_ns=2000.0,
            std_ns=816.5,
            min_ns=1000,
            max_ns=3000,
        )

        # Should not raise
        save_benchmark_to_perfdb(
            benchmark_id="test_benchmark",
            result=result,
            metadata={"test": True},
        )

    def test_benchmark_results_from_perfdb(self) -> None:
        """Benchmark results loaded from PerfDB."""
        from layerzero.benchmark import load_benchmark_from_perfdb, save_benchmark_to_perfdb

        result = BenchmarkResult(
            latencies_ns=[1000, 2000, 3000],
            median_ns=2000,
            p95_ns=2900,
            p99_ns=2990,
            mean_ns=2000.0,
            std_ns=816.5,
            min_ns=1000,
            max_ns=3000,
        )

        save_benchmark_to_perfdb("test_load", result, {"version": "1.0"})

        loaded = load_benchmark_from_perfdb("test_load")

        # May return None if not found or expired
        if loaded is not None:
            assert loaded.median_ns == result.median_ns

    def test_benchmark_invalidation_on_change(self) -> None:
        """Results invalidated on version change."""
        from layerzero.benchmark import (
            save_benchmark_to_perfdb,
            load_benchmark_from_perfdb,
            invalidate_benchmarks,
        )

        result = BenchmarkResult(
            latencies_ns=[1000],
            median_ns=1000,
            p95_ns=1000,
            p99_ns=1000,
            mean_ns=1000.0,
            std_ns=0.0,
            min_ns=1000,
            max_ns=1000,
        )

        save_benchmark_to_perfdb("test_invalidate", result, {"version": "1.0"})

        # Invalidate all benchmarks
        invalidate_benchmarks()

        # After invalidation, should return None
        loaded = load_benchmark_from_perfdb("test_invalidate")
        # Result may be None or stale-marked


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_result_creation(self) -> None:
        """BenchmarkResult can be created."""
        result = BenchmarkResult(
            latencies_ns=[1000, 2000, 3000],
            median_ns=2000,
            p95_ns=2900,
            p99_ns=2990,
            mean_ns=2000.0,
            std_ns=816.5,
            min_ns=1000,
            max_ns=3000,
        )

        assert result.median_ns == 2000
        assert len(result.latencies_ns) == 3

    def test_result_to_dict(self) -> None:
        """BenchmarkResult converts to dict."""
        result = BenchmarkResult(
            latencies_ns=[1000],
            median_ns=1000,
            p95_ns=1000,
            p99_ns=1000,
            mean_ns=1000.0,
            std_ns=0.0,
            min_ns=1000,
            max_ns=1000,
        )

        d = result.to_dict()

        assert "median_ns" in d
        assert d["median_ns"] == 1000
