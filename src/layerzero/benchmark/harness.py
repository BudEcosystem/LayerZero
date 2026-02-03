"""
Benchmark Harness

Provides micro-benchmark harness for kernel performance testing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import torch

from layerzero.benchmark.stats import calculate_percentile

if TYPE_CHECKING:
    pass


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        warmup_iters: Number of warmup iterations (not timed).
        timed_iters: Number of timed iterations.
        sync_cuda: Whether to synchronize CUDA before/after timing.
        device: Device to run benchmarks on.
    """

    warmup_iters: int = 10
    timed_iters: int = 100
    sync_cuda: bool = True
    device: str = "cuda"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes:
        latencies_ns: List of latencies in nanoseconds.
        median_ns: Median latency in nanoseconds.
        p95_ns: 95th percentile latency in nanoseconds.
        p99_ns: 99th percentile latency in nanoseconds.
        mean_ns: Mean latency in nanoseconds.
        std_ns: Standard deviation of latency in nanoseconds.
        min_ns: Minimum latency in nanoseconds.
        max_ns: Maximum latency in nanoseconds.
    """

    latencies_ns: list[int]
    median_ns: int
    p95_ns: int
    p99_ns: int
    mean_ns: float
    std_ns: float
    min_ns: int
    max_ns: int

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "latencies_ns": self.latencies_ns,
            "median_ns": self.median_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "mean_ns": self.mean_ns,
            "std_ns": self.std_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        """Create result from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            BenchmarkResult instance.
        """
        return cls(
            latencies_ns=data["latencies_ns"],
            median_ns=data["median_ns"],
            p95_ns=data["p95_ns"],
            p99_ns=data["p99_ns"],
            mean_ns=data["mean_ns"],
            std_ns=data["std_ns"],
            min_ns=data["min_ns"],
            max_ns=data["max_ns"],
        )

    def throughput(self, batch_size: int, seq_len: int) -> float:
        """Calculate throughput in tokens/second.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Throughput in tokens per second.
        """
        tokens_per_iter = batch_size * seq_len
        seconds_per_iter = self.median_ns / 1e9
        return tokens_per_iter / seconds_per_iter


class BenchmarkHarness:
    """Harness for running micro-benchmarks.

    Provides warmup iterations, GPU synchronization, and
    statistical analysis of results.

    Example:
        ```python
        config = BenchmarkConfig(warmup_iters=10, timed_iters=100)
        harness = BenchmarkHarness(config)

        result = harness.run(lambda: model(input))
        print(f"Median: {result.median_ns / 1e6:.2f} ms")
        ```
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark harness.

        Args:
            config: Benchmark configuration.
        """
        self.config = config

    def run(self, fn: Callable[[], Any]) -> BenchmarkResult:
        """Run benchmark on a function.

        Args:
            fn: Function to benchmark (should take no arguments).

        Returns:
            BenchmarkResult with timing statistics.
        """
        # Run warmup
        self.warmup(fn)

        # Run timed iterations
        latencies_ns = self.time_iterations(fn)

        # Calculate statistics
        latencies_float = [float(x) for x in latencies_ns]
        median_ns = int(calculate_percentile(latencies_float, 50))
        p95_ns = int(calculate_percentile(latencies_float, 95))
        p99_ns = int(calculate_percentile(latencies_float, 99))

        mean_ns = sum(latencies_ns) / len(latencies_ns)
        variance = sum((x - mean_ns) ** 2 for x in latencies_ns) / len(latencies_ns)
        std_ns = variance ** 0.5

        return BenchmarkResult(
            latencies_ns=latencies_ns,
            median_ns=median_ns,
            p95_ns=p95_ns,
            p99_ns=p99_ns,
            mean_ns=mean_ns,
            std_ns=std_ns,
            min_ns=min(latencies_ns),
            max_ns=max(latencies_ns),
        )

    def warmup(self, fn: Callable[[], Any]) -> None:
        """Run warmup iterations.

        Args:
            fn: Function to run for warmup.
        """
        for _ in range(self.config.warmup_iters):
            fn()

        # Sync CUDA after warmup
        if self.config.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def time_iterations(self, fn: Callable[[], Any]) -> list[int]:
        """Time iterations and return latencies.

        Args:
            fn: Function to time.

        Returns:
            List of latencies in nanoseconds.
        """
        latencies: list[int] = []

        for _ in range(self.config.timed_iters):
            # Sync before timing
            if self.config.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter_ns()
            fn()

            # Sync after timing
            if self.config.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter_ns()
            latencies.append(end - start)

        return latencies


def benchmark(
    fn: Callable[[], Any],
    warmup_iters: int = 10,
    timed_iters: int = 100,
    sync_cuda: bool = True,
) -> BenchmarkResult:
    """Convenience function for running a single benchmark.

    Args:
        fn: Function to benchmark.
        warmup_iters: Number of warmup iterations.
        timed_iters: Number of timed iterations.
        sync_cuda: Whether to synchronize CUDA.

    Returns:
        BenchmarkResult with timing statistics.
    """
    config = BenchmarkConfig(
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        sync_cuda=sync_cuda,
    )
    harness = BenchmarkHarness(config)
    return harness.run(fn)
